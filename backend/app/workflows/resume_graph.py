import re
import logging
import time
import json as py_json
from typing import TypedDict, Annotated, List, Literal, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from ..core.llm import get_analyst_llm, get_editor_llm, get_arbitrator_llm
from ..core.config import settings
from ..core.supabase import SupabaseService
from ..utils.text_cleaners import (
    extract_text_from_latex,
    clean_resume_response,
)
from ..utils.web_scraper import get_url_content_from_tavily
from ..utils.token_utils import estimate_tokens
from ..utils.timing import Timer
from ..core.exceptions import SystemFailure

logger = logging.getLogger(__name__)
logger.info("Resume Graph initialized.")

analyst_llm = get_analyst_llm()
editor_llm = get_editor_llm()
arbitrator_llm = get_arbitrator_llm()


class ResumeOptimizationState(TypedDict):
    messages: Annotated[List[dict], lambda x, y: x + y]
    job_description_raw: str
    job_description_text: str
    resume_raw_content: str
    resume_format: Literal["auto", "markdown", "pdf", "docx"]
    resume_plain_text: str
    extracted_keywords: List[str]
    analysis_report: str
    edited_resume_content: str
    human_feedback: str
    next_agent: str
    task_complete: bool
    current_task: str
    old_ats_score: Optional[int]
    new_ats_score: Optional[int]
    cover_letter_text: str
    cover_letter_markdown: str
    cover_letter_analysis: str
    services_requested: List[str]
    memory_context: Optional[dict]
    reflection_report: str
    user_id: Optional[str]
    self_correction_count: int
    # Council of Agents fields
    editor_proposals: Optional[List[dict]]
    winning_proposal_index: Optional[int]
    # Phase 4: JSON-first output
    resume_json: Optional[dict]
    # Phase 6: HITL + RAG
    hitl_feedback: Optional[str]
    vault_context: Optional[str]


class _SimpleResp:
    def __init__(self, content):
        self.content = content


def _safe_invoke(target, *args, **kwargs):
    """Call `target.invoke(...)` if available, otherwise call `target(...)` if callable.
    Throws RuntimeError on failure instead of swallowing error into a string.
    """
    try:
        fn = getattr(target, "invoke", None)
        if callable(fn):
            resp = fn(*args, **kwargs)
        elif callable(target):
            resp = target(*args, **kwargs)
        else:
            raise RuntimeError(f"Cannot call target of type {type(target)!r}")

        if isinstance(resp, str):
            return _SimpleResp(resp)
        if hasattr(resp, "content"):
            return resp
        # Fallback: coerce to string
        return _SimpleResp(str(resp))

    except Exception as e:
        logger.error(f"Error invoking target: {e}")
        raise RuntimeError(f"LLM Call Failed: {e}")


def emit(event: str, payload: dict | None = None) -> dict:
    """
    Helper to emit SSE-friendly deltas.
    These are meant for streaming to the client, NOT for internal logic.
    """
    return {"_event": event, "_payload": payload or {}}


### Core Workflow Nodes ###


def ingestion_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description_raw = state["job_description_raw"]
    resume_raw_content = state["resume_raw_content"]
    resume_format = state["resume_format"]
    user_id = state.get("user_id", "default_user")

    messages.append(HumanMessage(content=f"Starting ingestion process for user: {user_id}.").model_dump())
    messages.append(
        AIMessage(
            content="Node: `ingestion_node` - Processing raw inputs."
        ).model_dump()
    )

    # Fetch latest resume from Supabase for comparison/context
    latest_version = SupabaseService.get_latest_resume(user_id)
    if latest_version:
        messages.append(
            AIMessage(
                content=f"Sub-task: Found previous resume version (ATS Score: {latest_version.get('ats_score')}%). Using as historical context."
            ).model_dump()
        )
        # We can store this in state if we want nodes to explicitly see it
        memory_context = state.get("memory_context", {})
        memory_context["latest_resume_content"] = latest_version.get("content")
        memory_context["latest_ats_score"] = latest_version.get("ats_score")
        state["memory_context"] = memory_context

    job_description_text = ""
    if job_description_raw.startswith("http"):
        messages.append(
            AIMessage(
                content=f"Sub-task: Scraping job description from URL: {job_description_raw} using Tavily."
            ).model_dump()
        )
        scraped_content = get_url_content_from_tavily(job_description_raw)
        if "Error" in scraped_content or "No content found" in scraped_content:
            messages.append(
                AIMessage(
                    content=f"Warning: Failed to scrape URL with Tavily. Using raw input as fallback. Error: {scraped_content}"
                ).model_dump()
            )
            job_description_text = job_description_raw
        else:
            job_description_text = scraped_content
            messages.append(
                AIMessage(
                    content="Sub-task: Successfully scraped job description content."
                ).model_dump()
            )
    else:
        job_description_text = job_description_raw
        messages.append(
            AIMessage(
                content="Sub-task: Using provided job description text directly."
            ).model_dump()
        )

    resume_plain_text = ""
    # Normalize resume_format
    fmt = (resume_format or "auto").lower().strip()

    if fmt == "auto":
        # quick heuristic: if it looks like LaTeX source, use the LaTeX extractor
        sample = (resume_raw_content or "")[:2000]
        if (
            "\\begin{" in sample
            or "\\documentclass" in sample
            or re.search(r"\\[a-zA-Z]+\{", sample)
        ):
            messages.append(
                AIMessage(
                    content="Sub-task: Auto-detected LaTeX content. Using LaTeX extractor."
                ).model_dump()
            )
            resume_plain_text = (
                extract_text_from_latex.invoke({"latex_content": resume_raw_content})
                if hasattr(extract_text_from_latex, "invoke")
                else extract_text_from_latex(resume_raw_content)
            )
        else:
            messages.append(
                AIMessage(
                    content="Sub-task: Auto-detected plain text resume. Using plain text."
                ).model_dump()
            )
            resume_plain_text = resume_raw_content

    elif fmt in ("plain", "pdf", "docx", "doc"):
        # For these formats we assume the uploaded file was already converted to plain text
        messages.append(
            AIMessage(
                content=f"Sub-task: Treating resume format '{fmt}' as plain text (already extracted if uploaded)."
            ).model_dump()
        )
        resume_plain_text = resume_raw_content

    elif fmt == "markdown":
        # Safe normalization
        if isinstance(resume_raw_content, dict):
            resume_raw_content = resume_raw_content.get(
                "md_content", ""
            ) or resume_raw_content.get("text", "")

        if not isinstance(resume_raw_content, str):
            resume_raw_content = str(resume_raw_content)

        # For markdown, we keep it as is because it's already a high-signal structured format
        # that LLMs understand better than flattened text.
        resume_plain_text = resume_raw_content

    else:
        messages.append(
            AIMessage(
                content=f"Warning: Unsupported resume format '{resume_format}'. Treating as plain text."
            ).model_dump()
        )
        resume_plain_text = resume_raw_content

    messages.append(
        SystemMessage(
            content="Node: `ingestion_node` - Job description and resume ingested and converted to plain text."
        ).model_dump()
    )
    return {
        **state,
        "job_description_text": job_description_text,
        "resume_plain_text": resume_plain_text,
        "messages": messages,
        "next_agent": "keyword_extraction",
        "current_task": "Extracting keywords",
        "edited_resume_content": "",
        "cover_letter_text": "",
        "cover_letter_markdown": "",
        "analysis_report": "",
        "reflection_report": "",
        "old_ats_score": None,
        "new_ats_score": None,
        "self_correction_count": 0,
        **emit(
            event="token_diagnostics",
            payload={
                "job_description_tokens": estimate_tokens(job_description_text),
                "resume_tokens": estimate_tokens(resume_plain_text),
            },
        ),
    }


def keyword_extraction_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description = state["job_description_text"]

    messages.append(
        HumanMessage(
            content="Node: `keyword_extraction_node` - Initiating keyword extraction from job description."
        ).model_dump()
    )

    prompt = (
        "TASK: Extract high-impact ATS keywords from this job description (15-25 total).\n\n"
        "OUTPUT FORMAT:\n"
        "Comma-separated keywords, most important first. NO preamble, NO explanation.\n\n"
        "KEYWORD TYPES - INCLUDE:\n"
        "- Technical Languages: Python 3.8+, TypeScript, Java, C++\n"
        "- Frameworks: React, FastAPI, Django, Spring, Vue\n"
        "- Cloud & DevOps: AWS, Azure, Docker, Kubernetes, CI/CD\n"
        "- Databases: PostgreSQL, MongoDB, Redis, Elasticsearch\n"
        "- Methodologies: Agile, Scrum, TDD, Microservices\n"
        "- Certifications: AWS Solutions Architect, Google Cloud, Salesforce\n\n"
        "STRICTLY EXCLUDE (NEVER include):\n"
        "- Soft skills: communication, leadership, teamwork, motivation\n"
        "- Vague adjectives: strong, experienced, excellent, proven\n"
        "- Business jargon: solutions, services, innovation, synergy\n"
        "- Company/location names: [specific company names], cities\n"
        "- Generic metrics: years of experience, performance\n\n"
        "SELECTION RULES:\n"
        "1. Pick most specific term: 'Machine Learning' > 'ML' + 'Artificial Intelligence'\n"
        "2. Use exact phrases from JD: copy-paste preferred\n"
        "3. Only standard abbreviations: AWS (yes), SK (no unless stated)\n"
        "4. Order by: Seniority requirement → Specificity → Frequency in JD\n\n"
        f"{job_description}\n\n"
        "KEYWORDS (comma-separated, most important first):"
    )
    messages.append(
        AIMessage(
            content=f"Sub-task: Sending prompt to LLM for keyword extraction. Prompt snippet: '{prompt[:100]}...'"
        ).model_dump()
    )
    logger.info("[LLM] Call started: keyword_extraction")
    try:
        with Timer("llm_invoke for keyword_extraction"):
            response = _safe_invoke(analyst_llm, prompt)
    except Exception as e:
        logger.warning(f"Analyst model failed for keyword_extraction, falling back to Editor: {e}")
        try:
            with Timer("llm_invoke fallback for keyword_extraction"):
                response = _safe_invoke(editor_llm, prompt)
        except Exception as e2:
            logger.exception("[LLM] Both models failed: keyword_extraction")
            raise SystemFailure(message="Keyword extraction failed", details={"reason": str(e2)})
    logger.info("[LLM] Call completed: keyword_extraction")
    keywords = [kw.strip() for kw in response.content.split(",") if kw.strip()]

    messages.append(
        AIMessage(
            content=f"Sub-task: LLM extracted keywords: {', '.join(keywords)}"
        ).model_dump()
    )
    messages.append(
        SystemMessage(
            content="Node: `keyword_extraction_node` - Keywords extracted successfully."
        ).model_dump()
    )
    return {
        **state,
        "extracted_keywords": keywords,
        "messages": messages,
        "next_agent": "resume_analysis",
        "current_task": "Analyzing resume",
        **emit(
            event="keywords_extracted",
            payload={"count": len(keywords), "preview": keywords[:5]},
        ),
    }


def resume_analysis_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description = state.get("job_description_text", "")
    resume_text = state.get("resume_plain_text", "")
    keywords = state.get("extracted_keywords", [])
    old_ats_score = None

    messages.append(
        HumanMessage(
            content="Node: `resume_analysis_node` - Starting resume analysis against job description and keywords."
        ).model_dump()
    )

    prompt = (
        "TASK: Analyze resume's ATS match to target job. Score 0-100%.\n\n"
        "ANALYSIS RULES - DO:\n"
        "- Count exact target keywords + reasonable variations (e.g., 'Python 3.8' matches 'Python')\n"
        "- Map keywords to resume sections (Title, Summary, Bullets, Skills)\n"
        "- Assess section structure: clear headings, logical flow\n"
        "- Identify legitimate rephrasing/reordering opportunities only\n\n"
        "ANALYSIS RULES - NEVER:\n"
        "- Suggest adding any skills/tech NOT explicitly in resume\n"
        "- Recommend new certs, degrees, or experience not present\n"
        "- Propose any fabrication of metrics, dates, or companies\n"
        "- Add soft-skill language (teamwork, communication, etc.)\n\n"
        "SCORING METHODOLOGY (Transparent Breakdown):\n"
        "- Keyword Coverage (40%): Found / Total Target Keywords\n"
        "- Keyword Placement (30%): Prominence of keywords (title > summary > bullets)\n"
        "- Relevance Match (20%): Skills/experience align with JD responsibilities\n"
        "- ATS Structure (10%): Clear sections, no formatting tricks, readable format\n\n"
        "OUTPUT FORMAT (EXACT STRUCTURE):\n"
        "**ATS Score: [XX]%** (e.g., **ATS Score: 62%**)\n"
        "**Found Keywords:** [comma-separated list, in order of importance]\n"
        "**Missing Keywords:** [comma-separated list]\n"
        "**Legitimate Improvements:** [bullets with ONLY rephrasing/reordering suggestions]\n\n"
        f"Target Keywords: {', '.join(keywords)}\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Job Description:\n{job_description}\n\n"
        "START ANALYSIS:"
    )
    messages.append(
        AIMessage(
            content=f"Sub-task: Sending prompt to LLM for initial resume analysis. Prompt snippet: '{prompt[:100]}...'"
        ).model_dump()
    )
    logger.info("[LLM] Call started: resume_analysis")
    # Smart Stagger for Rate Limits
    time.sleep(1.0)
    try:
        with Timer("llm_invoke for resume_analysis"):
            response = _safe_invoke(analyst_llm, prompt)
    except Exception as e:
        logger.warning(f"Analyst model failed for resume_analysis, falling back to Editor: {e}")
        try:
            with Timer("llm_invoke fallback for resume_analysis"):
                response = _safe_invoke(editor_llm, prompt)
        except Exception as e2:
            logger.exception("[LLM] Both models failed: resume_analysis")
            raise SystemFailure(message="Resume analysis failed", details={"reason": str(e2)})
    logger.info("[LLM] Call completed: resume_analysis")
    analysis_report = response.content

    # Debug: Log the raw response for score extraction debugging
    logger.info(f"[DEBUG] Resume Analysis Response (first 500 chars): {analysis_report[:500]}")

    # More flexible regex to catch various score formats
    # Matches: "ATS Score: 85%", "Score: 85", "**ATS Score: 85%**", etc.
    score_match = re.search(r"(?:ATS\s*)?Score\D*?(\d+)\s*%?", analysis_report, re.IGNORECASE)

    if score_match:
        old_ats_score = int(score_match.group(1))
        logger.info(f"[DEBUG] Extracted old_ats_score: {old_ats_score}")
        messages.append(
            AIMessage(
                content=f"Sub-task: Estimated Original ATS Score: {old_ats_score}%"
            ).model_dump()
        )
    else:
        logger.warning(f"[DEBUG] Failed to extract ATS score. Response snippet: {analysis_report[:200]}")
        messages.append(
            AIMessage(
                content="Sub-task: Could not parse original ATS Score from LLM response."
            ).model_dump()
        )

    messages.append(
        AIMessage(
            content=f"Sub-task: Initial resume analysis report generated: \n{analysis_report[:500]}..."
        ).model_dump()
    )  # Truncate for log
    messages.append(
        SystemMessage(
            content="Node: `resume_analysis_node` - Resume analysis completed. Moving to human review."
        ).model_dump()
    )
    return {
        **state,
        "analysis_report": analysis_report,
        "messages": messages,
        "old_ats_score": old_ats_score,
        "next_agent": "human_review",  # This is just a label for the current agent's intention
        "current_task": "Awaiting human review (automated)",
        **emit(
            event="resume_analyzed",
            payload={
                "old_ats_score": old_ats_score,
                "summary_preview": analysis_report[:200],
            },
        ),
    }


def human_review_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    """
    Phase 6.2 HITL: Pauses the workflow using LangGraph interrupt().
    The analysis report is sent to the client for review.
    The user can provide custom feedback to guide the editor,
    or simply 'proceed' to continue with default behavior.
    """
    messages = state["messages"]
    analysis_report = state["analysis_report"]
    old_ats_score = state.get("old_ats_score")
    keywords = state.get("extracted_keywords", [])

    messages.append(
        AIMessage(
            content="Node: `human_review_node` - Analysis report ready for human review."
        ).model_dump()
    )

    # --- LangGraph HITL Interrupt ---
    # This pauses the graph execution and returns data to the caller.
    # The caller (background_runner) catches GraphInterrupt and stores
    # the analysis report in the job status for the frontend to display.
    human_response = interrupt({
        "analysis_report": analysis_report,
        "old_ats_score": old_ats_score,
        "extracted_keywords": keywords[:15],
        "message": "Analysis is complete. Review the report and provide feedback to guide the optimization, or click 'Proceed' to continue.",
    })

    # After resume: human_response is the feedback string from the user
    feedback_text = human_response if isinstance(human_response, str) else "proceed"
    logger.info(f"[HITL] Received human feedback: {feedback_text[:100]}")

    messages.append(
        SystemMessage(
            content=f"Node: `human_review_node` - Human feedback received: '{feedback_text[:200]}'"
        ).model_dump()
    )

    return {
        **state,
        "human_feedback": feedback_text,
        "hitl_feedback": feedback_text if feedback_text.lower() != "proceed" else None,
        "messages": messages,
        "current_task": "Processing human feedback",
    }


def resume_editing_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    resume_text = state["resume_plain_text"]
    human_feedback = state["human_feedback"]
    memory_context = state.get("memory_context", {})
    latest_resume = memory_context.get("latest_resume_content")
    latest_score = memory_context.get("latest_ats_score")

    messages.append(
        HumanMessage(
            content="Node: `resume_editing_node` - Generating professionally enhanced version of the resume."
        ).model_dump()
    )

    history_context = ""
    if latest_resume and latest_score:
        history_context = f"""
=== HISTORICAL CONTEXT (PREVIOUS BEST VERSION) ===
Previous ATS Score: {latest_score}%
Previous Content:
{latest_resume[:1000]}... [truncated]

GOAL: Your goal is to EXCEED the previous score by refining the content further without over-optimizing. Identify what worked in the last version and build upon it.
"""

    job_description = state.get("job_description_text", "")
    editing_instructions = f"""TASK: Act as an expert Senior Technical Recruiter. Optimize the provided resume for ATS matching against the specific Job Description below.

{history_context}

=== JOB DESCRIPTION ===
{job_description}

=== ABSOLUTE CONSTRAINTS (NEVER violate) ===
NEVER:
1. Add ANY skills, tools, or technologies to a specific job/project that were NOT originally in that specific section of the resume. 
2. Bleed information between sections: Do not move tech stack from "Projects" to "Experience" or vice versa unless it was originally present in both.
3. Invent or modify dates, company names, contact info, or locations.
4. Fabricate metrics, numbers, or specific accomplishments.
5. Create new projects or experience items.

CRITICAL RULE - NO SECTION BLEEDING:
Each job in "Professional Experience" must ONLY describe work done AT THAT COMPANY.
Do NOT add project descriptions or personal project tech to job experience bullets.
Keep "Projects" and "Professional Experience" strictly separate.

ALWAYS:
1. Preserve ALL original facts (dates, titles, companies unchanged).
2. ONLY REPHRASE: You may improve the prose and verb-strength of EXISTING descriptions.
3. Maintain 100% factual fidelity to the source section.
4. REJECT any suggestion to "borrow" skills from one job to benefit another.

=== ALLOWED OPTIMIZATION TECHNIQUES (4 ONLY) ===

[1] PERSUASIVE REPHRASING (SOURCE-ONLY)
  - Transform passive duties into achievement-oriented results using EXISTING facts.
  - Use high-impact verbs: "Engineered", "Optimized", "Spearheaded", "Architected", "Automated".
  - DO NOT add new outcomes. Only rephrase what is already there.

[2] STRATEGIC KEYWORD ALIGNMENT
  - Integrate target keywords from the JD into the Professional Summary ONLY IF you actually possess those skills (as per the whole resume).
  - In experience bullets, only use keywords that were originally present in that specific job.

[3] GENERALIZED SUMMARY
  - Rewrite the Professional Summary to be a bridge between the JD and your actual resume.
  - It should be professional, slightly generalized to cover your career trajectory, but grounded in resume facts.

[4] CONCISION & CLARITY
  - Remove fluff and professional filler.
  - Ensure the formatting is crisp and professional.

=== OUTPUT FORMAT REQUIREMENTS (CRITICAL) ===
You MUST output the resume in PROPER MARKDOWN format. This is non-negotiable.

```
# [Full Name]
[Contact info on one line: city, email, phone, LinkedIn]

---

## Professional Summary
[2-3 sentence summary paragraph.]

## Technical Skills
- **Languages:** [list]
- **Frameworks:** [list]
- **Tools & Platforms:** [list]

## Professional Experience

### [Job Title] | [Company Name]
*[Start Date] - [End Date]*

- [Achievement bullet with action verb]
- [Achievement bullet with action verb]

### [Previous Job Title] | [Previous Company]
*[Start Date] - [End Date]*

- [Achievement bullet]
- [Achievement bullet]

## Education

### [Degree] | [University/College]
*[Year]*

## Certifications
- [Certification name], [Issuer], [Date]

## Projects
### [Project Name]
- [Brief description with technologies used]
```

FORMAT RULES:
- Use # for name, ## for sections, ### for job titles/degrees
- Use **bold** for skill categories
- Use - (bullet points) for achievements
- Use *italics* for dates
- Add blank lines between sections
- NO code blocks around the output
- Output ONLY the resume, no explanations

ORIGINAL RESUME:
{resume_text}

TARGET KEYWORDS (integrate these strategically, especially in Professional Summary):
{", ".join(state.get("extracted_keywords", []))}

OUTPUT THE OPTIMIZED RESUME IN MARKDOWN FORMAT NOW. 

CRITICAL: 
1. Use DOUBLE NEWLINES between every header and every paragraph. 
2. Use EXACTLY the headers provided (# for name, ## for sections).
3. DO NOT output a single paragraph. 
4. DO NOT use '•' bullets, use '-' instead.
5. NO code blocks, NO intros, NO outros. ONLY THE RESUME.
"""

    if human_feedback and human_feedback.lower() != "proceed":
        messages.append(
            AIMessage(
                content=f"Sub-task: Incorporating human feedback: '{human_feedback}'"
            ).model_dump()
        )
        editing_instructions += f"\n\n=== USER GUIDANCE (PRIORITY) ===\n{human_feedback}\nFollow this instruction while maintaining all constraints above.\n"

    messages.append(
        AIMessage(
            content="Sub-task: Sending enhanced prompt to LLM for professional rewriting."
        ).model_dump()
    )

    logger.info("[LLM] Call started: resume_editing")
    # Smart Stagger for Rate Limits
    time.sleep(1.5)
    try:
        with Timer("llm_invoke for resume_editing"):
            response = _safe_invoke(editor_llm, editing_instructions)
        logger.info("[LLM] Call completed: resume_editing")
        raw_response = response.content.strip()

        # Visual/Structure Repair: If the AI failed to include Markdown symbols (common with 8b models),
        # we perform a "soft-repair" to ensure the frontend renders properly.
        if "#" not in raw_response[:200]:
            print("DEBUG - Detecting missing Markdown headers. Repairing...")
            # Try to identify common resume sections and inject headers using regex
            # Regex aims to match standalone lines or lines starting with these keywords
            sections = {
                "Professional Summary": r"(?i)^(?:##\s*)?Professional Summary",
                "Technical Skills": r"(?i)^(?:##\s*)?Technical Skills?",
                "Professional Experience": r"(?i)^(?:##\s*)?Professional Experience",
                "Experience": r"(?i)^(?:##\s*)?Experience",
                "Education": r"(?i)^(?:##\s*)?Education",
                "Certifications": r"(?i)^(?:##\s*)?Certifications?",
                "Projects": r"(?i)^(?:##\s*)?Projects"
            }
            
            # Inject Name header if it looks like a name start (short first line, no header)
            lines = raw_response.split("\n")
            if lines and not lines[0].strip().startswith("#") and len(lines[0]) < 60:
                 # Heuristic: likely a name
                 lines[0] = f"# {lines[0]}"
                 raw_response = "\n".join(lines)

            # Apply replacements
            for key, pattern in sections.items():
                if re.search(pattern, raw_response, re.MULTILINE):
                    # Replace found section title with strict Markdown header
                    # We use a lambda to ensure we don't double-add ## if it exists (though regex handles optional)
                    # simpler approach: Replace the match with the clean header
                    raw_response = re.sub(pattern, f"## {key}", raw_response, flags=re.MULTILINE | re.IGNORECASE)

        # Debug output
        print(f"\nDEBUG - Raw LLM Response Length: {len(raw_response)}")
        print("DEBUG - First 300 chars of response:")
        print(f"'{raw_response[:300]}...'")

        # Simple cleaning - just remove common intro phrases
        edited_resume = clean_resume_response(raw_response)

        print(f"DEBUG - Cleaned Response Length: {len(edited_resume)}")
        print("DEBUG - First 200 chars of cleaned response:")
        print(f"'{edited_resume[:200]}...'")

        # Safety check
        if len(edited_resume) < 50:
            print("WARNING: Cleaned response is too short, using original response")
            edited_resume = raw_response

        # Final safety check - if still empty, use original resume with basic improvements
        if len(edited_resume) < 50:
            print("ERROR: LLM returned empty response, using original resume")
            edited_resume = resume_text  # Fallback to original

    except Exception as e:
        logger.exception(
            "[LLM] Call failed: resume_editing", extra={"error_type": "system_error"}
        )
        raise SystemFailure(message="Resume editing failed", details={"reason": str(e)})

    # Post-processing to ensure no hallucinations were added
    # Commenting out for now as it might be too aggressive
    # edited_resume = remove_added_content(edited_resume, resume_text)

    messages.append(
        AIMessage(
            content="Sub-task: Professionally enhanced resume content generated."
        ).model_dump()
    )
    messages.append(
        SystemMessage(
            content="Node: `resume_editing_node` - Resume professionally enhanced. Moving to final ATS analysis."
        ).model_dump()
    )
    # Post-check: detect common placeholder / hallucinated contact info added by model
    placeholders = [
        r"John Doe",
        r"johndoe",
        r"example@",
        r"123 Main St",
        r"\(123\)\s*456-7890",
        r"email@example.com",
    ]
    try:
        hallucinated = False
        for p in placeholders:
            if re.search(p, edited_resume, flags=re.IGNORECASE) and not re.search(
                p, resume_text or "", flags=re.IGNORECASE
            ):
                hallucinated = True
                break

        if hallucinated:
            messages.append(
                AIMessage(
                    content="Warning: LLM introduced placeholder personal data — falling back to original resume content."
                ).model_dump()
            )
            edited_resume = resume_text
    except Exception:
        pass

    # Additional safety: ensure edited resume preserves a minimum amount of original content.
    try:
        orig_words = re.findall(r"\w+", (resume_text or "").lower())
        edited_words = re.findall(r"\w+", (edited_resume or "").lower())
        if orig_words:
            common = set(orig_words) & set(edited_words)
            overlap_ratio = len(common) / max(1, len(set(orig_words)))
            if overlap_ratio < 0.25:
                messages.append(
                    AIMessage(
                        content=f"Warning: Edited resume retains only {overlap_ratio:.2f} of original content — reverting to original."
                    ).model_dump()
                )
                edited_resume = resume_text
    except Exception:
        pass

    return {
        **state,
        "edited_resume_content": edited_resume,
        "messages": messages,
        "next_agent": "final_ats_analysis",
        "current_task": "Analyzing new ATS score",
        **emit(
            event="resume_optimized", payload={"resume_preview": edited_resume[:300]}
        ),
    }


# ---------------------------------------------------------------------------
# Council of Agents: 3 strategy-differentiated editors + Arbitrator
# ---------------------------------------------------------------------------

_EDITOR_STRATEGIES = [
    {
        "name": "Keyword Maximizer",
        "preamble": (
            "STRATEGY OVERRIDE — KEYWORD MAXIMIZER:\n"
            "Your PRIMARY goal is to maximize the density and placement of target JD keywords.\n"
            "- Ensure EVERY target keyword appears at least once, ideally in Summary AND first bullet of relevant jobs.\n"
            "- Use exact JD phrasing wherever factually accurate.\n"
            "- Prioritize keyword density over narrative flow.\n\n"
        ),
    },
    {
        "name": "Narrative Polisher",
        "preamble": (
            "STRATEGY OVERRIDE — NARRATIVE POLISHER:\n"
            "Your PRIMARY goal is to maximize the impact of achievement descriptions.\n"
            "- Use the strongest possible action verbs: Engineered, Architected, Spearheaded, Orchestrated.\n"
            "- Frame every bullet as a quantifiable achievement where data exists.\n"
            "- Prioritize persuasive storytelling while maintaining factual accuracy.\n\n"
        ),
    },
    {
        "name": "Structure Optimizer",
        "preamble": (
            "STRATEGY OVERRIDE — STRUCTURE OPTIMIZER:\n"
            "Your PRIMARY goal is to maximize ATS parseability through structure.\n"
            "- Place the most JD-relevant experience items FIRST within each section.\n"
            "- Ensure crystal-clear section headers matching standard ATS parsers.\n"
            "- Reorder bullet points within each job: most relevant to JD first.\n"
            "- Prioritize structural clarity and section ordering over prose polish.\n\n"
        ),
    },
]


def _run_single_editor(state: ResumeOptimizationState, strategy: dict, index: int) -> dict:
    """
    Run a single editor variant with a specific strategy preamble.
    Returns {strategy, content, success} dict.
    """
    resume_text = state["resume_plain_text"]
    job_description = state.get("job_description_text", "")
    keywords = state.get("extracted_keywords", [])
    memory_context = state.get("memory_context", {})
    latest_resume = memory_context.get("latest_resume_content") if memory_context else None
    latest_score = memory_context.get("latest_ats_score") if memory_context else None

    history_context = ""
    if latest_resume and latest_score:
        history_context = f"""
=== HISTORICAL CONTEXT (PREVIOUS BEST VERSION) ===
Previous ATS Score: {latest_score}%
Previous Content:
{latest_resume[:1000]}... [truncated]

GOAL: Your goal is to EXCEED the previous score by refining the content further without over-optimizing. Identify what worked in the last version and build upon it.
"""

    editing_instructions = f"""{strategy['preamble']}TASK: Act as an expert Senior Technical Recruiter. Optimize the provided resume for ATS matching against the specific Job Description below.

{history_context}

=== JOB DESCRIPTION ===
{job_description}

=== ABSOLUTE CONSTRAINTS (NEVER violate) ===
NEVER:
1. Add ANY skills, tools, or technologies to a specific job/project that were NOT originally in that specific section of the resume.
2. Bleed information between sections: Do not move tech stack from "Projects" to "Experience" or vice versa unless it was originally present in both.
3. Invent or modify dates, company names, contact info, or locations.
4. Fabricate metrics, numbers, or specific accomplishments.
5. Create new projects or experience items.

CRITICAL RULE - NO SECTION BLEEDING:
Each job in "Professional Experience" must ONLY describe work done AT THAT COMPANY.
Do NOT add project descriptions or personal project tech to job experience bullets.
Keep "Projects" and "Professional Experience" strictly separate.

ALWAYS:
1. Preserve ALL original facts (dates, titles, companies unchanged).
2. ONLY REPHRASE: You may improve the prose and verb-strength of EXISTING descriptions.
3. Maintain 100% factual fidelity to the source section.
4. REJECT any suggestion to "borrow" skills from one job to benefit another.

=== ALLOWED OPTIMIZATION TECHNIQUES (4 ONLY) ===

[1] PERSUASIVE REPHRASING (SOURCE-ONLY)
  - Transform passive duties into achievement-oriented results using EXISTING facts.
  - Use high-impact verbs: "Engineered", "Optimized", "Spearheaded", "Architected", "Automated".
  - DO NOT add new outcomes. Only rephrase what is already there.

[2] STRATEGIC KEYWORD ALIGNMENT
  - Integrate target keywords from the JD into the Professional Summary ONLY IF you actually possess those skills (as per the whole resume).
  - In experience bullets, only use keywords that were originally present in that specific job.

[3] GENERALIZED SUMMARY
  - Rewrite the Professional Summary to be a bridge between the JD and your actual resume.
  - It should be professional, slightly generalized to cover your career trajectory, but grounded in resume facts.

[4] CONCISION & CLARITY
  - Remove fluff and professional filler.
  - Ensure the formatting is crisp and professional.

=== OUTPUT FORMAT REQUIREMENTS (CRITICAL) ===
You MUST output the resume in PROPER MARKDOWN format. This is non-negotiable.

```
# [Full Name]
[Contact info on one line: city, email, phone, LinkedIn]

---

## Professional Summary
[2-3 sentence summary paragraph.]

## Technical Skills
- **Languages:** [list]
- **Frameworks:** [list]
- **Tools & Platforms:** [list]

## Professional Experience

### [Job Title] | [Company Name]
*[Start Date] - [End Date]*

- [Achievement bullet with action verb]
- [Achievement bullet with action verb]

### [Previous Job Title] | [Previous Company]
*[Start Date] - [End Date]*

- [Achievement bullet]
- [Achievement bullet]

## Education

### [Degree] | [University/College]
*[Year]*

## Certifications
- [Certification name], [Issuer], [Date]

## Projects
### [Project Name]
- [Brief description with technologies used]
```

FORMAT RULES:
- Use # for name, ## for sections, ### for job titles/degrees
- Use **bold** for skill categories
- Use - (bullet points) for achievements
- Use *italics* for dates
- Add blank lines between sections
- NO code blocks around the output
- Output ONLY the resume, no explanations

ORIGINAL RESUME:
{resume_text}

TARGET KEYWORDS (integrate these strategically, especially in Professional Summary):
{", ".join(keywords)}

OUTPUT THE OPTIMIZED RESUME IN MARKDOWN FORMAT NOW.

CRITICAL:
1. Use DOUBLE NEWLINES between every header and every paragraph.
2. Use EXACTLY the headers provided (# for name, ## for sections).
3. DO NOT output a single paragraph.
4. DO NOT use '•' bullets, use '-' instead.
5. NO code blocks, NO intros, NO outros. ONLY THE RESUME.
"""

    # Inject HITL feedback from the human review step
    hitl_feedback = state.get("hitl_feedback")
    if hitl_feedback:
        editing_instructions += f"\n\n=== USER GUIDANCE (PRIORITY) ===\n{hitl_feedback}\nFollow this instruction while maintaining all constraints above.\n"

    logger.info(f"[COUNCIL] Editor #{index} ({strategy['name']}) — invoking LLM")
    time.sleep(2.0)  # Rate-limit stagger

    try:
        with Timer(f"council_editor_{index}"):
            response = _safe_invoke(editor_llm, editing_instructions)
        raw = response.content.strip()

        # Apply the same Markdown repair as the original resume_editing_node
        if "#" not in raw[:200]:
            sections = {
                "Professional Summary": r"(?i)^(?:##\s*)?Professional Summary",
                "Technical Skills": r"(?i)^(?:##\s*)?Technical Skills?",
                "Professional Experience": r"(?i)^(?:##\s*)?Professional Experience",
                "Experience": r"(?i)^(?:##\s*)?Experience",
                "Education": r"(?i)^(?:##\s*)?Education",
                "Certifications": r"(?i)^(?:##\s*)?Certifications?",
                "Projects": r"(?i)^(?:##\s*)?Projects"
            }
            lines = raw.split("\n")
            if lines and not lines[0].strip().startswith("#") and len(lines[0]) < 60:
                lines[0] = f"# {lines[0]}"
                raw = "\n".join(lines)
            for key, pattern in sections.items():
                if re.search(pattern, raw, re.MULTILINE):
                    raw = re.sub(pattern, f"## {key}", raw, flags=re.MULTILINE | re.IGNORECASE)

        cleaned = clean_resume_response(raw)
        if len(cleaned) < 50:
            cleaned = raw

        # Hallucination check
        placeholders = [
            r"John Doe", r"johndoe", r"example@",
            r"123 Main St", r"\(123\)\s*456-7890", r"email@example.com",
        ]
        for p in placeholders:
            if re.search(p, cleaned, flags=re.IGNORECASE) and not re.search(
                p, resume_text or "", flags=re.IGNORECASE
            ):
                logger.warning(f"[COUNCIL] Editor #{index} hallucinated placeholder — skipping")
                return {"strategy": strategy["name"], "content": "", "success": False}

        logger.info(f"[COUNCIL] Editor #{index} ({strategy['name']}) — {len(cleaned)} chars generated")
        return {"strategy": strategy["name"], "content": cleaned, "success": True}

    except Exception as e:
        logger.error(f"[COUNCIL] Editor #{index} ({strategy['name']}) failed: {e}")
        return {"strategy": strategy["name"], "content": "", "success": False}


def council_editing_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    """
    Council of Editors: runs multiple editor instances with different strategies.
    Each produces an independent resume proposal.
    """
    messages = state["messages"]
    count = min(settings.council_editor_count, len(_EDITOR_STRATEGIES))

    messages.append(
        HumanMessage(
            content=f"Node: `council_editing_node` — Spawning {count} editor variants."
        ).model_dump()
    )

    proposals = []
    for i in range(count):
        strategy = _EDITOR_STRATEGIES[i]
        proposal = _run_single_editor(state, strategy, i)
        proposals.append(proposal)
        messages.append(
            AIMessage(
                content=f"Council Editor #{i} ({strategy['name']}): {'✓ Success' if proposal['success'] else '✗ Failed'} — {len(proposal.get('content', ''))} chars"
            ).model_dump()
        )

    # Filter to only successful proposals
    valid = [p for p in proposals if p["success"] and len(p.get("content", "")) > 50]

    if not valid:
        # Fallback: run original single-editor path
        logger.warning("[COUNCIL] All editors failed — falling back to single editor")
        messages.append(
            AIMessage(
                content="Warning: All council editors failed. Falling back to single editor."
            ).model_dump()
        )
        return resume_editing_node(state)

    logger.info(f"[COUNCIL] {len(valid)}/{count} editors succeeded. Passing to arbitrator.")

    return {
        **state,
        "editor_proposals": proposals,
        "messages": messages,
        "next_agent": "arbitrator",
        "current_task": "Arbitrating best proposal",
        **emit(
            event="council_complete",
            payload={"total": count, "successful": len(valid)},
        ),
    }


def arbitrator_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    """
    Arbitrator: Uses the high-reasoning model (Gemini Pro) to score and
    select the best proposal from the Council of Editors.
    """
    messages = state["messages"]
    proposals = state.get("editor_proposals", [])
    job_description = state.get("job_description_text", "")
    keywords = state.get("extracted_keywords", [])

    messages.append(
        HumanMessage(
            content="Node: `arbitrator_node` — Scoring editor proposals."
        ).model_dump()
    )

    valid = [p for p in proposals if p.get("success") and len(p.get("content", "")) > 50]

    if len(valid) == 1:
        # Only one valid proposal — skip arbitration
        logger.info("[ARBITRATOR] Only 1 valid proposal — auto-selecting.")
        winner = valid[0]
        winner_idx = proposals.index(winner)
        messages.append(
            AIMessage(
                content=f"Arbitrator: Only one valid proposal ({winner['strategy']}) — auto-selected."
            ).model_dump()
        )
        return {
            **state,
            "edited_resume_content": winner["content"],
            "winning_proposal_index": winner_idx,
            "messages": messages,
            "next_agent": "final_ats_analysis",
            "current_task": "Analyzing new ATS score",
        }

    # Build comparison prompt for the Arbitrator (Gemini Pro)
    proposals_text = ""
    for i, p in enumerate(valid):
        proposals_text += f"\n--- PROPOSAL {i} ({p['strategy']}) ---\n"
        proposals_text += p["content"][:2500]  # Truncate for prompt size
        proposals_text += "\n"

    prompt = (
        "TASK: You are a Senior ATS Scoring Expert. Compare the following resume proposals "
        "and select the ONE that will score highest on an ATS system for the given job description.\n\n"
        "SCORING CRITERIA (weight in order):\n"
        "1. Keyword Coverage (40%): How many target keywords are present and prominent?\n"
        "2. Keyword Placement (30%): Are keywords in high-value positions (title, summary, first bullets)?\n"
        "3. Factual Integrity (20%): Does the proposal avoid hallucination or fabrication?\n"
        "4. Structure Quality (10%): Clean sections, professional formatting?\n\n"
        f"TARGET KEYWORDS: {', '.join(keywords[:20])}\n\n"
        f"JOB DESCRIPTION (summary):\n{job_description[:1500]}\n\n"
        f"{proposals_text}\n\n"
        "OUTPUT FORMAT (STRICT JSON, no markdown code fences):\n"
        '{\n'
        '  "scores": [<score_0>, <score_1>, ...],\n'
        '  "winner": <index_of_best>,\n'
        '  "justification": "<1-2 sentence reason>"\n'
        '}\n\n'
        "RESPOND WITH JSON ONLY:"
    )

    logger.info("[ARBITRATOR] Invoking Gemini Pro for proposal scoring")
    time.sleep(1.0)

    winner_idx = 0  # Default fallback
    try:
        with Timer("arbitrator_scoring"):
            response = _safe_invoke(arbitrator_llm, prompt)
        res_text = response.content.strip()

        # Parse JSON response
        json_data = None
        try:
            json_data = py_json.loads(res_text)
        except Exception:
            # Try extracting JSON from markdown blocks
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", res_text, re.DOTALL)
            if json_match:
                try:
                    json_data = py_json.loads(json_match.group(0))
                except Exception:
                    pass

        if json_data and "winner" in json_data:
            winner_idx = int(json_data["winner"])
            scores = json_data.get("scores", [])
            justification = json_data.get("justification", "N/A")
            logger.info(
                f"[ARBITRATOR] Winner: Proposal #{winner_idx} | Scores: {scores} | {justification}"
            )
            messages.append(
                AIMessage(
                    content=(
                        f"Arbitrator Decision: Proposal #{winner_idx} ({valid[winner_idx]['strategy']}) wins.\n"
                        f"Scores: {scores}\n"
                        f"Reason: {justification}"
                    )
                ).model_dump()
            )
        else:
            logger.warning("[ARBITRATOR] Could not parse winner — defaulting to Proposal #0")
            messages.append(
                AIMessage(
                    content="Arbitrator: Could not parse scoring response — defaulting to Proposal #0."
                ).model_dump()
            )

    except Exception as e:
        logger.error(f"[ARBITRATOR] Scoring failed: {e} — defaulting to Proposal #0")
        messages.append(
            AIMessage(
                content=f"Arbitrator: Scoring failed ({e}) — defaulting to Proposal #0."
            ).model_dump()
        )

    # Clamp index to valid range
    winner_idx = max(0, min(winner_idx, len(valid) - 1))
    selected = valid[winner_idx]

    return {
        **state,
        "edited_resume_content": selected["content"],
        "winning_proposal_index": winner_idx,
        "messages": messages,
        "next_agent": "final_ats_analysis",
        "current_task": "Analyzing new ATS score",
        **emit(
            event="arbitrator_decided",
            payload={
                "winner_strategy": selected["strategy"],
                "winner_index": winner_idx,
            },
        ),
    }


def final_ats_analysis_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description = state.get("job_description_text", "")
    edited_resume_text = state.get("edited_resume_content", "")
    keywords = state.get("extracted_keywords", [])
    new_ats_score = None

    messages.append(
        HumanMessage(
            content="Node: `final_ats_analysis_node` - Performing final ATS analysis on the optimized resume."
        ).model_dump()
    )

    prompt = (
        "TASK: Score optimized resume's ATS match (0-100%). Show keyword evidence. List improvements made.\n\n"
        "SCORING METHODOLOGY (Transparent Weights):\n"
        "- Keyword Density (40%): Count found keywords, weight by frequency vs. industry baseline\n"
        "- Keyword Placement (30%): Location scoring (title +3, summary +2, first bullets +1, later -0.5)\n"
        "- Context Relevance (20%): Keywords match stated responsibilities/experience (binary per skill)\n"
        "- ATS Format (10%): Clean sections, no graphics, readable structure, standard formatting\n\n"
        "SCORE REFERENCE BANDS:\n"
        "0-25%: Poor - Few keywords found, weak structure\n"
        "26-50%: Below Average - Basic keyword coverage, average structure\n"
        "51-75%: Good - Strong keyword alignment, clean structure\n"
        "76-100%: Excellent - Comprehensive match, optimized format\n\n"
        "OUTPUT FORMAT (EXACT STRUCTURE - NO DEVIATION):\n"
        "**ATS Score: [XX]%**\n"
        "**Keywords Found:** [20-word summary of top matches, with count, e.g., '8 of 15 target keywords found (Python, AWS, Docker, etc.)']\n"
        "**Key Improvements:** [2-3 specific optimizations made, with before/after detail]\n"
        "**Assessment:** [2-sentence summary of match quality and any gaps]\n\n"
        f"Target Keywords (15-25): {', '.join(keywords)}\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Optimized Resume:\n{edited_resume_text}\n\n"
        "START FINAL ASSESSMENT:"
    )
    messages.append(
        AIMessage(
            content=f"Sub-task: Sending prompt to LLM for final ATS score. Prompt snippet: '{prompt[:100]}...'"
        ).model_dump()
    )
    logger.info("[LLM] Call started: final_ats_analysis")
    # Smart Stagger for Rate Limits
    time.sleep(1.0)
    try:
        with Timer("llm_invoke for final_ats_analysis"):
            response = _safe_invoke(analyst_llm, prompt)
    except Exception as e:
        logger.warning(f"Analyst model failed for final_ats_analysis, falling back to Editor: {e}")
        try:
            with Timer("llm_invoke fallback for final_ats_analysis"):
                response = _safe_invoke(editor_llm, prompt)
        except Exception as e2:
            logger.exception("[LLM] Both models failed: final_ats_analysis")
            raise SystemFailure(message="Final ATS analysis failed", details={"reason": str(e2)})
    logger.info("[LLM] Call completed: final_ats_analysis")
    new_analysis_summary = response.content

    # Debug: Log the raw response for score extraction debugging
    logger.info(f"[DEBUG] Final ATS Analysis Response (first 500 chars): {new_analysis_summary[:500]}")

    # More flexible regex to catch various score formats
    score_match = re.search(r"(?:ATS\s*)?Score\D*?(\d+)\s*%?", new_analysis_summary, re.IGNORECASE)

    if score_match:
        new_ats_score = int(score_match.group(1))
        logger.info(f"[DEBUG] Extracted new_ats_score: {new_ats_score}")
        messages.append(
            AIMessage(
                content=f"Sub-task: Estimated Optimized ATS Score: {new_ats_score}%"
            ).model_dump()
        )
    else:
        logger.warning(f"[DEBUG] Failed to extract new ATS score. Response snippet: {new_analysis_summary[:200]}")
        messages.append(
            AIMessage(
                content="Sub-task: Could not parse new ATS Score from LLM response."
            ).model_dump()
        )

    messages.append(
        AIMessage(
            content=f"Sub-task: Final analysis summary: \n{new_analysis_summary[:500]}..."
        ).model_dump()
    )
    messages.append(
        SystemMessage(
            content="Node: `final_ats_analysis_node` - Final ATS analysis completed."
        ).model_dump()
    )
    return {
        **state,
        "new_ats_score": new_ats_score,
        "messages": messages,
        "next_agent": "json_extraction",
        "current_task": "Extracting structured JSON from resume",
    }


def json_extraction_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    """
    Phase 4.1: Converts raw Markdown resume content into structured ResumeJSON.
    This enables headless rendering into PDF, LaTeX, or any format.
    """
    messages = state["messages"]
    edited_resume = state.get("edited_resume_content", "")

    messages.append(
        HumanMessage(
            content="Node: `json_extraction_node` — Parsing resume into structured JSON."
        ).model_dump()
    )

    if not edited_resume or len(edited_resume.strip()) < 50:
        logger.warning("[JSON_EXTRACT] No resume content to parse — skipping.")
        messages.append(
            AIMessage(content="JSON extraction skipped: no resume content.").model_dump()
        )
        return {
            **state,
            "resume_json": None,
            "messages": messages,
            "next_agent": "reflection",
            "current_task": "Self-reflecting on results",
        }

    prompt = (
        "TASK: Parse the following resume text into a structured JSON object.\n\n"
        "OUTPUT FORMAT (STRICT JSON, no markdown fences, no explanation):\n"
        '{\n'
        '  "contact": {\n'
        '    "name": "...",\n'
        '    "email": "...",\n'
        '    "phone": "...",\n'
        '    "location": "...",\n'
        '    "linkedin": "...",\n'
        '    "github": "...",\n'
        '    "portfolio": null\n'
        '  },\n'
        '  "summary": "Professional summary paragraph...",\n'
        '  "skills": [\n'
        '    {"category": "Languages", "skills": ["Python", "JavaScript"]},\n'
        '    {"category": "Frameworks", "skills": ["React", "FastAPI"]}\n'
        '  ],\n'
        '  "experience": [\n'
        '    {\n'
        '      "title": "Software Engineer",\n'
        '      "company": "Acme Corp",\n'
        '      "location": "San Francisco, CA",\n'
        '      "start_date": "Jan 2023",\n'
        '      "end_date": "Present",\n'
        '      "bullets": ["Engineered X achieving Y..."]\n'
        '    }\n'
        '  ],\n'
        '  "education": [\n'
        '    {\n'
        '      "degree": "B.S. Computer Science",\n'
        '      "institution": "MIT",\n'
        '      "location": null,\n'
        '      "year": "2022",\n'
        '      "gpa": null,\n'
        '      "details": []\n'
        '    }\n'
        '  ],\n'
        '  "projects": [\n'
        '    {\n'
        '      "name": "Project Name",\n'
        '      "description": "Brief description",\n'
        '      "technologies": ["React", "Node.js"],\n'
        '      "bullets": ["Built X..."],\n'
        '      "url": null\n'
        '    }\n'
        '  ],\n'
        '  "certifications": [\n'
        '    {"name": "AWS Solutions Architect", "issuer": "Amazon", "date": "2024"}\n'
        '  ]\n'
        '}\n\n'
        "RULES:\n"
        "1. Extract EVERY section from the resume — do not skip any.\n"
        "2. If a field is missing from the source, use null (not empty string).\n"
        "3. Preserve exact dates, company names, and facts.\n"
        "4. Do NOT invent or add any information.\n"
        "5. Return ONLY the JSON object, no explanation or wrapping.\n\n"
        f"RESUME TEXT:\n{edited_resume}\n\n"
        "JSON:"
    )

    logger.info("[JSON_EXTRACT] Invoking LLM for structured extraction")
    time.sleep(1.0)

    resume_json = None
    try:
        with Timer("json_extraction"):
            response = _safe_invoke(analyst_llm, prompt)
        res_text = response.content.strip()

        # Parse JSON — try multiple strategies
        try:
            resume_json = py_json.loads(res_text)
        except Exception:
            # Strip markdown code fences if present
            cleaned = res_text
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            try:
                resume_json = py_json.loads(cleaned)
            except Exception:
                # Last resort: regex extract
                json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if json_match:
                    try:
                        resume_json = py_json.loads(json_match.group(0))
                    except Exception:
                        pass

        if resume_json and "contact" in resume_json:
            # Validate with Pydantic schema
            from ..schemas.resume_schema import ResumeJSON as ResumeJSONSchema
            validated = ResumeJSONSchema(**resume_json)
            resume_json = validated.model_dump()
            logger.info(
                f"[JSON_EXTRACT] Success — {len(resume_json.get('experience', []))} experiences, "
                f"{len(resume_json.get('skills', []))} skill categories extracted."
            )
            messages.append(
                AIMessage(
                    content=f"JSON extraction complete: {validated.contact.name} — "
                    f"{len(validated.experience)} jobs, "
                    f"{len(validated.skills)} skill categories."
                ).model_dump()
            )
        else:
            logger.warning("[JSON_EXTRACT] Parsed JSON is missing 'contact' field — marking as failed.")
            resume_json = None
            messages.append(
                AIMessage(
                    content="JSON extraction produced incomplete data — falling back to Markdown only."
                ).model_dump()
            )

    except Exception as e:
        logger.error(f"[JSON_EXTRACT] Failed: {e}")
        messages.append(
            AIMessage(
                content=f"JSON extraction failed ({e}) — falling back to Markdown only."
            ).model_dump()
        )

    return {
        **state,
        "resume_json": resume_json,
        "messages": messages,
        "next_agent": "reflection",
        "current_task": "Self-reflecting on results",
    }


def reflection_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    edited_resume = state.get("edited_resume_content", "")

    messages.append(
        HumanMessage(
            content="Node: `reflection_node` - Critiquing the optimized resume against Job Description."
        ).model_dump()
    )

    prompt = (
        "TASK: Act as a strict Factual Auditor. Compare the ORIGINAL resume against the OPTIMIZED version.\n\n"
        "GOAL:\n"
        "Identify if the AI Editor 'hallucinated' or added any skills, tools, or experiences that did not exist in the source material.\n\n"
        "CRITIQUE CRITERIA:\n"
        "1. Hallucination Check: Did the editor add any specific tools (e.g., 'Snowflake', 'AWS') that were NOT in the original? (INSTANT FAIL if so)\n"
        "2. Section Bleeding: Did the editor move skills/tech from projects into job experience where they don't belong?\n"
        "3. Factual Drift: Did bullet points change their meaning to sound more impressive than the source facts? \n"
        "OUTPUT FORMAT (Strictly JSON):\n"
        "{\n"
        "  \"good\": \"Points about factual rephrasing\",\n"
        "  \"gap\": \"Hallucinated skills or metrics found (be specific)\",\n"
        "  \"integrity_status\": \"PASS/FAIL\",\n"
        "  \"should_retry\": true/false,\n"
        "  \"retry_reason\": \"Specific instruction to REMOVE hallucinated content\"\n"
        "}\n\n"
        f"ORIGINAL RESUME:\n{state.get('resume_plain_text', '')[:2000]}\n\n"
        f"OPTIMIZED RESUME:\n{edited_resume[:2000]}\n\n"
        "AUDIT JSON:"
    )

    logger.info("[LLM] Call started: reflection")
    # Smart Stagger for Rate Limits
    time.sleep(1.5)
    try:
        try:
            response = _safe_invoke(analyst_llm, prompt)
        except Exception as e:
            logger.warning(f"Analyst model failed for reflection, falling back to Editor: {e}")
            response = _safe_invoke(editor_llm, prompt)
        
        import json as py_json
        res_text = response.content.strip()
        
        # Debug: Log raw reflection response
        logger.info(f"[DEBUG] Reflection raw response (first 300 chars): {res_text[:300]}")
        
        # More resilient JSON extraction - try multiple strategies
        json_data = None
        
        # Strategy 1: Direct JSON parse
        try:
            json_data = py_json.loads(res_text)
        except Exception:
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        if not json_data:
            if res_text.startswith("```"):
                if "json" in res_text[:20]:
                    res_text = res_text.split("```json", 1)[1].split("```", 1)[0].strip()
                else:
                    res_text = res_text.split("```", 2)[1].strip()
                try:
                    json_data = py_json.loads(res_text)
                except Exception:
                    pass
        
        # Strategy 3: Regex extraction of JSON object
        if not json_data:
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", res_text, re.DOTALL)
            if json_match:
                try:
                    json_data = py_json.loads(json_match.group(0))
                except Exception:
                    pass
        
        if json_data:
            reflection_data = json_data
            reflection_report = f"- **The Good**: {reflection_data.get('good', 'N/A')}\n- **The Gap**: {reflection_data.get('gap', 'N/A')}\n- **Integrity**: {reflection_data.get('integrity_status', 'N/A')}"
            should_retry = reflection_data.get("should_retry", False)
            retry_instruction = reflection_data.get("retry_reason", "")
            logger.info("[DEBUG] Reflection JSON parsed successfully")
        else:
            raise ValueError("Could not extract valid JSON from reflection response")
            
    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        logger.error(f"[DEBUG] Reflection response that failed: {response.content[:500] if hasattr(response, 'content') else 'N/A'}")
        reflection_report = "Reflection node failed to generate report."
        should_retry = False
        retry_instruction = ""

    # Prevent infinite loops (max 1 retry)
    current_retries = state.get("self_correction_count", 0)
    if current_retries >= 1:
        should_retry = False

    messages.append(AIMessage(content=f"Reflection Insight:\n{reflection_report}").model_dump())

    new_state = {
        **state,
        "reflection_report": reflection_report,
        "messages": messages,
        "current_task": "Finalizing output",
    }
    
    # DISABLED: Retry logic causes backend crashes - always proceed forward
    should_retry = False
    
    if should_retry:
        new_state["human_feedback"] = f"AUTO-RETRY INSTRUCTION: {retry_instruction}"
        new_state["self_correction_count"] = current_retries + 1
        new_state["next_agent"] = "resume_editing"
        logger.info(f"[REFLECTION] Triggering self-correction retry #{new_state['self_correction_count']}")
    else:
        new_state["next_agent"] = "final_response"
        
    return new_state


def route_after_reflection(
    state: ResumeOptimizationState,
) -> Literal["resume_editing", "cover_letter_analysis", "final_response"]:
    """
    Route based on reflection outcome and service requests.
    """
    next_agent = state.get("next_agent")
    if next_agent == "resume_editing":
        return "resume_editing"
    
    services = state.get("services_requested", [])
    if "cover" in services:
        return "cover_letter_analysis"
    else:
        return "final_response"


def final_response_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    old_ats_score = state.get("old_ats_score")
    new_ats_score = state.get("new_ats_score")
    analysis_report = state.get("analysis_report", "N/A")
    edited_resume = state.get("edited_resume_content", "")

    cover_letter = state.get("cover_letter_text", "")
    cover_letter_markdown = state.get("cover_letter_markdown", "")
    cover_letter_path = (
        cover_letter_markdown.split("Saved to: ")[-1]
        if cover_letter_markdown
        else "N/A"
    )

    messages.append(
        HumanMessage(
            content="Node: `final_response_node` - Generating final report."
        ).model_dump()
    )

    # Save the optimized resume to a file if it exists
    saved_filepath = "N/A"
    if edited_resume:
        saved_filepath = save_resume_to_markdown(edited_resume)
        messages.append(
            AIMessage(
                content=f"Optimized resume saved to: {saved_filepath}"
            ).model_dump()
        )

    # Construct report conditionally
    report_lines = [
        "--- Resume Optimization Report ---",
        f"**Original ATS Score:** {old_ats_score if old_ats_score is not None else 'N/A'}%",
        f"**Optimized ATS Score:** {new_ats_score if new_ats_score is not None else 'N/A'}%",
        "",
        "--- Detailed Analysis of Original Resume ---",
        analysis_report,
        "",
    ]

    if edited_resume:
        report_lines.extend(
            [
                "--- Optimized Resume Content ---",
                f"Saved to file: {saved_filepath}",
                "",
                f"```markdown\n{edited_resume}\n```",
                "",
            ]
        )

    if cover_letter:
        report_lines.extend(
            [
                "--- Professional Cover Letter ---",
                f"Saved to: {cover_letter_path}",
                f"{cover_letter}",
                "",
            ]
        )

    report_lines.extend(
        [
            "--- Next Steps ---",
            "1. Review both documents" if cover_letter else "1. Review the document",
            "2. Customize further if needed",
            "3. Submit with your application!",
        ]
    )

    final_report_content = "\n".join(report_lines)

    messages.append(AIMessage(content=final_report_content).model_dump())
    messages.append(
        SystemMessage(
            content="Node: `final_response_node` - Final report generated. Workflow complete."
        ).model_dump()
    )
    return {
        **state,
        "messages": messages,
        "task_complete": True,
        "next_agent": "end",
        "current_task": "Completed",
        "saved_resume_path": saved_filepath,  # Add the filepath to the state
        **emit(
            event="workflow_completed",
            payload={
                "old_ats_score": old_ats_score,
                "new_ats_score": new_ats_score,
                "saved_resume_path": saved_filepath,
            },
        ),
    }


### Cover letter workflow ###


def cover_letter_analysis_node(
    state: ResumeOptimizationState,
) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description = state["job_description_text"]
    resume_text = state["resume_plain_text"]

    messages.append(
        HumanMessage(
            content="Node: `cover_letter_analysis_node` - Analyzing resume for letter content extraction."
        ).model_dump()
    )

    prompt = (
        "TASK: Extract key elements from resume+JD to inform cover letter. NO preamble.\n\n"
        "OUTPUT FORMAT:\n"
        "Bullet-point list ONLY. 6-10 bullets total. Categories:\n"
        "• Relevant Experience: [2-3 bullets, list job titles/companies/key projects]\n"
        "• Matching Skills: [2-3 bullets, only those in JD requirements]\n"
        "• Key Achievements: [1-2 bullets, quantifiable results if available]\n"
        "• Tone/Style: [1 bullet, describe communication style from resume]\n\n"
        "SELECTION RULES:\n"
        "1. ONLY include info explicitly in resume (no inferences)\n"
        "2. ONLY skills that match job description keywords\n"
        "3. Most recent/relevant experience first\n"
        "4. Verify each bullet against resume source\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Resume Content:\n{resume_text}\n\n"
        "KEY ELEMENTS FOR COVER LETTER (bullets only):"
    )

    logger.info("[LLM] Call started: cover_letter_analysis")
    try:
        with Timer("llm_invoke for cover_letter_analysis"):
            response = _safe_invoke(analyst_llm, prompt)
    except Exception as e:
        logger.exception(
            "[LLM] Call failed: cover_letter_analysis",
            extra={"error_type": "system_error"},
        )
        raise SystemFailure(
            message="Cover letter analysis failed", details={"reason": str(e)}
        )
    logger.info("[LLM] Call completed: cover_letter_analysis")
    analysis = response.content
    return {
        **state,
        "cover_letter_analysis": analysis,
        "messages": messages,
        "next_agent": "cover_letter_generation",
        "current_task": "Generating cover letter",
    }


def cover_letter_generation_node(
    state: ResumeOptimizationState,
) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description = state["job_description_text"]
    resume_analysis = state["cover_letter_analysis"]

    messages.append(
        HumanMessage(
            content="Node: `cover_letter_generation_node` - Generating professional cover letter."
        ).model_dump()
    )

    prompt = (
        "TASK: Write professional cover letter. Single paragraph. Use ONLY resume facts.\n\n"
        "LETTER STRUCTURE (REQUIRED):\n"
        "1. Opening (1 sentence): [Your name, position applied for, brief intro]\n"
        "2. Value Proposition (2-3 sentences): [2-3 specific reasons you're a strong fit from resume]\n"
        "3. Relevant Example (1 sentence): [One specific achievement that matches JD]\n"
        "4. Closing (1 sentence): [Interest in discussing, gratitude]\n\n"
        "CONTENT RULES - NEVER:\n"
        "- Use information NOT in provided resume analysis\n"
        "- Fabricate achievements, skills, or experience\n"
        "- Make generic statements (no corporate jargon)\n"
        "- Repeat resume verbatim (paraphrase, be conversational)\n"
        "- Include phone/email (these are implied)\n\n"
        "FORMAT REQUIREMENTS:\n"
        "- Length: 4-6 sentences, ~200 words max\n"
        "- Output: Plain markdown paragraph (no section headers)\n"
        "- Tone: Professional, warm, authentic\n"
        "- NO preamble, NO closing signature block\n\n"
        f"Resume Key Points (use these only):\n{resume_analysis}\n\n"
        f"Target Position Details:\n{job_description}\n\n"
        "COVER LETTER (4-6 sentences, plain markdown):"
    )

    logger.info("[LLM] Call started: cover_letter_generation")
    try:
        with Timer("llm_invoke for cover_letter_generation"):
            response = _safe_invoke(editor_llm, prompt)
    except Exception as e:
        logger.exception(
            "[LLM] Call failed: cover_letter_generation",
            extra={"error_type": "system_error"},
        )
        raise SystemFailure(
            message="Cover letter generation failed", details={"reason": str(e)}
        )
    logger.info("[LLM] Call completed: cover_letter_generation")
    cover_letter_md = response.content

    # Save to file
    save_cover_letter_to_markdown(cover_letter_md)

    return {
        **state,
        "cover_letter_text": cover_letter_md.replace("```markdown", "")
        .replace("```", "")
        .strip(),
        "cover_letter_markdown": cover_letter_md,
        "messages": messages,
        "next_agent": "final_response",
        "current_task": "Finalizing documents",
    }


### Router ###


def determine_next_step(
    state: ResumeOptimizationState,
) -> Literal["resume_editing", "cover_letter_analysis", "final_response", END]:
    feedback = state.get("human_feedback", "").lower().strip()
    if feedback == "exit" or feedback == "done":
        return END

    services = state.get("services_requested", [])

    # Prioritize resume optimization if requested
    if "resume" in services:
        return "resume_editing"

    # If no resume requested, but cover letter is
    if "cover" in services:
        return "cover_letter_analysis"

    # Fallback/Default
    return "final_response"


def route_after_ats_analysis(
    state: ResumeOptimizationState,
) -> Literal["cover_letter_analysis", "final_response"]:
    """
    Route to cover letter generation only if cover letter was requested.
    Otherwise skip to final response.
    """
    services = state.get("services_requested", [])
    if "cover" in services:
        return "cover_letter_analysis"
    else:
        return "final_response"


### File persistence helpers (used by nodes -> move) ###


def save_resume_to_markdown(
    resume_content: str, filename_prefix: str = "optimized_resume"
) -> str:
    """
    Saves the optimized resume content to a markdown file with a timestamp.
    Returns the path to the saved file.
    """
    import datetime
    import os

    # Create an 'outputs' directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.md"
    filepath = os.path.join("outputs", filename)

    # Write the content to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(resume_content)

    return filepath


def save_cover_letter_to_markdown(content: str) -> str:
    """Saves cover letter to a markdown file with timestamp"""
    import datetime
    import os

    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cover_letter_{timestamp}.md"
    filepath = os.path.join("outputs", filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


### Graph Construction ###

COUNCIL_MODE = settings.enable_council_mode
logger.info(f"[GRAPH] Council mode: {'ENABLED' if COUNCIL_MODE else 'DISABLED'}")

workflow = StateGraph(ResumeOptimizationState)

workflow.add_node("ingestion", ingestion_node)
workflow.add_node("keyword_extraction", keyword_extraction_node)
workflow.add_node("resume_analysis", resume_analysis_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node(
    "determine_next_step", determine_next_step
)  # This is now a router node
workflow.add_node("final_ats_analysis", final_ats_analysis_node)
workflow.add_node("json_extraction", json_extraction_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("cover_letter_analysis", cover_letter_analysis_node)
workflow.add_node("cover_letter_generation", cover_letter_generation_node)
workflow.add_node("final_response", final_response_node)

if COUNCIL_MODE:
    # Council of Agents path
    workflow.add_node("council_editing", council_editing_node)
    workflow.add_node("arbitrator", arbitrator_node)
    # Keep single editor as fallback node
    workflow.add_node("resume_editing", resume_editing_node)
else:
    workflow.add_node("resume_editing", resume_editing_node)

workflow.set_entry_point("ingestion")

workflow.add_edge("ingestion", "keyword_extraction")
workflow.add_edge("keyword_extraction", "resume_analysis")
workflow.add_edge("resume_analysis", "human_review")


if COUNCIL_MODE:
    # Route: human_review -> council_editing (for resume) or cover_letter/final
    def determine_next_step_council(
        state: ResumeOptimizationState,
    ) -> Literal["council_editing", "cover_letter_analysis", "final_response", END]:
        feedback = state.get("human_feedback", "").lower().strip()
        if feedback == "exit" or feedback == "done":
            return END
        services = state.get("services_requested", [])
        if "resume" in services:
            return "council_editing"
        if "cover" in services:
            return "cover_letter_analysis"
        return "final_response"

    workflow.add_conditional_edges(
        "human_review",
        determine_next_step_council,
        {
            "council_editing": "council_editing",
            "cover_letter_analysis": "cover_letter_analysis",
            "final_response": "final_response",
            END: END,
        },
    )
    workflow.add_edge("council_editing", "arbitrator")
    workflow.add_edge("arbitrator", "final_ats_analysis")
    # Fallback edge: if council falls back to single editor
    workflow.add_edge("resume_editing", "final_ats_analysis")
else:
    # Original single-editor path
    workflow.add_conditional_edges(
        "human_review",
        determine_next_step,
        {
            "resume_editing": "resume_editing",
            "cover_letter_analysis": "cover_letter_analysis",
            "final_response": "final_response",
            END: END,
        },
    )
    workflow.add_edge("resume_editing", "final_ats_analysis")

workflow.add_edge("final_ats_analysis", "json_extraction")
workflow.add_edge("json_extraction", "reflection")


# Reflection routing — same for both modes
def _route_after_reflection_for_mode(
    state: ResumeOptimizationState,
) -> Literal["resume_editing", "cover_letter_analysis", "final_response"]:
    next_agent = state.get("next_agent")
    if next_agent == "resume_editing":
        return "resume_editing"
    services = state.get("services_requested", [])
    if "cover" in services:
        return "cover_letter_analysis"
    return "final_response"

workflow.add_conditional_edges(
    "reflection",
    _route_after_reflection_for_mode,
    {
        "resume_editing": "resume_editing",
        "cover_letter_analysis": "cover_letter_analysis",
        "final_response": "final_response",
    },
)
workflow.add_edge("cover_letter_analysis", "cover_letter_generation")
workflow.add_edge("cover_letter_generation", "final_response")
workflow.add_edge("final_response", END)

final_workflow = workflow.compile(checkpointer=InMemorySaver())


def build_resume_graph():
    """
    Factory function to build and return the compiled resume optimization graph.
    UI and APIs must never construct graphs directly.
    """
    return final_workflow
