import re
import logging
import time
from typing import TypedDict, Annotated, List, Literal, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from ..core.llm import get_llm
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

llm = get_llm()


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


class _SimpleResp:
    def __init__(self, content):
        self.content = content


def _safe_invoke(target, *args, **kwargs):
    """Call `target.invoke(...)` if available, otherwise call `target(...)` if callable.
    Always returns an object with a `.content` attribute.
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
        return _SimpleResp(f"Error invoking target: {e}")


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
            response = _safe_invoke(llm, prompt)
    except Exception as e:
        logger.exception(
            "[LLM] Call failed: keyword_extraction",
            extra={"error_type": "system_error"},
        )
        raise SystemFailure(
            message="Keyword extraction failed", details={"reason": str(e)}
        )
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
            response = _safe_invoke(llm, prompt)
    except Exception as e:
        logger.exception(
            "[LLM] Call failed: resume_analysis", extra={"error_type": "system_error"}
        )
        raise SystemFailure(
            message="Resume analysis failed", details={"reason": str(e)}
        )
    logger.info("[LLM] Call completed: resume_analysis")
    analysis_report = response.content

    # Robust regex to catch "ATS Score: **85%**", "ATS Score: 85", etc.
    score_match = re.search(r"ATS\s*Score\D+(\d+)", analysis_report, re.IGNORECASE)

    if score_match:
        old_ats_score = int(score_match.group(1))
        messages.append(
            AIMessage(
                content=f"Sub-task: Estimated Original ATS Score: {old_ats_score}%"
            ).model_dump()
        )
    else:
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
    Simulates human review and automatically 'approves' to proceed.
    The actual interrupt() is commented out for automated execution.
    """
    messages = state["messages"]
    analysis_report = state["analysis_report"]

    # Simulating the human review by printing the report and auto-setting feedback
    messages.append(
        AIMessage(
            content=f"Node: `human_review_node` - Analysis report for human review:\n{analysis_report}\n\n"
        ).model_dump()
    )
    messages.append(
        AIMessage(
            content="Simulating human review: Automatically setting feedback to 'proceed'."
        ).model_dump()
    )

    # The actual interrupt() for human interaction would go here if not automating:
    # human_prompt_data = {"analysis_report": analysis_report, "message": "Analysis is complete. Please review and provide feedback, or type 'proceed' to continue."}
    # human_response_from_ui = interrupt(human_prompt_data)
    # feedback_text = human_response_from_ui if isinstance(human_response_from_ui, str) else ""

    # For this automated version, we simply set feedback to "proceed"
    feedback_text = "proceed"

    messages.append(
        SystemMessage(
            content="Node: `human_review_node` - Human review (automated) completed. Proceeding."
        ).model_dump()
    )

    return {
        **state,
        "human_feedback": feedback_text,
        "messages": messages,
        # IMPORTANT: Remove "next_agent" from here. This node just updates state.
        # The routing decision is made by the `determine_next_step` router function.
        "current_task": "Processing human decision (automated)",
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

VALIDATION RULE: If a keyword is in the Job Description but NOT in the original resume section, DO NOT add it to that section. 

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

TARGET KEYWORDS (use for alignment reference only):
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
        editing_instructions += f"\n\nAdditional Instructions: {human_feedback}"

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
            response = _safe_invoke(llm, editing_instructions)
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
            response = _safe_invoke(llm, prompt)
    except Exception as e:
        logger.exception(
            "[LLM] Call failed: final_ats_analysis",
            extra={"error_type": "system_error"},
        )
        raise SystemFailure(
            message="Final ATS analysis failed", details={"reason": str(e)}
        )
    logger.info("[LLM] Call completed: final_ats_analysis")
    new_analysis_summary = response.content

    # Robust regex for ATS score
    score_match = re.search(r"ATS\s*Score\D+(\d+)", new_analysis_summary, re.IGNORECASE)

    if score_match:
        new_ats_score = int(score_match.group(1))
        messages.append(
            AIMessage(
                content=f"Sub-task: Estimated Optimized ATS Score: {new_ats_score}%"
            ).model_dump()
        )
    else:
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
        response = _safe_invoke(llm, prompt)
        import json as py_json
        res_text = response.content.strip()
        # Clean potential markdown blocks
        if res_text.startswith("```"):
            if "json" in res_text[:10]:
                res_text = res_text.split("```json", 1)[1].split("```", 1)[0].strip()
            else:
                res_text = res_text.split("```", 2)[1].strip()
        
        reflection_data = py_json.loads(res_text)
        reflection_report = f"- **The Good**: {reflection_data.get('good')}\n- **The Gap**: {reflection_data.get('gap')}\n- **Integrity**: {reflection_data.get('integrity_status')}"
        should_retry = reflection_data.get("should_retry", False)
        retry_instruction = reflection_data.get("retry_reason", "")
    except Exception as e:
        logger.error(f"Reflection failed: {e}")
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
            response = _safe_invoke(llm, prompt)
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
            response = _safe_invoke(llm, prompt)
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

workflow = StateGraph(ResumeOptimizationState)

workflow.add_node("ingestion", ingestion_node)
workflow.add_node("keyword_extraction", keyword_extraction_node)
workflow.add_node("resume_analysis", resume_analysis_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node(
    "determine_next_step", determine_next_step
)  # This is now a router node
workflow.add_node("resume_editing", resume_editing_node)
workflow.add_node("final_ats_analysis", final_ats_analysis_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("cover_letter_analysis", cover_letter_analysis_node)
workflow.add_node("cover_letter_generation", cover_letter_generation_node)
workflow.add_node("final_response", final_response_node)

workflow.set_entry_point("ingestion")

workflow.add_edge("ingestion", "keyword_extraction")
workflow.add_edge("keyword_extraction", "resume_analysis")
workflow.add_edge("resume_analysis", "human_review")


# KEY CHANGE: Conditional edges from the router node itself
workflow.add_conditional_edges(
    "human_review",  # The node that returns the routing decision
    determine_next_step,  # The function that makes the routing decision
    {
        "resume_editing": "resume_editing",
        "cover_letter_analysis": "cover_letter_analysis",
        "final_response": "final_response",
        END: END,
    },
)

workflow.add_edge("resume_editing", "final_ats_analysis")
workflow.add_edge("final_ats_analysis", "reflection")

workflow.add_conditional_edges(
    "reflection",
    route_after_reflection,
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
