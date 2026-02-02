import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from ..core.llm import get_llm
from ..core.exceptions import SystemFailure
from ..utils.timing import Timer

logger = logging.getLogger(__name__)


def _extract_name_from_resume(resume_text: str) -> Optional[str]:
    """Very small heuristic to pick a sender name from resume text.
    Looks at the first non-empty line and returns the first 2-3 capitalized words.
    """
    if not resume_text:
        return None

    for line in resume_text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        # simple heuristic: words with initial caps and length>1
        parts = clean_line.split()
        candidate = []
        for w in parts[:4]:
            if w[0].isupper() and w.isalpha():
                candidate.append(w)
            else:
                break

        if len(candidate) >= 2:
            return " ".join(candidate[:3])
        # if single capitalized word and next word is all caps (like initials), accept two
        if len(candidate) == 1 and len(parts) >= 2 and parts[1].isalpha():
            return f"{candidate[0]} {parts[1]}"

    return None


def generate_cold_email(
    resume_text: str,
    job_description: Optional[str],
    sender_name: Optional[str],
    sender_email: Optional[str],
    recipient_name: Optional[str] = None,
    recipient_email: Optional[str] = None,
    company_name: Optional[str] = None,
    target_role: Optional[str] = None,
) -> str:
    """
    Generate a concise cold email based on the resume and optional job description.
    Returns plain text email body.
    """
    llm = get_llm()

    # Try to infer sender name from resume if not provided
    inferred_name = _extract_name_from_resume(resume_text)
    sender_display = sender_name or inferred_name or ""

    # Build the prompt using structured format matching other agents
    prompt_parts = [
        "TASK: Write a professional cold outreach email. Use ONLY facts from resume.\n",
        "=== ABSOLUTE CONSTRAINTS (NEVER violate) ===",
        "NEVER:",
        "1. Fabricate skills, projects, or experience not in resume",
        "2. Invent company names, job titles, or achievements",
        "3. Add metrics or numbers not explicitly stated",
        "4. Include phone numbers or email addresses in body",
        "5. Use generic corporate jargon or buzzwords\n",
        "ALWAYS:",
        "1. Reference specific skills/experience from resume only",
        "2. Keep tone professional but warm and conversational",
        "3. Be concise (3-5 sentences maximum)",
        "4. Make the value proposition clear in first 2 sentences\n",
        "=== EMAIL STRUCTURE (REQUIRED) ===",
        "1. Opening (1 sentence): Introduce yourself, mention target role/company if provided",
        "2. Value Proposition (1-2 sentences): 2-3 specific skills from resume that match the opportunity",
        "3. Call to Action (1 sentence): Request for brief conversation or meeting",
        "4. Sign-off: Professional closing with sender name only (no contact details)\n",
        "=== FORMAT REQUIREMENTS ===",
        "- Length: 3-5 sentences, ~100 words max",
        "- Output: Plain text, ready to send",
        "- NO subject line (recipient will add)",
        "- NO preamble or explanatory text",
        "- NO signature block with contact info\n",
    ]

    # Add context
    prompt_parts.append("=== CONTEXT ===")
    prompt_parts.append(f"Sender Name: {sender_display}")

    if recipient_name:
        prompt_parts.append(f"Recipient Name: {recipient_name}")

    if company_name:
        prompt_parts.append(f"Target Company: {company_name}")

    if target_role:
        prompt_parts.append(f"Target Role: {target_role}")

    if job_description:
        prompt_parts.append(f"\nJob Description:\n{job_description[:8000]}")

    prompt_parts.append(f"\nResume (use for facts only):\n{resume_text[:12000]}")
    prompt_parts.append("\n=== OUTPUT ===")
    prompt_parts.append("COLD EMAIL (plain text, 3-5 sentences):")

    prompt = "\n".join(prompt_parts)

    try:
        logger.info("[COLD_EMAIL] Sending prompt to LLM")
        with Timer("llm_invoke_cold_email"):
            response = llm.invoke([HumanMessage(content=prompt)])

        if hasattr(response, "content"):
            body = response.content
        elif isinstance(response, str):
            body = response
        else:
            body = str(response)

        return body.strip()

    except Exception as e:
        logger.exception("[COLD_EMAIL] LLM generation failed")
        raise SystemFailure(
            message="Cold email generation failed", details={"reason": str(e)}
        )
