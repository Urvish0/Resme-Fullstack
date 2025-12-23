import os
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from ..utils.file_parsers import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_doc
)
 
def prepare_resume_state(
    job_description_raw: str,
    resume_file,
    resume_raw_content: str,
    resume_format: str
) -> Dict[str, Any]:
    """
    Prepare and validate the initial LangGraph state.
    UI must never build workflow state directly.
    """

    if not job_description_raw or not job_description_raw.strip():
        raise ValueError("Job description is required")

    final_resume_content = ""

    # Handle uploaded file
    if resume_file is not None:
        path = getattr(resume_file, "name", None)
        ext = os.path.splitext(path)[-1].lower()

        if ext in [".md", ".txt"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                final_resume_content = f.read()

        elif ext == ".pdf":
            final_resume_content = extract_text_from_pdf(path)

        elif ext == ".docx":
            final_resume_content = extract_text_from_docx(path)

        elif ext == ".doc":
            final_resume_content = extract_text_from_doc(path)

    # Append pasted resume content
    if resume_raw_content and resume_raw_content.strip():
        final_resume_content = (
            final_resume_content + "\n" + resume_raw_content
            if final_resume_content
            else resume_raw_content
        )

    if not final_resume_content.strip():
        raise ValueError("Resume content is required")

    return {
        "messages": [HumanMessage(content="Optimize my resume!")],
        "job_description_raw": job_description_raw,
        "resume_raw_content": final_resume_content,
        "resume_format": resume_format or "markdown",
        "job_description_text": "",
        "resume_plain_text": "",
        "extracted_keywords": [],
        "analysis_report": "",
        "edited_resume_content": "",
        "human_feedback": "proceed",
        "next_agent": "",
        "task_complete": False,
        "current_task": "",
        "old_ats_score": None,
        "new_ats_score": None,
        "cover_letter_text": "",
        "cover_letter_markdown": "",
    }
