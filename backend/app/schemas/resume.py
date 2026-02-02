from pydantic import BaseModel, Field
from typing import Optional, List


class ResumeOptimizeRequest(BaseModel):
    job_description: Optional[str] = Field(None)
    resume_text: Optional[str] = None
    resume_format: Optional[str] = "markdown"
    # Which outputs the user wants: "resume", "cover", "coldEmail"
    services: List[str] = Field(default_factory=list)
    # Cold email related fields (optional)
    cold_email_sender_name: Optional[str] = None
    cold_email_sender_email: Optional[str] = None
    cold_email_recipient_name: Optional[str] = None
    cold_email_recipient_email: Optional[str] = None


class ResumeOptimizeResponse(BaseModel):
    optimized_resume: Optional[str] = None
    cover_letter: Optional[str] = None
    cold_email: Optional[str] = None
    old_ats_score: Optional[int] = None
    new_ats_score: Optional[int] = None
    extracted_keywords: List[str] = Field(default_factory=list)
