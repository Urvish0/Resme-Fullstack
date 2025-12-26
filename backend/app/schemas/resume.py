from pydantic import BaseModel, Field
from typing import Optional, List


class ResumeOptimizeRequest(BaseModel):
    job_description: str = Field(..., min_length=50)
    resume_text: Optional[str] = None
    resume_format: Optional[str] = "markdown"


class ResumeOptimizeResponse(BaseModel):
    optimized_resume: str
    cover_letter: Optional[str]
    old_ats_score: Optional[int]
    new_ats_score: Optional[int]
    extracted_keywords: List[str]
