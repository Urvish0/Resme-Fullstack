from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Header,
    Query,
    UploadFile,
    File,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import logging
import uuid
import threading
import tempfile
import os
import requests
from bs4 import BeautifulSoup

from ...schemas.resume import ResumeOptimizeRequest, ResumeOptimizeResponse
from ...services.workflow_service import stream_resume_workflow
from ...utils.token_guard import enforce_payload_limit
from ...utils.token_utils import enforce_token_limit
from ...core.model_limits import MODEL_LIMITS
from ...services.rate_limiter import is_rate_limited
from ...services.workflow_service import run_resume_workflow
from ...utils.cache import (
    make_cache_key,
    get_cached_result,
    set_cached_result,
)
from ...utils.timing import Timer
from ...core.exceptions import SystemFailure
from ...services.job_service import JobStatus, set_job_status, get_job_status
from app.core.idempotency import (
    get_idempotent_job,
    set_idempotent_job,
)
from ...services.load_control import (
    can_accept_job,
    increment_active_jobs,
    decrement_active_jobs,
)
from ...observability.metrics import (
    API_REQUESTS_TOTAL,
    API_ERRORS_TOTAL,
    API_REQUEST_DURATION,
)
from ...observability.metrics import (
    JOBS_ACTIVE,
    JOBS_STARTED_TOTAL,
    JOBS_COMPLETED_TOTAL,
    JOBS_FAILED_TOTAL,
    JOB_DURATION_SECONDS,
)
from time import perf_counter
from ...utils.file_parsers import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_doc,
)
from ...services.universal_scraper import get_job_description_from_url
from ...utils.text_cleaners import clean_resume_response
from ...services.cold_email_service import generate_cold_email


logger = logging.getLogger(__name__)

MODEL_NAME = "llama-3.1-8b-instant"
LIMITS = MODEL_LIMITS[MODEL_NAME]
MAX_INPUT = LIMITS["max_input_tokens"] - LIMITS["safety_margin"]

router = APIRouter(prefix="/optimize", tags=["Resume"])


@router.post("", response_model=ResumeOptimizeResponse)
def optimize_resume(payload: ResumeOptimizeRequest, request: Request):
    logger.info("[API] Optimize resume request started")

    client_ip = request.client.host

    if is_rate_limited(client_ip):
        logger.warning(f"[API] Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429, detail="Too many requests. Please try again later."
        )

    cache_payload = {
        "resume": payload.resume_text,
        "jd": payload.job_description,
        "format": payload.resume_format,
    }

    cache_key = make_cache_key(cache_payload)

    with Timer("cache_lookup"):
        cached = get_cached_result(cache_key)

    if cached:
        logger.info("[API] Returning cached result, cache_hit")
        return cached

    job_description = enforce_token_limit(
        payload.job_description, max_tokens=MAX_INPUT // 2
    )

    resume_text = enforce_token_limit(payload.resume_text, max_tokens=MAX_INPUT // 2)

    initial_state = {
        "job_description_raw": job_description,
        "resume_raw_content": resume_text,
        "resume_format": payload.resume_format,
    }

    with Timer("workflow_execution"):
        try:
            result = run_resume_workflow(initial_state)
        except Exception as e:
            logger.exception("[WORKFLOW] Failed")
            raise SystemFailure(
                message="Resume optimization workflow failed",
                details={"***REASON***": str(e)},
            )

    # Validate and clean results - ensure we have valid data
    optimized_resume = clean_resume_response(result.get("optimized_resume", ""))

    # Safety check - if resume is empty after cleaning, return original
    if not optimized_resume or len(optimized_resume.strip()) < 50:
        logger.warning("[API] Optimized resume empty or too short, using original")
        optimized_resume = resume_text

    # Ensure ATS scores are present and valid
    old_ats_score = result.get("old_ats_score")
    new_ats_score = result.get("new_ats_score")

    # Validate ATS scores are integers between 0-100
    if old_ats_score is not None and not isinstance(old_ats_score, int):
        try:
            old_ats_score = int(old_ats_score)
        except (ValueError, TypeError):
            logger.warning("[API] Invalid old_ats_score format")
            old_ats_score = None

    if new_ats_score is not None and not isinstance(new_ats_score, int):
        try:
            new_ats_score = int(new_ats_score)
        except (ValueError, TypeError):
            logger.warning("[API] Invalid new_ats_score format")
            new_ats_score = None

    # Ensure scores are within valid range
    if old_ats_score is not None and (old_ats_score < 0 or old_ats_score > 100):
        logger.warning(f"[API] old_ats_score out of range: {old_ats_score}")
        old_ats_score = None

    if new_ats_score is not None and (new_ats_score < 0 or new_ats_score > 100):
        logger.warning(f"[API] new_ats_score out of range: {new_ats_score}")
        new_ats_score = None

    response = ResumeOptimizeResponse(
        optimized_resume=optimized_resume,
        cover_letter=result.get("cover_letter_text", ""),
        old_ats_score=old_ats_score,
        new_ats_score=new_ats_score,
        extracted_keywords=result.get("extracted_keywords", []),
    )

    set_cached_result(cache_key, response.model_dump())

    logger.info("[API] Optimize resume request completed")
    return response


@router.post("/stream")  # For SSE streaming (server sent event)
def optimize_resume_stream(payload: ResumeOptimizeRequest, request: Request):
    logger.info("[API] Stream optimize resume request started")

    client_ip = request.client.host

    if is_rate_limited(client_ip):
        logger.warning(f"[API] Stream rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429, detail="Too many requests. Please try again later."
        )

    def event_generator():
        # Pre-flight token guard
        initial_state = {
            "job_description_raw": enforce_payload_limit(payload.job_description),
            "resume_raw_content": enforce_payload_limit(payload.resume_text),
            "resume_format": payload.resume_format,
        }

        # Stream workflow steps
        try:
            for step in stream_resume_workflow(initial_state, thread_id="sse_call"):
                yield f"data: {json.dumps(step)}\n\n"

            # 3️⃣ Explicit completion signal
            yield f"data: {json.dumps({'event': 'completed'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    logger.info("[API] Stream optimize resume request completed")
    return StreamingResponse(event_generator(), media_type="text/event-stream")


class OptimizeAsyncRequest(BaseModel):
    job_description: Optional[str] = None
    resume_text: str = Field(min_length=50)
    resume_format: str = "markdown"
    services: List[str] = Field(default_factory=list)
    cold_email_sender_name: Optional[str] = None
    cold_email_sender_email: Optional[str] = None
    cold_email_recipient_name: Optional[str] = None
    cold_email_recipient_email: Optional[str] = None
    cold_email_company_name: Optional[str] = None
    cold_email_target_role: Optional[str] = None


@router.post("/async")
def optimize_resume_async(
    payload: OptimizeAsyncRequest,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    retry: bool = Query(False),
):
    request_id = getattr(request.state, "request_id", None)

    logger.info(
        "[API] Async optimize request received",
        extra={"request_id": request_id},
    )

    # 1️ IDEMPOTENCY REPLAY CHECK (VERY TOP)
    idem_entry = get_idempotent_job(idempotency_key)

    parent_job_id = None

    if idem_entry:
        status = idem_entry["status"]
        parent_job_id = idem_entry["job_id"]

        if status == JobStatus.FAILED and retry:
            logger.info(
                f"[RETRY] Retrying failed job {parent_job_id}",
                extra={"request_id": request_id},
            )
        else:
            logger.info(
                "[IDEMPOTENCY] Replay detected",
                extra={"request_id": request_id},
            )
            return {
                "job_id": parent_job_id,
                "status": status,
                "source": "idempotent_replay",
            }

    if not can_accept_job():
        logger.warning(
            "[LOAD CONTROL] System overloaded - rejecting async job",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=429,
            detail="System is currently overloaded. Please try again later.",
        )

    # 2️ CREATE NEW JOB
    job_id = f"job-{uuid.uuid4().hex}"

    initial_state = {
        "job_description_raw": payload.job_description,
        "resume_raw_content": payload.resume_text,
        "resume_format": payload.resume_format,
    }

    # Store initial idempotency state
    set_idempotent_job(
        idempotency_key,
        job_id,
        JobStatus.PENDING,
    )

    set_job_status(
        job_id,
        JobStatus.PENDING,
        parent_job_id=parent_job_id,
    )

    # 3️ BACKGROUND EXECUTION
    def background_runner():
        job_start = perf_counter()
        try:
            ### JOB STARTED ###
            increment_active_jobs()
            JOBS_STARTED_TOTAL.inc()
            JOBS_ACTIVE.inc()

            API_REQUESTS_TOTAL.labels(
                "/optimize/async",
                "POST",
                "started",
            ).inc()

            set_job_status(job_id, JobStatus.RUNNING, idempotency_key=idempotency_key)
            set_idempotent_job(idempotency_key, job_id, JobStatus.RUNNING)

            services_requested = payload.services or []
            result = {}

            # Only run workflow if resume or cover letter requested
            if "resume" in services_requested or "cover" in services_requested:
                logger.info(
                    f"[ASYNC] Running workflow for services: {services_requested}"
                )
                with API_REQUEST_DURATION.labels(
                    "/optimize/async",
                    "POST",
                ).time():
                    # Pass services to workflow so it can conditionally generate
                    initial_state["services_requested"] = services_requested
                    result = run_resume_workflow(initial_state)
            else:
                # If only cold email, don't run workflow, initialize empty result
                logger.info("[ASYNC] Only cold email requested, skipping workflow")
                result = {
                    "optimized_resume": "",
                    "cover_letter": "",
                    "extracted_keywords": [],
                    "old_ats_score": None,
                    "new_ats_score": None,
                }

            # If cold email requested, generate it and attach to result
            try:
                if (
                    "coldEmail" in services_requested
                    or "cold_email" in services_requested
                ):
                    logger.info("[ASYNC] Cold email requested — generating")
                    cold_email_text = generate_cold_email(
                        resume_text=initial_state.get("resume_raw_content", ""),
                        job_description=initial_state.get("job_description_raw", ""),
                        sender_name=payload.cold_email_sender_name,
                        sender_email=payload.cold_email_sender_email,
                        recipient_name=payload.cold_email_recipient_name,
                        recipient_email=payload.cold_email_recipient_email,
                        company_name=payload.cold_email_company_name,
                        target_role=payload.cold_email_target_role,
                    )
                    # Attach to result dictionary
                    if isinstance(result, dict):
                        result["cold_email"] = cold_email_text
                    else:
                        result = {
                            "optimized_resume": "",
                            "cover_letter": "",
                            "cold_email": cold_email_text,
                        }
            except Exception:
                logger.exception(
                    "[ASYNC] Cold email generation failed; continuing without it"
                )

            ### JOB COMPLETED ###
            API_REQUESTS_TOTAL.labels(
                "/optimize/async",
                "POST",
                "success",
            ).inc()

            set_job_status(
                job_id,
                JobStatus.SUCCESS,
                result=result,
                idempotency_key=idempotency_key,
            )

            JOBS_COMPLETED_TOTAL.inc()

            set_idempotent_job(
                idempotency_key,
                job_id,
                JobStatus.SUCCESS,
            )

        except Exception as e:
            ### JOB FAILED ###
            logger.exception(
                "[ASYNC_JOB] Failed",
                extra={"request_id": request_id},
            )

            API_ERRORS_TOTAL.labels(
                "/optimize/async",
                "POST",
            ).inc()

            API_REQUESTS_TOTAL.labels(
                "/optimize/async",
                "POST",
                "failed",
            ).inc()

            JOBS_FAILED_TOTAL.inc()

            set_job_status(
                job_id,
                JobStatus.FAILED,
                error=str(e),
                idempotency_key=idempotency_key,
            )

            set_idempotent_job(
                idempotency_key,
                job_id,
                JobStatus.FAILED,
            )
        finally:
            duration = perf_counter() - job_start
            JOB_DURATION_SECONDS.observe(duration)
            JOBS_ACTIVE.dec()
            decrement_active_jobs()

    threading.Thread(
        target=background_runner,
        daemon=True,
    ).start()

    # 4️ RETURN ACCEPTED RESPONSE
    return {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "retry_of": parent_job_id,
    }


@router.get("/status/{job_id}")
def get_async_status(job_id: str):
    job = get_job_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.get("/pdf/{job_id}")
def download_resume_pdf(
    job_id: str,
    template: str = Query("modern", enum=["modern", "classic", "minimalist"]),
):
    """
    Download the optimized resume as a styled PDF.
    Requires the job to have completed successfully with resume_json.
    """
    from ...services.pdf_service import render_resume_pdf
    import io

    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") != JobStatus.SUCCESS:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete. Current status: {job.get('status')}",
        )

    result = job.get("result", {})
    resume_json = result.get("resume_json")

    if not resume_json:
        raise HTTPException(
            status_code=400,
            detail="No structured resume data available for this job. "
            "Resume JSON extraction may have failed.",
        )

    try:
        pdf_bytes = render_resume_pdf(resume_json, template=template)
    except Exception as e:
        logger.exception(f"[PDF] Rendering failed for job {job_id}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {str(e)}",
        )

    # Extract name for filename
    contact_name = resume_json.get("contact", {}).get("name", "resume")
    safe_name = "".join(c for c in contact_name if c.isalnum() or c in " -_").strip()
    safe_name = safe_name.replace(" ", "_") or "resume"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}_{template}.pdf"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )

@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    filename = file.filename.lower()

    if not filename.endswith((".pdf", ".docx", ".doc")):
        raise HTTPException(
            status_code=400, detail="Unsupported file type. Upload PDF, DOCX, or DOC."
        )

    try:
        # 1️⃣ Save uploaded file to temp location
        suffix = os.path.splitext(filename)[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 2️⃣ Extract text using YOUR existing utilities
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(tmp_path)

        elif filename.endswith(".docx"):
            text = extract_text_from_docx(tmp_path)

        elif filename.endswith(".doc"):
            text = extract_text_from_doc(tmp_path)

        else:
            raise HTTPException(status_code=400, detail="Unsupported file")

        # 3️⃣ Cleanup temp file
        os.remove(tmp_path)

        # 4️⃣ Validate extracted text
        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400, detail="Could not extract sufficient text from file"
            )

        return {"filename": file.filename, "text": text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {str(e)}")


class JDFromURLRequest(BaseModel):
    url: str = Field(..., min_length=10, example="https://example.com/...")


def extract_with_requests(url: str) -> str:
    """Fallback extraction using requests + BeautifulSoup"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text[:50000]  # Limit to 50k chars

    except Exception as e:
        logger.error(f"[REQUESTS_FALLBACK] Failed: {e}")
        return ""


# @router.post("/jd-from-url")
# def extract_jd_from_url(payload: JDFromURLRequest):
#     """
#     Accepts a job description URL
#     Fetches + cleans readable JD text
#     """
#     url = payload.url

#     if not url.startswith("http"):
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid URL"
#         )

#     try:
#         logger.info(f"[JD URL] Fetching content from: {url}")

#         # Method 1: Try Tavily first
#         raw_content = get_url_content_from_tavily(url)

#         # Method 2: If Tavily fails, try requests fallback
#         if not raw_content or len(raw_content.strip()) < 100:
#             logger.info(f"[JD URL] Tavily failed or returned short content, trying fallback...")
#             raw_content = extract_with_requests(url)

#         logger.info(f"[JD URL] Raw content length: {len(raw_content)}")
#         if raw_content:
#             logger.info(f"[JD URL] First 500 chars: {raw_content[:500]}")
#         else:
#             logger.error(f"[JD URL] No content fetched from any method")

#         if not raw_content or len(raw_content.strip()) < 100:
#             logger.error(f"[JD URL] Insufficient content fetched: {len(raw_content) if raw_content else 0} chars")
#             raise HTTPException(
#                 status_code=400,
#                 detail="Failed to fetch job description from URL or content too short"
#             )

#         cleaned_text = parse_markdown_to_plain_text(raw_content)
#         cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

#         logger.info(f"[JD URL] Cleaned text length: {len(cleaned_text)}")
#         logger.info(f"[JD URL] Cleaned first 300 chars: {cleaned_text[:300]}")

#         if len(cleaned_text.strip()) < 100:
#             logger.error(f"[JD URL] Cleaned text too short: {len(cleaned_text.strip())} chars")
#             raise HTTPException(
#                 status_code=400,
#                 detail="Extracted job description is too short"
#             )

#         return {
#             "url": url,
#             "job_description": cleaned_text
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception(f"[JD URL] Unexpected error processing URL: {url}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Internal server error while processing URL: {str(e)}"
#         )


@router.post("/jd-from-url")
async def extract_jd_from_url(payload: JDFromURLRequest):
    """
    Universal job description extractor for ANY website.
    Replaces the old Tavily-based version.
    """
    url = payload.url

    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        logger.info(f"[JD URL] Starting universal scraper for: {url}")

        # Use the new universal scraper
        result = await get_job_description_from_url(url)

        job_description = result.get("job_description", "")
        logger.info(f"[JD URL] Extracted {len(job_description)} chars")

        if not job_description or len(job_description.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient job description from URL",
            )

        # Optional: Add your existing cleaning if needed
        from ...utils.text_cleaners import parse_markdown_to_plain_text

        cleaned_description = parse_markdown_to_plain_text(job_description)

        return {
            "url": url,
            "job_description": cleaned_description,
            "job_title": result.get("job_title", ""),
            "company": result.get("company", ""),
            "location": result.get("location", ""),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[JD URL] Error processing {url}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to extract job description: {str(e)}"
        )
