from fastapi import APIRouter,HTTPException, Request, Header, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import logging
import uuid
import threading

from ...schemas.resume import (
    ResumeOptimizeRequest,
    ResumeOptimizeResponse
)
from ...services.resume_service import prepare_resume_state
from ...services.workflow_service import stream_resume_workflow
from ...schemas.resume import ResumeOptimizeRequest
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
from ...services.background__tasks import run_resume_job
from ...services.job_service import JobStatus, set_job_status, get_job_status
from backend.app.core.idempotency import (
    get_idempotent_job,
    set_idempotent_job,
)
from ...services.job_service import JobStatus 



logger = logging.getLogger(__name__)

MODEL_NAME = "llama-3.1-8b-instant"
LIMITS = MODEL_LIMITS[MODEL_NAME]
MAX_INPUT = LIMITS["max_input_tokens"] - LIMITS["safety_margin"]

router = APIRouter(prefix="/optimize", tags=["Resume"])

with Timer("total_request"):
    @router.post("", response_model=ResumeOptimizeResponse)
    def optimize_resume(payload: ResumeOptimizeRequest, request: Request):
        
        logger.info("[API] Optimize resume request started")
        
        client_ip = request.client.host

        if is_rate_limited(client_ip):
            logger.warning(f"[API] Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
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
            payload.job_description,
            max_tokens=MAX_INPUT // 2
        )

        resume_text = enforce_token_limit(
            payload.resume_text,
            max_tokens=MAX_INPUT // 2
        )
        
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
                    details={"***REASON***": str(e)}
                )

        response = ResumeOptimizeResponse(
            optimized_resume=result.get("edited_resume_content", ""),
            cover_letter=result.get("cover_letter_text"),
            old_ats_score=result.get("old_ats_score"),
            new_ats_score=result.get("new_ats_score"),
            extracted_keywords=result.get("extracted_keywords", [])
        )

        set_cached_result(cache_key, response.model_dump())
        
        logger.info("[API] Optimize resume request completed")
        return response

@router.post("/stream") #For SSE streaming (server sent event)
def optimize_resume_stream(payload: ResumeOptimizeRequest, request: Request):

    logger.info("[API] Stream optimize resume request started")
    
    client_ip = request.client.host
    
    if is_rate_limited(client_ip):
        logger.warning(f"[API] Stream rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
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
            for step in stream_resume_workflow(
                initial_state,
                thread_id="sse_call"
            ):
                yield f"data: {json.dumps(step)}\n\n"

            # 3️⃣ Explicit completion signal
            yield f"data: {json.dumps({'event': 'completed'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    logger.info("[API] Stream optimize resume request completed")
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

class OptimizeAsyncRequest(BaseModel):
    job_description: str = Field(min_length=50)
    resume_text: str = Field(min_length=50)
    resume_format: str = "markdown"


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
                f"[IDEMPOTENCY] Replay detected",
                extra={"request_id": request_id},
            )
            return {
                "job_id": parent_job_id,
                "status": status,
                "source": "idempotent_replay",
            }

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
        parent_job_id,
    )

    # 3️ BACKGROUND EXECUTION
    def background_runner():
        try:
            set_job_status(job_id, JobStatus.RUNNING)
            set_idempotent_job(
                idempotency_key,
                job_id,
                JobStatus.RUNNING,
            )

            result = run_resume_workflow(initial_state)

            set_job_status(
                job_id,
                JobStatus.SUCCESS,
                result=result,
            )

            set_idempotent_job(
                idempotency_key,
                job_id,
                JobStatus.SUCCESS,
            )

        except Exception as e:
            logger.exception(
                "[ASYNC_JOB] Failed",
                extra={"request_id": request_id},
            )

            set_job_status(
                job_id,
                JobStatus.FAILED,
                error=str(e),
            )

            set_idempotent_job(
                idempotency_key,
                job_id,
                JobStatus.FAILED,
            )

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
