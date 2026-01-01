from fastapi import APIRouter, HTTPException, Request, Header
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
from ...utils.idempotency import (
    compute_idempotency_key,
    get_idempotent_result,
    set_idempotent_result,
)

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
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    ):
    
    request_payload = payload.model_dump()
    
    idem_hash = compute_idempotency_key(
        idempotency_key,
        request_payload,
    )
    
    cached = get_idempotent_result(idem_hash)
    if cached:
        logger.info("[API] Returning idempotent cached result")
        return json.loads(cached)
    
    job_id = f"job-{uuid.uuid4().hex}"

    initial_state = {
        "job_description_raw": payload.job_description,
        "resume_raw_content": payload.resume_text,
        "resume_format": payload.resume_format,
    }

    # Set initial job state
    set_job_status(job_id, JobStatus.PENDING)

    # Run in background thread
    thread = threading.Thread(
        target=run_resume_job,
        args=(job_id, initial_state),
        daemon=True,
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "accepted",
    }

@router.get("/status/{job_id}")
def get_async_status(job_id: str):
    job = get_job_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job
