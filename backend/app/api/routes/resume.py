from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import json

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


MODEL_NAME = "llama-3.1-8b-instant"
LIMITS = MODEL_LIMITS[MODEL_NAME]
MAX_INPUT = LIMITS["max_input_tokens"] - LIMITS["safety_margin"]

router = APIRouter(prefix="/optimize", tags=["Resume"])

@router.post("", response_model=ResumeOptimizeResponse)
def optimize_resume(payload: ResumeOptimizeRequest, request: Request):
    
    client_ip = request.client.host

    if is_rate_limited(client_ip):
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
    cached = get_cached_result(cache_key)
    if cached:
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
    
    result = run_resume_workflow(initial_state)

    response = ResumeOptimizeResponse(
        optimized_resume=result.get("edited_resume_content", ""),
        cover_letter=result.get("cover_letter_text"),
        old_ats_score=result.get("old_ats_score"),
        new_ats_score=result.get("new_ats_score"),
        extracted_keywords=result.get("extracted_keywords", [])
    )

    set_cached_result(cache_key, response.model_dump())
    
    return response

@router.post("/stream") #For SSE streaming (server sent event)
def optimize_resume_stream(payload: ResumeOptimizeRequest, request: Request):

    client_ip = request.client.host
    
    if is_rate_limited(client_ip):
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

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
