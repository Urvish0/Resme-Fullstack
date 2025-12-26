from fastapi import APIRouter, HTTPException
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

from ...services.workflow_service import run_resume_workflow 

MODEL_NAME = "llama-3.1-8b-instant"
LIMITS = MODEL_LIMITS[MODEL_NAME]

MAX_INPUT = LIMITS["max_input_tokens"] - LIMITS["safety_margin"]

router = APIRouter(prefix="/optimize", tags=["Resume"])
@router.post("", response_model=ResumeOptimizeResponse)
def optimize_resume(payload: ResumeOptimizeRequest):
    job_description = enforce_token_limit(
        payload.job_description,
        max_tokens=MAX_INPUT // 2
    )

    resume_text = enforce_token_limit(
        payload.resume_text,
        max_tokens=MAX_INPUT // 2
    )
    
    try:
        initial_state = prepare_resume_state(
            job_description_raw=job_description,
            resume_file=None,
            resume_raw_content=resume_text,
            resume_format=payload.resume_format
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Run workflow (non-streaming)
    final_state = None 
    
    for step in stream_resume_workflow(initial_state, thread_id="api_call"):
        for key, value in step.items():
            if key != "__end__":
                final_state = value
                
    if not final_state:
        raise HTTPException(status_code=500, detail="Workflow failed")
    
    return ResumeOptimizeResponse(
        optimized_resume=final_state.get("edited_resume_content", ""),
        cover_letter=final_state.get("cover_letter_text"),
        old_ats_score=final_state.get("old_ats_score"),
        new_ats_score=final_state.get("new_ats_score"),
        extracted_keywords=final_state.get("extracted_keywords", [])
    )
    
router = APIRouter()
@router.post("/optimize/stream") #For SSE streaming (server sent event)
def optimize_resume_stream(payload: ResumeOptimizeRequest):

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

router = APIRouter()

@router.post("/optimize", response_model=ResumeOptimizeResponse)
def optimize_resume(payload: ResumeOptimizeRequest):
    initial_state = {
        "job_description_raw": payload.job_description,
        "resume_raw_content": payload.resume_text,
        "resume_format": payload.resume_format,
    }

    result = run_resume_workflow(initial_state)

    return {
        "optimized_resume": result.get("optimized_resume"),
        "cover_letter": result.get("cover_letter"),
        "old_ats_score": result.get("old_ats_score"),
        "new_ats_score": result.get("new_ats_score"),
        "extracted_keywords": result.get("extracted_keywords", []),
    }