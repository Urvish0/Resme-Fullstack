import uuid 
from fastapi import Request 
from .logging import request_id_ctx

async def request_id_middleware(request: Request, call_next):
    
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    request.state.request_id = request_id
    
    token = request_id_ctx.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
        
    response.headers["X-Request-ID"] = request_id
    return response