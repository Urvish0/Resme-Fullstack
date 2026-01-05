from fastapi import APIRouter
from ...core.redis import redis_client

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "ok"
    }
    
@router.get("/ready")
def readiness_check():
    try:
        redis_client.ping()
        return {"status": "ready"}
    except Exception:
        return {"status": "not ready"}
