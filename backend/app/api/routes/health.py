from fastapi import APIRouter
from ...core.redis import redis_client, REDIS_AVAILABLE

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/ready")
def readiness_check():
    if not REDIS_AVAILABLE:
        return {"status": "ready", "note": "running without Redis"}
    
    try:
        redis_client.ping()
        return {"status": "ready"}
    except Exception:
        return {"status": "degraded", "note": "Redis unavailable"}
