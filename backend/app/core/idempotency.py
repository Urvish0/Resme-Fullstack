import json
import logging
from app.core.redis import redis_client, REDIS_AVAILABLE

logger = logging.getLogger(__name__)

IDEMPOTENCY_TTL = 60 * 60  # 1 hour


def get_idempotent_job(idem_key: str):
    """Get idempotent job from Redis if available"""
    if not REDIS_AVAILABLE or not redis_client:
        return None
    
    try:
        data = redis_client.get(f"idem:{idem_key}")
        return json.loads(data) if data else None
    except Exception as e:
        logger.warning(f"Failed to get idempotent job: {e}")
        return None


def set_idempotent_job(idem_key: str, job_id: str, status: str):
    """Set idempotent job in Redis if available"""
    if not REDIS_AVAILABLE or not redis_client:
        return
    
    try:
        redis_client.setex(
            f"idem:{idem_key}",
            IDEMPOTENCY_TTL,
            json.dumps({"job_id": job_id, "status": status}),
        )
    except Exception as e:
        logger.warning(f"Failed to set idempotent job: {e}")
