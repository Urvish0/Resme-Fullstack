from app.core.redis import redis_client, REDIS_AVAILABLE
import logging

logger = logging.getLogger(__name__)

ACTIVE_JOBS_KEY = "system:active_jobs"
MAX_ACTIVE_JOBS = 10


def can_accept_job() -> bool:
    if not REDIS_AVAILABLE or not redis_client:
        # No Redis = always accept jobs (no load control)
        return True

    try:
        current = redis_client.get(ACTIVE_JOBS_KEY)
        return int(current or 0) < MAX_ACTIVE_JOBS
    except Exception as e:
        logger.warning(f"Load control check failed: {e}")
        return True


def increment_active_jobs():
    if not REDIS_AVAILABLE or not redis_client:
        return

    try:
        redis_client.incr(ACTIVE_JOBS_KEY)
    except Exception as e:
        logger.warning(f"Failed to increment active jobs: {e}")


def decrement_active_jobs():
    if not REDIS_AVAILABLE or not redis_client:
        return

    try:
        redis_client.decr(ACTIVE_JOBS_KEY)
    except Exception as e:
        logger.warning(f"Failed to decrement active jobs: {e}")
