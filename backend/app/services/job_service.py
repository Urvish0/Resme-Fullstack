import json
import logging
from enum import Enum
from ..core.redis import redis_client, REDIS_AVAILABLE
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


JOB_TTL_SECONDS = 60 * 10  # 10 minutes


def set_job_status(
    job_id: str,
    status: JobStatus,
    result: dict | None = None,
    error: str | None = None,
    idempotency_key: str | None = None,
    parent_job_id: str | None = None,
):
    if not REDIS_AVAILABLE or not redis_client:
        return
    
    try:
        payload = {
            "status": status,
            "result": result,
            "error": error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if idempotency_key:
            payload["idempotency_key"] = idempotency_key

        if parent_job_id:
            payload["parent_job_id"] = parent_job_id

        redis_client.setex(
            f"job:{job_id}",
            JOB_TTL_SECONDS,
            json.dumps(payload),
        )
    except Exception as e:
        logger.warning(f"Failed to set job status: {e}")


def get_job_status(job_id: str) -> dict | None:
    if not REDIS_AVAILABLE or not redis_client:
        return None
    
    try:
        raw = redis_client.get(f"job:{job_id}")
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Failed to get job status: {e}")
        return None


def get_all_running_jobs() -> dict:
    """
    Returns all jobs currently in RUNNING state.
    NOTE: This is a Redis scan, acceptable at small scale.
    """
    if not REDIS_AVAILABLE or not redis_client:
        return {}
    
    running_jobs = {}

    try:
        for key in redis_client.scan_iter(match="job:*"):
            raw = redis_client.get(key)
            if not raw:
                continue

            data = json.loads(raw)
            if data.get("status") == JobStatus.RUNNING:
                job_id = key.replace("job:", "")
                running_jobs[job_id] = data
    except Exception as e:
        logger.warning(f"Failed to get running jobs: {e}")

    return running_jobs
