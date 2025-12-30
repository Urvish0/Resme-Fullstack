import json
from enum import Enum
from ..core.redis import redis_client


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
):
    payload = {
        "status": status,
        "result": result,
        "error": error,
    }

    redis_client.setex(
        f"job:{job_id}",
        JOB_TTL_SECONDS,
        json.dumps(payload),
    )


def get_job_status(job_id: str) -> dict | None:
    raw = redis_client.get(f"job:{job_id}")
    if not raw:
        return None
    return json.loads(raw)
