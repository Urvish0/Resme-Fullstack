import json
from app.core.redis import redis_client

IDEMPOTENCY_TTL = 60 * 60  # 1 hour


def get_idempotent_job(idem_key: str):
    data = redis_client.get(f"idem:{idem_key}")
    return json.loads(data) if data else None


def set_idempotent_job(idem_key: str, job_id: str, status: str):
    redis_client.setex(
        f"idem:{idem_key}",
        IDEMPOTENCY_TTL,
        json.dumps({"job_id": job_id, "status": status}),
    )
