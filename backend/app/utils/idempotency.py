import hashlib
import json
from ..core.redis import redis_client

IDEMPOTENCY_TTL = 60 * 10


def compute_idempotency_key(
    client_key: str,
    payload: str,
) -> str:
    """
    Combines client-provided key+request body to create a stable idempotency hash.
    """
    normalized = json.dumps(payload, sort_keys=True)
    raw = f"{client_key}:{normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_idempotent_result(key: str):
    return redis_client.get(f"idempotency:{key}")


def set_idempotent_result(key: str, result: dict):
    redis_client.setex(f"idempotency:{key}", IDEMPOTENCY_TTL, json.dumps(result))
