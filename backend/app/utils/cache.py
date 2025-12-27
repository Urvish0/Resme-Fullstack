import hashlib
import json
from ..core.redis import redis_client

CACHE_TTL = 60 * 60

def make_cache_key(payload: dict) -> str:
    """
    Creates a deterministic cache key for a request.
    """
    serialized = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    return f"resume_cache:{digest}"

def get_cached_result(key: str):
    data = redis_client.get(key)
    if data:
        return json.loads(data)
    return None

def set_cached_result(key: str, value: dict):
    redis_client.setex(
        key, 
        CACHE_TTL, 
        json.dumps(value)
    )
