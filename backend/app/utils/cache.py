import hashlib
import json
import logging
from ..core.redis import redis_client

logger = logging.getLogger(__name__)

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
        logger.info(f"[CACHE] Hit for key: {key}")
        return json.loads(data)
    logger.info(f"[CACHE] Miss for key: {key}")
    return None

def set_cached_result(key: str, value: dict):
    redis_client.setex(
        key, 
        CACHE_TTL, 
        json.dumps(value)
    )
    logger.info(f"[CACHE] Set for key: {key}")
