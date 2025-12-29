import hashlib
import json
import logging
from redis.exceptions import RedisError
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
    try:
        data = redis_client.get(key)
        if data:
            logger.info(f"[CACHE] Hit for key: {key}")
            return json.loads(data)
        logger.info(f"[CACHE] Miss for key: {key}")
        return None
    except RedisError as e:
        logger.warning(f"[CACHE] Redis GET failed for key {key}", extra={"error_type": "retryable_error"})
        return None

def set_cached_result(key: str, value: dict):
    try:
        redis_client.setex(
            key, 
            CACHE_TTL, 
            json.dumps(value)
        )
        logger.info(f"[CACHE] Set for key: {key}")
    except RedisError as e:
        logger.warning(f"[CACHE] Redis SET failed for key {key}", extra={"error_type": "retryable_error"})
