import os
import redis
import logging

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD if REDIS_PASSWORD else None,
        ssl=REDIS_SSL,
        decode_responses=True,
        socket_keepalive=True,
        retry_on_timeout=True,
        health_check_interval=30,
    )
    # Test connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis unavailable: {e}. Proceeding without caching.")
    redis_client = None
    REDIS_AVAILABLE = False
