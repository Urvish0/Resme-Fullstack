import os
import redis
import logging

logger = logging.getLogger(__name__)

# Try to get Redis URL from Upstash first (recommended for Render)
REDIS_URL = os.getenv("REDIS_URL")

# Fall back to individual host/port config for local development
if not REDIS_URL:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"
    REDIS_URL = f"redis://{':' + REDIS_PASSWORD + '@' if REDIS_PASSWORD else ''}{REDIS_HOST}:{REDIS_PORT}"
    if REDIS_SSL:
        REDIS_URL = REDIS_URL.replace("redis://", "rediss://")

try:
    redis_client = redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_keepalive=True,
        retry_on_timeout=True,
        health_check_interval=30,
    )
    # Test connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connection established successfully")
except Exception as e:
    logger.warning(f"Redis unavailable: {e}. Proceeding without caching.")
    redis_client = None
    REDIS_AVAILABLE = False
