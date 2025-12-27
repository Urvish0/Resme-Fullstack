import time
from redis.exceptions import RedisError
from ..core.redis import redis_client

# Configuration
RATE_LIMIT = 5
RATE_WINDOW = 60  # seconds

def is_rate_limited(client_id: str) -> bool:
    """
    Uses a Sliding Window algorithm via Redis Sorted Sets.
    Returns True if the client has exceeded the limit.
    """
    key = f"rate_limit:{client_id}"
    now = time.time() 
    # The start of our window (60 seconds ago)
    window_start = now - RATE_WINDOW

    try:
        # We use a pipeline to ensure all commands are sent in one go
        pipe = redis_client.pipeline()

        # 1. Remove timestamps older than our window
        pipe.zremrangebyscore(key, 0, window_start)
        
        # 2. Add the current request timestamp to the set
        # Using the timestamp as both the 'member' and the 'score'
        pipe.zadd(key, {str(now): now})
        
        # 3. Count how many requests are in the set now
        pipe.zcard(key)
        
        # 4. Set an expiry on the whole set so it cleans up after inactivity
        pipe.expire(key, RATE_WINDOW)

        # Execute the pipeline
        results = pipe.execute()
        
        # The 3rd command in the pipe (index 2) was ZCARD
        request_count = results[2]

        return request_count > RATE_LIMIT

    except RedisError:
        # Fail open: if Redis is down, let the request through
        return False