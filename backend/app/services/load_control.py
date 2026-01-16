from app.core.redis import redis_client

ACTIVE_JOBS_KEY = "system:active_jobs"
MAX_ACTIVE_JOBS = 10

def can_accept_job() -> bool:
    current = redis_client.get(ACTIVE_JOBS_KEY)
    return int(current or 0) < MAX_ACTIVE_JOBS

def increment_active_jobs():
    redis_client.incr(ACTIVE_JOBS_KEY) 
    
def decrement_active_jobs():
    redis_client.decr(ACTIVE_JOBS_KEY)
    
    
    