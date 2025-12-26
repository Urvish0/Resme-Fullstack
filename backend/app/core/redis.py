import redis

redis_client = redis.Redis( # reusable Redis client instance
    host="localhost",
    port=6379,
    db=0, #default redis database
    decode_responses=True # returns strings instead of bytes   
)
