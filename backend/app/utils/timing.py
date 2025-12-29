import time 
import logging 

logger = logging.getLogger(__name__)

class Timer:
    def __init__(self, name: str):
        self.name = name 
        self.start = None 
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self 
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        duration_ms = (time.perf_counter() - self.start) * 1000
        logger.info(f"[TIMING] {self.name} - {round(duration_ms, 2)}ms")