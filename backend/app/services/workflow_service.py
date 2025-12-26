from asyncio.log import logger
from typing import Dict, Any, Generator
import json
import hashlib
from ..workflows.resume_graph import build_resume_graph
from ..utils.fingerprint import make_request_fingerprint
from ..core.redis import redis_client

CACHE_TTL_SECONDS = 60 * 60  # 1 hour

def make_cache_key(initial_state: dict) -> str:
    payload = json.dumps(initial_state, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"resume:optimize:{digest}"


def collect_final_result(graph, initial_state: dict) -> dict:
    """
    Runs the graph in streaming mode but aggregates final outputs
    for blocking API consumers.
    """
    final_result = {
        "optimized_resume": "",
        "cover_letter": "",
        "extracted_keywords": [],
        "old_ats_score": None,
        "new_ats_score": None,
    }

    for step in graph.stream(
        initial_state,
        {"configurable": {"thread_id": "blocking_collector"}}
    ):
        if not isinstance(step, dict):
            continue

        event = step.get("event")

        if event == "resume_generation":
            final_result["optimized_resume"] += step.get("delta", "")

        elif event == "cover_letter_generation":
            final_result["cover_letter"] += step.get("delta", "")

        elif event == "keyword_extraction":
            final_result["extracted_keywords"].extend(
                step.get("keywords_partial", [])
            )

        elif event == "ats_scoring":
            final_result["old_ats_score"] = step.get("old_score")
            final_result["new_ats_score"] = step.get("new_score")

    # Deduplicate keywords
    final_result["extracted_keywords"] = list(
        set(final_result["extracted_keywords"])
    )

    return final_result

def run_resume_workflow(initial_state: dict) -> dict:
    cache_key = make_cache_key(initial_state)

    # 1️⃣ Try cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # 2️⃣ Run workflow
    graph = build_resume_graph()
    result = collect_final_result(graph, initial_state)

    # 3️⃣ Store in cache
    redis_client.setex(
        cache_key,
        CACHE_TTL_SECONDS,
        json.dumps(result)
    )

    return result

def stream_resume_workflow(
    initial_state: Dict[str, Any],
    thread_id: str
) -> Generator[Dict[str, Any], None, None]:

    # 1. Generate fingerprint for caching
    fingerprint = make_request_fingerprint({
        "job_description": initial_state.get("job_description_raw"),
        "resume_text": initial_state.get("resume_raw_content"),
        "resume_format": initial_state.get("resume_format"),
    })
    
    cache_key = f"resume:result:{fingerprint}"
    
    logger.info(f"[CACHE] Key: {cache_key}")

    # 2. Check cache    
    cached = redis_client.get(cache_key)
    
    if cached:
        logger.info("[CACHE] HIT — returning cached result")
    else:
        logger.info("[CACHE] MISS — running workflow")
        
        
    if cached:
        # sending cached final result as a stream event
        yield {
            "event": "cached_result",
            "data": json.loads(cached)
        }

    # 3. Running the workflow if cache miss
    graph = build_resume_graph()
    final_result = None

    try:
        for step in graph.stream(
            initial_state,
            {"configurable": {"thread_id": thread_id}}
        ):
            yield step
            
            if step.get("event") == "final_result":
                final_result = step.get("data")
                
        # 4. Save to redis 
        if final_result:
            redis_client.setex(
                cache_key,
                60 * 60, # 1 hr TTL
                json.dumps(final_result)
            )

    except Exception as e:
        raise RuntimeError(f"Workflow streaming failed: {str(e)}")

