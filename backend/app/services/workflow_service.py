from typing import Dict, Any, Generator
import json
import hashlib
from redis.exceptions import RedisError
import logging
from ..workflows.resume_graph import build_resume_graph
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from ..utils.fingerprint import make_request_fingerprint
from ..core.redis import redis_client
from ..core.exceptions import SystemFailure
from ..utils.cache import get_session_memory, update_session_memory
from ..core.supabase import SupabaseService

logger = logging.getLogger(__name__)
logger.info("Workflow Service initialized.")

CACHE_VERSION = "v1"
CACHE_TTL_SECONDS = 60 * 60  # 1 hour

MAX_RESUME_CHARS = 6000
MAX_JD_CHARS = 6000


def trim_text(text: str, max_chars: int) -> str:
    """
    Safely trim text without breaking structure.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def make_cache_key(initial_state: dict) -> str:
    payload = json.dumps(initial_state, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"resume:optimize:{CACHE_VERSION}:{digest}"


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
        initial_state, {"configurable": {"thread_id": "blocking_collector"}}
    ):
        if not isinstance(step, dict):
            continue

        event = step.get("event")

        if event == "resume_generation":
            final_result["optimized_resume"] += step.get("delta", "")

        elif event == "cover_letter_generation":
            final_result["cover_letter"] += step.get("delta", "")

        elif event == "keyword_extraction":
            final_result["extracted_keywords"].extend(step.get("keywords_partial", []))

        elif event == "ats_scoring":
            final_result["old_ats_score"] = step.get("old_score")
            final_result["new_ats_score"] = step.get("new_score")

    # Deduplicate keywords
    final_result["extracted_keywords"] = list(set(final_result["extracted_keywords"]))

    return final_result


def run_resume_workflow(initial_state: dict, thread_id: str = "blocking") -> dict:
    logger.info("[WORKFLOW] Started")

    initial_state["resume_raw_content"] = trim_text(
        initial_state["resume_raw_content"], MAX_RESUME_CHARS
    )

    initial_state["job_description_raw"] = trim_text(
        initial_state["job_description_raw"], MAX_JD_CHARS
    )

    cache_key = make_cache_key(initial_state)

    # 1️⃣ Cache check
    try:
        cached = redis_client.get(cache_key)
        if cached:
            logger.info("[CACHE] Hit — returning cached result")
            return json.loads(cached)
    except RedisError:
        logger.warning("[CACHE] Redis GET failed, bypassing cache")

    # 2️⃣ Run graph
    try:
        graph = build_resume_graph()
        # 1.5. Load session memory context
        session_id = initial_state.get("session_id", "default")
        session_memory = get_session_memory(session_id)
        
        config = {"configurable": {"thread_id": thread_id}}
        
        final_state = graph.invoke(
            {
                **initial_state,
                "messages": [],
                "extracted_keywords": [],
                "human_feedback": "proceed",
                "task_complete": False,
                "memory_context": session_memory,  # Inject memory
                "user_id": initial_state.get("user_id", "default_user"),
            },
            config,
        )
        
        # Phase 6.2 HITL: Check if the graph paused at an interrupt
        # When using a checkpointer, graph.invoke() returns normally even on interrupt.
        # We detect interrupts by inspecting the graph state snapshot.
        state_snapshot = graph.get_state(config)
        
        if state_snapshot and state_snapshot.next:
            # The graph has pending nodes (paused at interrupt)
            logger.info(f"[WORKFLOW] Graph paused at interrupt. Pending nodes: {state_snapshot.next}. Thread: {thread_id}")
            
            # Extract interrupt data from the state snapshot
            interrupt_data = {}
            try:
                # state_snapshot.tasks contains the interrupt info
                if hasattr(state_snapshot, 'tasks') and state_snapshot.tasks:
                    for task in state_snapshot.tasks:
                        if hasattr(task, 'interrupts') and task.interrupts:
                            for intr in task.interrupts:
                                if hasattr(intr, 'value') and isinstance(intr.value, dict):
                                    interrupt_data = intr.value
                                    break
                            if interrupt_data:
                                break
            except Exception as parse_err:
                logger.warning(f"[WORKFLOW] Failed to parse interrupt data from snapshot: {parse_err}")
            
            # Fallback: extract from the state values directly
            if not interrupt_data:
                interrupt_data = {
                    "analysis_report": final_state.get("analysis_report", ""),
                    "old_ats_score": final_state.get("old_ats_score"),
                    "extracted_keywords": final_state.get("extracted_keywords", [])[:15],
                    "message": "Analysis complete. Provide feedback to guide optimization.",
                }
            
            return {
                "__interrupted__": True,
                "thread_id": thread_id,
                "analysis_report": interrupt_data.get("analysis_report", ""),
                "old_ats_score": interrupt_data.get("old_ats_score"),
                "extracted_keywords": interrupt_data.get("extracted_keywords", []),
                "message": interrupt_data.get("message", "Awaiting feedback"),
            }
    except Exception as e:
        logger.exception("[WORKFLOW] Graph execution failed")
        raise SystemFailure(
            message="Workflow execution failed", details={"reason": str(e)}
        )

    # 3️⃣ Extract REAL outputs
    optimized_resume = final_state.get("edited_resume_content", "").strip()
    cover_letter = final_state.get("cover_letter_text", "").strip()
    old_ats_score = final_state.get("old_ats_score")
    new_ats_score = final_state.get("new_ats_score")
    extracted_keywords = final_state.get("extracted_keywords", [])
    reflection_report = final_state.get("reflection_report", "")
    resume_json = final_state.get("resume_json")

    # Log extracted values for debugging
    logger.info(
        f"[WORKFLOW] Extracted results - Resume length: {len(optimized_resume)}, "
        f"Cover letter length: {len(cover_letter)}, "
        f"Old ATS: {old_ats_score}, New ATS: {new_ats_score}, "
        f"Keywords: {len(extracted_keywords)}, "
        f"Reflection length: {len(reflection_report)}, "
        f"JSON: {'YES' if resume_json else 'NO'}"
    )

    result = {
        "optimized_resume": optimized_resume,
        "cover_letter": cover_letter,
        "old_ats_score": old_ats_score,
        "new_ats_score": new_ats_score,
        "extracted_keywords": extracted_keywords,
        "reflection_report": reflection_report,
        "resume_json": resume_json,
    }

    # 3.5. Update session memory with new insights
    update_session_memory(session_id, {
        "last_ats_score": new_ats_score,
        "last_keywords": extracted_keywords
    })

    # 3.6. Persist to Supabase (Long-Term Memory)
    user_id = initial_state.get("user_id", "default_user")
    SupabaseService.save_resume_version(
        user_id=user_id,
        content=optimized_resume,
        score=new_ats_score,
        keywords=extracted_keywords
    )

    # 4️⃣ Cache
    try:
        redis_client.setex(
            cache_key,
            CACHE_TTL_SECONDS,
            json.dumps(result),
        )
        logger.info("[CACHE] Stored result")
    except RedisError:
        logger.warning("[CACHE] Redis SET failed")

    logger.info("[WORKFLOW] Completed")
    return result


def resume_workflow_with_feedback(thread_id: str, feedback: str) -> dict:
    """
    Phase 6.2 HITL: Resumes a paused workflow after the user provides feedback.
    Uses LangGraph Command(resume=...) to unblock the interrupt() call.
    """
    logger.info(f"[WORKFLOW] Resuming with feedback. Thread: {thread_id}, Feedback: {feedback[:100]}")

    graph = build_resume_graph()

    try:
        final_state = graph.invoke(
            Command(resume=feedback),
            {"configurable": {"thread_id": thread_id}},
        )
    except GraphInterrupt:
        # Should not happen again, but handle gracefully
        logger.error("[WORKFLOW] Unexpected second interrupt during resume")
        return {"__interrupted__": True, "thread_id": thread_id, "message": "Unexpected interrupt"}
    except Exception as e:
        logger.exception("[WORKFLOW] Resume failed")
        raise SystemFailure(
            message="Workflow resume failed", details={"reason": str(e)}
        )

    # Extract results (same as run_resume_workflow)
    optimized_resume = final_state.get("edited_resume_content", "").strip()
    cover_letter = final_state.get("cover_letter_text", "").strip()
    old_ats_score = final_state.get("old_ats_score")
    new_ats_score = final_state.get("new_ats_score")
    extracted_keywords = final_state.get("extracted_keywords", [])
    reflection_report = final_state.get("reflection_report", "")
    resume_json = final_state.get("resume_json")

    logger.info(
        f"[WORKFLOW] Resumed — Resume length: {len(optimized_resume)}, "
        f"New ATS: {new_ats_score}"
    )

    return {
        "optimized_resume": optimized_resume,
        "cover_letter": cover_letter,
        "old_ats_score": old_ats_score,
        "new_ats_score": new_ats_score,
        "extracted_keywords": extracted_keywords,
        "reflection_report": reflection_report,
        "resume_json": resume_json,
    }


def stream_resume_workflow(
    initial_state: Dict[str, Any], thread_id: str
) -> Generator[Dict[str, Any], None, None]:
    logger.info("[STREAM_WORKFLOW] Started")

    # 1. Generate fingerprint for caching
    fingerprint = make_request_fingerprint(
        {
            "job_description": initial_state.get("job_description_raw"),
            "resume_text": initial_state.get("resume_raw_content"),
            "resume_format": initial_state.get("resume_format"),
        }
    )

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
        yield {"event": "cached_result", "data": json.loads(cached)}

    # 3. Running the workflow if cache miss
    try:
        graph = build_resume_graph()
    except Exception as e:
        logger.error(
            "[STREAM_WORKFLOW] Graph build failed", extra={"error_type": "system_error"}
        )
        raise SystemFailure(
            message="Workflow initialization failed", details={"reason": str(e)}
        )

    final_result = None

    try:
        for step in graph.stream(
            initial_state, {"configurable": {"thread_id": thread_id}}
        ):
            yield step

            if step.get("event") == "final_result":
                final_result = step.get("data")

        # 4. Save to redis
        if final_result:
            redis_client.setex(
                cache_key,
                60 * 60,  # 1 hr TTL
                json.dumps(final_result),
            )
            logger.info("[CACHE] Saved final result")

    except Exception as e:
        logger.exception(
            "[STREAM_WORKFLOW] Streaming failed", extra={"error_type": "system_error"}
        )
        raise SystemFailure(
            message="Workflow streaming failed", details={"reason": str(e)}
        )

    logger.info("[STREAM_WORKFLOW] Completed")
