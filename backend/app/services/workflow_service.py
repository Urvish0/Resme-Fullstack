from typing import Dict, Any, Generator
from ..workflows.resume_graph import build_resume_graph


def stream_resume_workflow(
    initial_state: Dict[str, Any],
    thread_id: str
) -> Generator[Dict[str, Any], None, None]:
    """
    Streams execution of the resume optimization workflow.
    """
    graph = build_resume_graph()

    try:
        for step in graph.stream(
            initial_state,
            {"configurable": {"thread_id": thread_id}}
        ):
            yield step

    except Exception as e:
        raise RuntimeError(f"Workflow streaming failed: {str(e)}")
