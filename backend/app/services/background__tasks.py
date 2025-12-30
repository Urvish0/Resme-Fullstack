import logging
from ..services.job_service import set_job_status, JobStatus
from ..services.workflow_service import run_resume_workflow

logger = logging.getLogger(__name__)


def run_resume_job(job_id: str, state: dict):
    try:
        set_job_status(job_id, JobStatus.RUNNING)

        result = run_resume_workflow(state)

        set_job_status(
            job_id,
            JobStatus.SUCCESS,
            result=result,
        )

    except Exception as e:
        logger.exception("Async resume job failed")

        set_job_status(
            job_id,
            JobStatus.FAILED,
            error=str(e),
        )
