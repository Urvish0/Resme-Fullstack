import logging
from ..services.job_service import set_job_status, JobStatus
from ..services.workflow_service import run_resume_workflow
from ..core.idempotency import set_idempotent_job

logger = logging.getLogger(__name__)

def run_resume_job(job_id: str, state: dict, idempotency_key: str):
    try:
        # RUNNING
        set_job_status(job_id, JobStatus.RUNNING)
        set_idempotent_job(idempotency_key, job_id, JobStatus.RUNNING)

        result = run_resume_workflow(state)

        # SUCCESS
        set_job_status(
            job_id,
            JobStatus.SUCCESS,
            result=result,
        )
        set_idempotent_job(idempotency_key, job_id, JobStatus.SUCCESS)

    except Exception as e:
        logger.exception("Async resume job failed")

        set_job_status(
            job_id,
            JobStatus.FAILED,
            error=str(e),
        )
        set_idempotent_job(idempotency_key, job_id, JobStatus.FAILED)
