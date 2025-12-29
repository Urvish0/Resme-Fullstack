from fastapi import FastAPI
from .api.routes.resume import router as resume_router
from .core.logging import setup_logging

setup_logging()

app = FastAPI(
    title="ResMe API",
    version="1.0.0"
)

app.include_router(resume_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
