from fastapi import FastAPI
from .api.routes.resume import router as resume_router
from .core.logging import setup_logging
from .core.middleware import request_id_middleware

setup_logging()


app = FastAPI(
    title="ResMe API",
    version="1.0.0"
)

app.middleware("http")(request_id_middleware)
app.include_router(resume_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
