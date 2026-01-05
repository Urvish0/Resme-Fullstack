from fastapi import FastAPI
from .core.logging import setup_logging
from .core.middleware import request_id_middleware
from .api.routes.resume import router as resume_router
from .api.routes.health import router as health_router

setup_logging()


app = FastAPI(
    title="ResMe API",
    version="1.0.0"
)

app.middleware("http")(request_id_middleware)
app.include_router(resume_router)
app.include_router(health_router)

