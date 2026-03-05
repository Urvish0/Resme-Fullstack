from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .core.logging import setup_logging
from .core.middleware import request_id_middleware
from .core.limiter import limiter
from .api.routes.resume import router as resume_router
from .api.routes.health import router as health_router
from .api.routes.metrics import router as metrics_router

setup_logging()

app = FastAPI(title="ResMe API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://resme-lake.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing middleware
app.middleware("http")(request_id_middleware)

# Routers
app.include_router(resume_router)
app.include_router(health_router)
app.include_router(metrics_router)
