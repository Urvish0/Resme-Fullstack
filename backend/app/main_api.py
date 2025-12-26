from fastapi import FastAPI
from .api.routes.resume import router as resume_router

app = FastAPI(
    title="ResMe API",
    version="1.0.0"
)

app.include_router(resume_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
