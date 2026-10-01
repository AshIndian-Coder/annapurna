from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import get_settings
from app.database import SessionLocal


settings = get_settings()

app = FastAPI(
    title="Annapurna API",
    description="AI-Powered Smart Food Waste Reduction & Sustainable Redistribution Ecosystem",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/db")
def database_health_check() -> dict[str, str]:
    db = SessionLocal()

    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    finally:
        db.close()


@app.get("/api/v1")
def api_root() -> dict[str, str]:
    return {
        "name": "Annapurna API",
        "version": "v1",
        "status": "ok",
    }
