from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import get_settings
from app.database import SessionLocal
from app.routes.auth import router as auth_router
from app.routes.kitchens import router as kitchens_router
from app.utils.auth import get_current_user


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

app.include_router(
    auth_router,
    prefix="/api/v1",
)

app.include_router(
    kitchens_router,
    prefix="/api/v1",
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


@app.get("/api/v1/auth/me")
def get_me(
    current_user: dict = Depends(get_current_user),
) -> dict:
    return current_user
