from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router


def create_app() -> FastAPI:
    """
    Application factory (useful for tests and future config).
    """
    app = FastAPI(title="production-rag-agent", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()

