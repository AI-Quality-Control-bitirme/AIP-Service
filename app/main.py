"""FastAPI application for the AI Evaluation Service."""

from fastapi import FastAPI

from app import __version__
from app.api import health


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Evaluation Service",
        description="Precheck, specialist agents, validation and scoring for recorded hotel reservation calls.",
        version=__version__,
    )
    app.include_router(health.router)
    return app


app = create_app()
