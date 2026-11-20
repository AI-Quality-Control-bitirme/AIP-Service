"""Liveness check used by deployment tooling and the other services."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__

SERVICE_NAME = "ai-evaluation"

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the process is up. It does not check external dependencies."""
    return HealthResponse(status="ok", service=SERVICE_NAME, version=__version__)
