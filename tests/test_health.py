from fastapi.testclient import TestClient

from app import __version__
from app.main import create_app


def test_health_returns_200_with_service_info():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ai-evaluation",
        "version": __version__,
    }
