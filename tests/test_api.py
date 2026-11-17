
import pytest
from fastapi.testclient import TestClient

from api.main import app


def test_api_root_endpoint():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "BookMind"
    assert data["status"] == "running"
    assert "docs" in data


def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_limit_validation_out_of_bounds():
    client = TestClient(app)

    low_response = client.get("/recommendations/popular?limit=0")
    assert low_response.status_code == 400

    high_response = client.get("/recommendations/popular?limit=101")
    assert high_response.status_code == 400
