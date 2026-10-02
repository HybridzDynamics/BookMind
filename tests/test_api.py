"""
Tests for FastAPI Backend Service.

Tests cover:
- Root info endpoint GET /.
- Health check endpoint GET /health.
- Limit query parameter boundary validation.

Note: Strictly functional test implementations using only `def` (no `class`).
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


def test_api_root_endpoint():
    """
    Test GET / returns service name, running status, and docs link.
    """
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "BookMind"
    assert data["status"] == "running"
    assert "docs" in data


def test_api_health_endpoint():
    """
    Test GET /health returns healthy status.
    """
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_limit_validation_out_of_bounds():
    """
    Test that limit values below 1 or above 100 return 400 Bad Request.
    """
    client = TestClient(app)

    low_response = client.get("/recommendations/popular?limit=0")
    assert low_response.status_code == 400

    high_response = client.get("/recommendations/popular?limit=101")
    assert high_response.status_code == 400
