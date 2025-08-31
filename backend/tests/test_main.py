"""
Basic tests for the main application
"""
import os
import pytest
from fastapi.testclient import TestClient

# Set test environment variables before importing app
os.environ["SECRET_KEY"] = "test_secret_key_for_testing_only_32_chars"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["JWT_SECRET_KEY"] = "test_jwt_secret_key_for_testing_only_32_chars"

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_api_docs():
    """Test API documentation is accessible"""
    response = client.get("/api/v1/docs")
    assert response.status_code == 200


def test_openapi_json():
    """Test OpenAPI JSON is accessible"""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data