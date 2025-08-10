#!/usr/bin/env python3
"""
Unit tests for Health Endpoints
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ml_api.routes.health import check_database_health, router


class TestHealthEndpoints:
    """Test cases for health endpoints"""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing"""
        mock_manager = Mock()
        mock_manager.get_session.return_value.__enter__ = Mock()
        mock_manager.get_session.return_value.__exit__ = Mock()
        return mock_manager

    @pytest.fixture
    def test_client(self):
        """Create test client for FastAPI"""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router, prefix="/health")
        return TestClient(app)

    def test_basic_health_endpoint(self, test_client):
        """Test basic health check endpoint"""
        response = test_client.get("/health/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @patch("ml_api.routes.health.check_database_health")
    @patch("ml_api.routes.health.get_database_manager")
    def test_database_health_success(self, mock_get_db, mock_check_db, test_client):
        """Test successful database health check"""
        # Setup mocks
        mock_db_manager = Mock()
        mock_get_db.return_value = mock_db_manager
        mock_check_db.return_value = {
            "status": "healthy",
            "database": "connected",
            "tables_exist": True,
            "timescaledb_enabled": True,
        }

        response = test_client.get("/health/db")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        mock_check_db.assert_called_once_with(mock_db_manager)

    @patch("ml_api.routes.health.check_database_health")
    @patch("ml_api.routes.health.get_database_manager")
    def test_database_health_failure(self, mock_get_db, mock_check_db, test_client):
        """Test failed database health check"""
        # Setup mocks for failure
        mock_db_manager = Mock()
        mock_get_db.return_value = mock_db_manager
        mock_check_db.return_value = {
            "status": "unhealthy",
            "database": "connection_failed",
            "error": "Database connection timeout",
        }

        response = test_client.get("/health/db")

        assert response.status_code == 503
        data = response.json()
        assert "Database connection timeout" in str(data)

    def test_info_endpoint_structure(self, test_client):
        """Test info endpoint returns proper structure"""
        response = test_client.get("/health/info")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "service" in data
        assert "version" in data
        assert "environment" in data
        assert "model_status" in data
        assert "features" in data
        assert "timestamp" in data

    def test_info_endpoint_model_status(self, test_client):
        """Test that info endpoint includes model status"""
        response = test_client.get("/health/info")

        assert response.status_code == 200
        data = response.json()

        model_status = data["model_status"]
        assert "available_models" in model_status
        assert "loaded_models" in model_status
        assert isinstance(model_status["available_models"], int)
        assert isinstance(model_status["loaded_models"], int)


class TestCheckDatabaseHealth:
    """Test cases for database health check function"""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager"""
        mock_manager = Mock()
        mock_session = Mock()
        mock_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_manager.get_session.return_value.__exit__.return_value = None
        return mock_manager, mock_session

    def test_database_health_success(self, mock_db_manager):
        """Test successful database health check"""
        db_manager, mock_session = mock_db_manager

        # Mock successful queries
        mock_session.execute.return_value.scalar.side_effect = [1, True, True, True]

        result = check_database_health(db_manager)

        assert result["status"] == "healthy"
        assert result["database"] == "connected"
        assert result["tables_exist"] is True
        assert result["timescaledb_enabled"] is True

    def test_database_health_connection_failure(self, mock_db_manager):
        """Test database connection failure"""
        db_manager, mock_session = mock_db_manager

        # Mock connection failure
        db_manager.get_session.side_effect = Exception("Connection failed")

        result = check_database_health(db_manager)

        assert result["status"] == "unhealthy"
        assert result["database"] == "connection_failed"
        assert "Connection failed" in result["error"]

    def test_database_health_missing_tables(self, mock_db_manager):
        """Test missing database tables"""
        db_manager, mock_session = mock_db_manager

        # Mock successful connection but missing tables
        mock_session.execute.return_value.scalar.side_effect = [1, False, True, True]

        result = check_database_health(db_manager)

        assert result["status"] == "unhealthy"
        assert result["tables_exist"] is False
        assert "Required tables are missing" in result["error"]

    def test_database_health_timescaledb_disabled(self, mock_db_manager):
        """Test TimescaleDB extension disabled"""
        db_manager, mock_session = mock_db_manager

        # Mock TimescaleDB not available
        mock_session.execute.return_value.scalar.side_effect = [1, True, False, True]

        result = check_database_health(db_manager)

        assert result["status"] == "unhealthy"
        assert result["timescaledb_enabled"] is False
        assert "TimescaleDB extension not available" in result["error"]

    def test_database_health_query_failure(self, mock_db_manager):
        """Test database query failure"""
        db_manager, mock_session = mock_db_manager

        # Mock query failure
        mock_session.execute.side_effect = Exception("Query failed")

        result = check_database_health(db_manager)

        assert result["status"] == "unhealthy"
        assert result["database"] == "query_failed"
        assert "Query failed" in result["error"]


@pytest.mark.integration
class TestHealthEndpointsIntegration:
    """Integration tests for health endpoints"""

    @pytest.fixture
    def app_client(self):
        """Create test client with full app"""
        from ml_api.app import create_app

        app = create_app()
        return TestClient(app)

    def test_health_endpoints_integration(self, app_client):
        """Test health endpoints with full app context"""
        # Test basic health
        response = app_client.get("/health/")
        assert response.status_code == 200

        # Test info endpoint
        response = app_client.get("/health/info")
        assert response.status_code == 200

        # Note: Database health test would require actual database connection
        # This could be added as a separate integration test with test database

