#!/usr/bin/env python3

"""
Unit tests for Database
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from database.connection import DatabaseManager
from ml_api.routes.health import check_database_health, router


class TestDatabaseManager:
    """Test cases for DatabaseManager"""

    def test_connection_success(self):
        """Test that test_connection returns True when database query is successful"""

        # ARRANGE mock all the dependencies
        with patch("database.connection.get_config") as mock_get_config:
            with patch("database.connection.create_engine") as mock_create_engine:
                with patch("database.connection.sessionmaker") as mock_sessionmaker:

                    # Mock config object
                    mock_config_obj = Mock()
                    mock_config_obj.get_database_url.return_value = "sqlite:///test.db"
                    mock_config_obj.database.pool_size = 5
                    mock_config_obj.database.max_overflow = 10
                    mock_config_obj.database.echo = False
                    mock_get_config.return_value = mock_config_obj

                    # Mock session and engine
                    mock_session = Mock()
                    mock_sessionmaker.return_value = Mock(return_value=mock_session)

                    # Create Database Manager
                    db_manager = DatabaseManager("sqlite:///test.db")

                    # Mock session_scope
                    with patch.object(db_manager, "session_scope") as mock_session_scope:
                        mock_session_scope.return_value.__enter__.return_value = mock_session
                        mock_session_scope.return_value.__exit__.return_value = None

                        # ACT
                        result = db_manager.test_connection()

                        # ASSERT
                        assert result is True
                        mock_session.execute.assert_called_once()

    def test_connection_failure(self):
        """Test that the test_connection returns False if database query is not successful"""

        # ARRANGE mock all the dependencies
        with patch("database.connection.get_config") as mock_get_config:
            with patch("database.connection.create_engine") as mock_create_engine:
                with patch("database.connection.sessionmaker") as mock_sessionmaker:

                    # Mock config object
                    mock_config_obj = Mock()
                    mock_config_obj.get_database_url.return_value = "sqlite:///test.db"
                    mock_config_obj.database.pool_size = 5
                    mock_config_obj.database.max_overflow = 10
                    mock_config_obj.database.echo = False
                    mock_get_config.return_value = mock_config_obj

                    # Create Database Manager
                    db_manager = DatabaseManager("sqlite:///test.db")

                    # Mock session_scope to raise exception
                    with patch.object(db_manager, "session_scope") as mock_session_scope:
                        mock_session_scope.side_effect = Exception("Database connection failed")

                        # ACT
                        result = db_manager.test_connection()

                        # ASSERT
                        assert result is False


class TestDatabaseHealthCheck:
    """Test cases for database health check functionality - integration level"""

    def test_check_database_health_basic_structure(self):
        """Test that check_database_health returns proper structure"""
        # ARRANGE
        mock_db_manager = Mock()
        mock_db_manager.test_connection.return_value = False  # Simple case

        # ACT
        result = check_database_health(mock_db_manager)

        # ASSERT - Test the response structure, not complex integration behavior
        assert "status" in result
        assert "connection" in result
        assert "tables_exist" in result
        assert "missing_tables" in result
        assert "timescaledb_enabled" in result
        assert "response_time_ms" in result
        assert result["status"] == "unhealthy"  # Should be unhealthy when connection fails


class TestDatabaseBasicSetup:
    """Test basic database setup works"""

    def test_database_manager_initialization(self):
        """Test that DatabaseManager can be created without errors"""
        # ARRANGE mock all dependencies
        with patch("database.connection.get_config") as mock_get_config:
            with patch("database.connection.create_engine") as mock_create_engine:
                with patch("database.connection.sessionmaker") as mock_sessionmaker:

                    # Mock config object
                    mock_config_obj = Mock()
                    mock_config_obj.get_database_url.return_value = "sqlite:///test.db"
                    mock_config_obj.database.pool_size = 5
                    mock_config_obj.database.max_overflow = 10
                    mock_config_obj.database.echo = False
                    mock_get_config.return_value = mock_config_obj

                    # ACT create Database Manager
                    db_manager = DatabaseManager("sqlite:///test.db")

                    # ASSERT
                    assert db_manager is not None
                    assert db_manager.database_url == "sqlite:///test.db"
                    mock_create_engine.assert_called_once()
                    mock_sessionmaker.assert_called_once()
