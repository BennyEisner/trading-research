#!/usr/bin/env python3

"""
Database Connection Management
Simple PostgreSQL connection handling with connection pooling
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from config.config import get_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Simple database connection manager"""
    
    def __init__(self, database_url: str = None):
        config = get_config()
        
        self.database_url = database_url or config.get_database_url()
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            echo=config.database.echo
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database manager initialized: {self.database_url}")
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Context manager for database sessions with automatic cleanup"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def setup_timescale(self) -> None:
        """Setup TimescaleDB extension (run once)"""
        try:
            with self.session_scope() as session:
                # Enable TimescaleDB extension
                session.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                logger.info("TimescaleDB extension enabled")
        except Exception as e:
            logger.error(f"Failed to setup TimescaleDB: {e}")
            raise
    
    def close(self) -> None:
        """Close database connections"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager
_db_manager = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_session() -> Session:
    """Get a database session (convenience function)"""
    return get_database_manager().get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Database session context manager (convenience function)"""
    with get_database_manager().session_scope() as session:
        yield session