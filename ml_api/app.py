#!/usr/bin/env python3

"""
FastAPI Application
Simple API for trading research model serving
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.config import get_config
from database.connection import get_database_manager

from .routes import health, portfolio, predictions

logger = logging.getLogger(__name__)

# Global application state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    
    # Startup
    logger.info("Starting Trading Research API...")
    
    config = get_config()
    
    # Initialize database
    db_manager = get_database_manager()
    if not db_manager.test_connection():
        raise RuntimeError("Database connection failed")
    
    # Load models
    from .model_manager import ModelManager
    model_manager = ModelManager()
    await model_manager.load_models()
    
    app_state['db_manager'] = db_manager
    app_state['model_manager'] = model_manager
    app_state['config'] = config
    app_state['startup_time'] = datetime.now()
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    if 'db_manager' in app_state:
        app_state['db_manager'].close()
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    config = get_config()
    
    app = FastAPI(
        title="Trading Research API",
        description="Personal trading research and model serving API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware for development
    if config.environment == "development":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
    app.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Trading Research API",
            "version": "1.0.0",
            "environment": config.environment,
            "startup_time": app_state.get('startup_time', '').isoformat() if app_state.get('startup_time') else None
        }
    
    @app.get("/info")
    async def info():
        """Application information"""
        config = app_state.get('config')
        model_manager = app_state.get('model_manager')
        
        return {
            "environment": config.environment if config else "unknown",
            "models_loaded": len(model_manager.models) if model_manager else 0,
            "database_connected": app_state.get('db_manager').test_connection() if app_state.get('db_manager') else False,
            "uptime_seconds": (datetime.now() - app_state.get('startup_time')).total_seconds() if app_state.get('startup_time') else 0
        }
    
    return app



# Dependency injection
def get_app_state() -> Dict[str, Any]:
    """Get application state (for dependency injection)"""
    return app_state


def get_model_manager():
    """Get model manager dependency"""
    model_manager = app_state.get('model_manager')
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    return model_manager


def get_db_manager():
    """Get database manager dependency"""
    db_manager = app_state.get('db_manager')
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    return db_manager
