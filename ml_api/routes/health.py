#!/usr/bin/env python3

"""
FastAPI Health Check
"""

import time
from datetime import datetime

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from ..app import get_db_manager, get_model_manager

router = APIRouter()  # Creates router instance


@router.get("/")  # /health/
async def health_check():
    return {"status": "healthy"}


@router.get("/db")  # /health/db
async def database_health(db_manager=Depends(get_db_manager)):
    """Checks to ensure database is functional and operating correctly"""

    try:
        health_data = check_database_health(db_manager)

        if health_data["status"] == "unhealthy":
            raise HTTPException(status_code=503, detail=health_data)
        return health_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


def check_database_health(db_manager):
    """Check database health"""

    start_time = time.time()
    health_data = {
        "status": "healthy",
        "connection": False,
        "tables_exist": [],
        "missing_tables": [],
        "timescaledb_enabled": False,
        "response_time_ms": 0,
        "details": "",
    }

    try:
        # Test basic connection
        health_data["connection"] = db_manager.test_connection()

        if not health_data["connection"]:
            health_data["status"] = "unhealthy"
            health_data["details"] = "Database connection failed"
            return health_data

        # Check if tables exist
        expected_tables = ["backtest_results", "features", "market_data", "model_registry", "predictions"]

        with db_manager.engine.connect() as conn:
            for table in expected_tables:
                try:
                    conn.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
                    health_data["tables_exist"].append(table)
                except:
                    health_data["missing_tables"].append(table)

            # Check TimescaleDB
            try:
                result = conn.execute(
                    text(
                        """
                     SELECT extname 
                     FROM pg_extension 
                     WHERE extname = 'timescaledb'
                """
                    )
                )
                health_data["timescaledb_enabled"] = result.fetchone() is not None
            except:
                health_data["timescaledb_enabled"] = False

    except Exception as e:
        health_data["status"] = "unhealthy"
        health_data["details"] = str(e)

    # Calculate response time
    health_data["response_time_ms"] = round((time.time() - start_time) * 1000, 2)

    # Determine status
    if health_data["missing_tables"] or not health_data["timescaledb_enabled"]:
        health_data["status"] = "degraded"

    return health_data
