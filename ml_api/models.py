#!/usr/bin/env python3

"""
Pydantic Models for API Request/Response
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for stock predictions"""
    ticker: str = Field(..., description="Stock ticker symbol")
    horizons: List[int] = Field(default=[1, 3, 5], description="Prediction horizons in days")
    

class PredictionResponse(BaseModel):
    """Response model for stock predictions"""
    ticker: str
    predictions: Dict[int, float] = Field(..., description="Predictions by horizon")
    confidence: Dict[int, float] = Field(..., description="Confidence scores by horizon")
    model_version: str
    timestamp: str


class PortfolioRequest(BaseModel):
    """Request model for portfolio operations"""
    tickers: List[str] = Field(..., description="List of stock tickers")
    operation: str = Field(..., description="Portfolio operation type")
    

class PortfolioResponse(BaseModel):
    """Response model for portfolio operations"""
    status: str
    results: Dict[str, Any]
    timestamp: str