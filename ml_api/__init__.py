"""
Trading Research API
Simple FastAPI application for model serving and portfolio management
"""

from .app import create_app
from .models import PredictionRequest, PredictionResponse, PortfolioRequest
from .routes import predictions, portfolio, health

__all__ = ['create_app', 'PredictionRequest', 'PredictionResponse', 'PortfolioRequest']