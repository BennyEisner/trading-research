#!/usr/bin/env python3

"""
Pattern Detection Feature Engineering Package

This package provides pattern-focused feature engineering for financial time series data.
Aligned with 17 pattern features for LSTM pattern detection specialist approach.

Main Components:
- PatternFeatureCalculator: 17 pattern features calculation
- MultiTickerPatternEngine: Multi-ticker pattern processing
- BaseFeatureProcessor: Base class for feature processors  
- Pattern-focused processors only
- Pattern validation utilities

Example Usage:
    from src.features import PatternFeatureCalculator, MultiTickerPatternEngine
    
    # Calculate 17 pattern features
    calculator = PatternFeatureCalculator(symbol="AAPL")
    pattern_features = calculator.calculate_all_features(raw_data)
    
    # Multi-ticker processing
    engine = MultiTickerPatternEngine(tickers=["AAPL", "MSFT"])
    portfolio_features = engine.calculate_portfolio_features(ticker_data)
"""

# Pattern detection focused imports
from .base import BaseFeatureProcessor, FEATURE_REGISTRY
from .pattern_feature_calculator import FeatureCalculator as PatternFeatureCalculator
from .multi_ticker_engine import MultiTickerPatternEngine

# Import only pattern-focused processor
from .processors.pattern_features_processor import PatternFeaturesProcessor

# Import pattern validation utilities
from .utils import (
    DataValidator,
    prepare_lstm_sequences,
    clean_financial_data
)
from .utils.pattern_feature_validator import PatternFeatureValidator
from .utils.multi_ticker_validator import MultiTickerValidator

__version__ = "1.0.0"

__all__ = [
    # Pattern detection core classes
    'PatternFeatureCalculator',
    'MultiTickerPatternEngine', 
    'BaseFeatureProcessor',
    'FEATURE_REGISTRY',
    
    # Pattern-focused processor
    'PatternFeaturesProcessor',
    
    # Pattern validation utilities
    'PatternFeatureValidator',
    'MultiTickerValidator',
    'DataValidator',
    'prepare_lstm_sequences',
    'clean_financial_data'
]