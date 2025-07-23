#!/usr/bin/env python3

"""
Modular Feature Engineering Package

This package provides a comprehensive, modular feature engineering system for financial time series data.

Main Components:
- FeatureEngineer: Main orchestrator class
- BaseFeatureProcessor: Base class for feature processors  
- Feature Processors: Specialized processors for different feature types
- Feature Selectors: Advanced feature selection methods
- Utilities: Data validation, scaling, and LSTM preparation

Example Usage:
    from src.features import FeatureEngineer
    
    # Initialize
    fe = FeatureEngineer(symbol="AAPL")
    
    # Calculate all features  
    features_data = fe.calculate_all_features(raw_data)
    
    # Select best features
    best_features = fe.select_best_features(features_data, method="ensemble")
    
    # Prepare for LSTM
    X, y, feature_names = fe.prepare_for_lstm(features_data, best_features)
"""

from .feature_engineer import FeatureEngineer
from .base import BaseFeatureProcessor, FEATURE_REGISTRY

# Import processors
from .processors import (
    PriceFeaturesProcessor,
    TechnicalIndicatorsProcessor,
    VolumeFeaturesProcessor, 
    VolatilityFeaturesProcessor,
    MomentumFeaturesProcessor,
    MarketFeaturesProcessor
)

# Import simple selector (replaced complex selectors)
from .simple_selector import SimpleCategorySelector

# Import utilities
from .utils import (
    DataValidator,
    FeatureScaler,
    prepare_lstm_sequences,
    clean_financial_data
)

__version__ = "1.0.0"

__all__ = [
    # Main classes
    'FeatureEngineer',
    'BaseFeatureProcessor',
    'FEATURE_REGISTRY',
    
    # Processors
    'PriceFeaturesProcessor',
    'TechnicalIndicatorsProcessor',
    'VolumeFeaturesProcessor',
    'VolatilityFeaturesProcessor', 
    'MomentumFeaturesProcessor',
    'MarketFeaturesProcessor',
    
    # Selector
    'SimpleCategorySelector',
    
    # Utilities
    'DataValidator',
    'FeatureScaler',
    'prepare_lstm_sequences',
    'clean_financial_data'
]