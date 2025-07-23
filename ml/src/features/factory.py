#!/usr/bin/env python3

"""
Factory for creating feature engineering pipeline
"""

from .base import BaseFeatureProcessor
from .processors.price_features import PriceFeaturesProcessor
from .processors.technical_indicators import TechnicalIndicatorsProcessor
from .processors.volatility_features import VolatilityFeaturesProcessor
from .processors.volume_features import VolumeFeaturesProcessor
from .processors.market_features import MarketFeaturesProcessor
from .selectors.ensemble_selector import EnsembleFeatureSelector


class FeatureEngineeringFactory:
    """Factory for creating complete feature engineering pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.processors = []
        self.selector = None
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup the feature engineering pipeline"""
        # Initialize processors in order
        self.processors = [
            PriceFeaturesProcessor(),
            TechnicalIndicatorsProcessor(),
            VolatilityFeaturesProcessor(),
            VolumeFeaturesProcessor(),
            MarketFeaturesProcessor()
        ]
        
        # Initialize feature selector
        self.selector = EnsembleFeatureSelector()
    
    def calculate_all_features(self, data):
        """Calculate all features using the pipeline"""
        # Import the main feature engineering class
        from .feature_engineering import FeatureEngineer
        
        # Create instance and calculate features
        feature_eng = FeatureEngineer()
        processed_data = feature_eng.calculate_all_features(data)
        
        return processed_data
    
    def select_stable_features(self, data):
        """Select stable features using ensemble selector"""
        # Get all feature columns (exclude basic OHLCV)
        basic_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data.columns if col not in basic_columns]
        
        if not feature_columns:
            return basic_columns[1:]  # Return OHLCV without date
        
        # Use ensemble selector if we have features
        try:
            selected_features = self.selector.select_features(data, feature_columns)
            return basic_columns[1:] + selected_features  # Include OHLCV + selected
        except Exception as e:
            print(f"Warning: Feature selection failed ({e}), using all features")
            return feature_columns
    
    def get_feature_dimensions(self, data):
        """Get number of features after processing and selection"""
        processed_data = self.calculate_all_features(data)
        selected_columns = self.select_stable_features(processed_data)
        return len(selected_columns)
    
    def validate_pipeline(self, data):
        """Validate the feature engineering pipeline"""
        issues = []
        
        try:
            # Test feature calculation
            processed_data = self.calculate_all_features(data)
            if processed_data is None or len(processed_data) == 0:
                issues.append("Feature calculation returned empty data")
            
            # Test feature selection
            selected_features = self.select_stable_features(processed_data)
            if not selected_features:
                issues.append("Feature selection returned no features")
            
            # Check for NaN values in selected features
            feature_data = processed_data[selected_features]
            if feature_data.isna().sum().sum() > 0:
                issues.append("Selected features contain NaN values")
                
        except Exception as e:
            issues.append(f"Pipeline validation error: {e}")
        
        return len(issues) == 0, issues


def create_feature_engineer(config=None):
    """Convenience function to create feature engineering factory"""
    return FeatureEngineeringFactory(config)