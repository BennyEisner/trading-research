#!/usr/bin/env python3

"""
Pattern Detection Feature Factory
Focused on 17 pattern features for LSTM pattern detection specialist
"""

from .base import BaseFeatureProcessor
from .processors.pattern_features_processor import PatternFeaturesProcessor
from .pattern_feature_calculator import FeatureCalculator
from .multi_ticker_engine import MultiTickerPatternEngine


class PatternDetectionFactory:
    """Factory for creating pattern detection feature pipeline (17 features only)"""

    def __init__(self, config=None, market_data=None, sector_data=None, vix_data=None):
        self.config = config or {}
        self.market_data = market_data
        self.sector_data = sector_data or {}
        self.vix_data = vix_data
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Setup the pattern detection pipeline"""
        # Only use pattern features processor (17 features)
        self.pattern_processor = PatternFeaturesProcessor(
            market_data=self.market_data,
            sector_data=self.sector_data,
            vix_data=self.vix_data
        )

    def calculate_pattern_features(self, data, symbol="UNKNOWN"):
        """Calculate 17 pattern features only"""
        # Use pattern processor to calculate all 17 pattern features
        pattern_features = self.pattern_processor.calculate(data)
        return pattern_features

    def create_pattern_calculator(self, symbol="UNKNOWN"):
        """Create pattern feature calculator for single ticker"""
        return FeatureCalculator(
            symbol=symbol,
            market_data=self.market_data,
            sector_data=self.sector_data.get(symbol),
            vix_data=self.vix_data
        )

    def create_multi_ticker_engine(self, tickers, max_workers=4):
        """Create multi-ticker pattern engine"""
        return MultiTickerPatternEngine(
            tickers=tickers,
            max_workers=max_workers,
            market_data=self.market_data,
            sector_data=self.sector_data,
            vix_data=self.vix_data
        )

    def get_pattern_feature_names(self):
        """Get list of 17 pattern feature names"""
        return self.pattern_processor.feature_names

    def validate_pattern_pipeline(self, data, symbol="UNKNOWN"):
        """Validate the pattern detection pipeline"""
        issues = []

        try:
            # Test pattern feature calculation
            pattern_features = self.calculate_pattern_features(data, symbol)
            if pattern_features is None or len(pattern_features) == 0:
                issues.append("Pattern feature calculation returned empty data")

            # Check that we have exactly 17 pattern features + context
            expected_features = len(self.pattern_processor.feature_names)
            actual_features = len([col for col in pattern_features.columns 
                                if col in self.pattern_processor.feature_names])
            
            if actual_features != expected_features:
                issues.append(f"Expected {expected_features} pattern features, got {actual_features}")

            # Check for excessive NaN values
            pattern_feature_data = pattern_features[self.pattern_processor.feature_names]
            nan_percentage = pattern_feature_data.isna().sum().sum() / (len(pattern_feature_data) * len(self.pattern_processor.feature_names))
            
            if nan_percentage > 0.1:  # More than 10% NaN
                issues.append(f"Pattern features contain {nan_percentage:.1%} NaN values")

        except Exception as e:
            issues.append(f"Pattern pipeline validation error: {e}")

        return len(issues) == 0, issues


def create_pattern_detection_factory(config=None, market_data=None, sector_data=None, vix_data=None):
    """Convenience function to create pattern detection factory"""
    return PatternDetectionFactory(
        config=config,
        market_data=market_data,
        sector_data=sector_data,
        vix_data=vix_data
    )

