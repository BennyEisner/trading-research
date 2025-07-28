#!/usr/bin/env python3

"""
Strategy Compatibility Checker for requirement validation
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from ..base import BaseStrategy


class FeatureValidationError(Exception):
    """Raised when feature validation fails for strategy execution"""
    pass


class StrategyCompatibilityChecker:
    """
    Focused on validating strategy data requirements and compatibility
    
    Single responsibility: Check if data meets strategy requirements
    """
    
    def __init__(self, min_data_points: int = 50, max_nan_ratio: float = 0.1):
        """
        Initialize compatibility checker
        
        Args:
            min_data_points: Minimum data points required
            max_nan_ratio: Maximum allowed NaN ratio (0.0-1.0)
        """
        self.min_data_points = min_data_points
        self.max_nan_ratio = max_nan_ratio
        
    def check_strategy_compatibility(self, data: pd.DataFrame, strategy: BaseStrategy) -> Tuple[bool, List[str]]:
        """
        Check if data is compatible with strategy requirements
        
        Args:
            data: Input dataframe
            strategy: Strategy to check compatibility for
            
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        issues = []
        
        # Check data size requirements
        if len(data) < self.min_data_points:
            issues.append(f"Insufficient data: {len(data)} < {self.min_data_points} required")
            
        # Check required features
        required_features = strategy.get_required_features()
        missing_features = []
        
        for feature in required_features:
            if feature not in data.columns:
                missing_features.append(feature)
                
        if missing_features:
            # Check for fallbacks
            critical_missing = [f for f in missing_features if not self._has_fallback(f, data.columns)]
            if critical_missing:
                issues.append(f"Critical features missing: {critical_missing}")
                
        # Check feature quality
        for feature in required_features:
            if feature in data.columns:
                feature_issues = self._validate_feature_quality(data, feature)
                issues.extend(feature_issues)
                
        # Check strategy parameters
        try:
            if not strategy.validate_parameters():
                issues.append("Strategy parameter validation failed")
        except Exception as e:
            issues.append(f"Strategy validation error: {str(e)}")
            
        return len(issues) == 0, issues
        
    def _has_fallback(self, feature: str, available_columns: List[str]) -> bool:
        """Check if missing feature has fallback available"""
        fallback_mappings = {
            "rsi_14": ["rsi"],
            "rsi": ["rsi_14"],
            "close": [],
            "atr": [],
            "macd": [],
            "macd_signal": [],
            "macd_histogram": []
        }
        
        fallbacks = fallback_mappings.get(feature, [])
        return any(fallback in available_columns for fallback in fallbacks)
        
    def _validate_feature_quality(self, data: pd.DataFrame, feature: str) -> List[str]:
        """Validate quality of individual feature"""
        issues = []
        
        if feature not in data.columns:
            return issues
            
        feature_data = data[feature]
        
        # Check NaN ratio
        nan_ratio = feature_data.isna().sum() / len(feature_data)
        if nan_ratio > self.max_nan_ratio:
            issues.append(f"High NaN ratio in {feature}: {nan_ratio:.2%}")
            
        # Check for infinite values
        if feature_data.dtype in ['float64', 'float32']:
            if np.isinf(feature_data).any():
                issues.append(f"Infinite values in {feature}")
                
        # Feature-specific validation
        if feature.startswith('rsi'):
            invalid_rsi = (feature_data < 0) | (feature_data > 100)
            if invalid_rsi.any():
                issues.append(f"Invalid RSI values in {feature} (outside 0-100 range)")
                
        elif feature == 'atr':
            if (feature_data < 0).any():
                issues.append(f"Negative ATR values in {feature}")
                
        return issues