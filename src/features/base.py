#!/usr/bin/env python3
"""
Base classes and utilities for feature engineering
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class BaseFeatureProcessor(ABC):
    """
    Abstract base class for feature processors
    """

    def __init__(self, name: str):
        self.name = name
        self.feature_names = []
        self.dependencies = []

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features and add to DataFrame

        Args:
            data: Input DataFrame with OHLCV data

        Returns:
            DataFrame with added features
        """
        pass

    def get_feature_names(self) -> List[str]:
        """Return list of features this processor creates"""
        return self.feature_names

    def get_dependencies(self) -> List[str]:
        """Return list of features this processor depends on"""
        return self.dependencies

    def validate_dependencies(self, data: pd.DataFrame) -> bool:
        """Check if all required dependencies exist in data"""
        missing = [dep for dep in self.dependencies if dep not in data.columns]
        if missing:
            raise ValueError(f"{self.name} missing dependencies: {missing}")
        return True


class FeatureGroup:
    """
    Container for related features with metadata
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.features = []
        self.processor = None

    def add_features(self, features: List[str]):
        """Add features to this group"""
        self.features.extend(features)

    def set_processor(self, processor: BaseFeatureProcessor):
        """Set the processor that creates these features"""
        self.processor = processor


class FeatureRegistry:
    """
    Registry to track all available features and their groups
    """

    def __init__(self):
        self.groups = {}
        self.feature_to_group = {}
        self.processors = {}

    def register_group(self, group: FeatureGroup):
        """Register a feature group"""
        self.groups[group.name] = group
        for feature in group.features:
            self.feature_to_group[feature] = group.name

    def register_processor(self, processor: BaseFeatureProcessor):
        """Register a feature processor"""
        self.processors[processor.name] = processor

    def get_group_features(self, group_name: str) -> List[str]:
        """Get all features in a group"""
        return self.groups.get(group_name, FeatureGroup("")).features

    def get_feature_group(self, feature_name: str) -> str:
        """Get the group name for a feature"""
        return self.feature_to_group.get(feature_name, "unknown")

    def get_all_features(self) -> List[str]:
        """Get all registered features"""
        return list(self.feature_to_group.keys())


# Global feature registry
FEATURE_REGISTRY = FeatureRegistry()


def safe_divide(
    numerator: Union[pd.Series, np.ndarray, float], denominator: Union[pd.Series, np.ndarray, float], default: float = 0.0
) -> Union[pd.Series, np.ndarray, float]:
    """
    Safe division with default value for zero division

    Args:
        numerator: Numerator values
        denominator: Denominator values
        default: Default value when denominator is zero

    Returns:
        Division result with default for zero division
    """
    if isinstance(denominator, (pd.Series, np.ndarray)):
        return numerator / (denominator + 1e-10)
    else:
        return numerator / (denominator + 1e-10) if abs(denominator) > 1e-10 else default


def rolling_correlation(series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling correlation between two series

    Args:
        series1: First time series
        series2: Second time series
        window: Rolling window size

    Returns:
        Rolling correlation series
    """
    return series1.rolling(window).corr(series2)


def normalize_feature(series: pd.Series, method: str = "zscore") -> pd.Series:
    """
    Normalize a feature series

    Args:
        series: Input series
        method: Normalization method ('zscore', 'minmax', 'robust')

    Returns:
        Normalized series
    """
    if method == "zscore":
        return (series - series.mean()) / (series.std() + 1e-10)
    elif method == "minmax":
        return (series - series.min()) / (series.max() - series.min() + 1e-10)
    elif method == "robust":
        median = series.median()
        mad = (series - median).abs().median()
        return (series - median) / (mad + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_lagged_features(data: pd.DataFrame, features: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lagged versions of features

    Args:
        data: Input DataFrame
        features: List of feature names to lag
        lags: List of lag periods

    Returns:
        DataFrame with original data plus lagged features
    """
    result = data.copy()

    for feature in features:
        if feature in data.columns:
            for lag in lags:
                result[f"{feature}_lag{lag}"] = data[feature].shift(lag)

    return result


def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains required OHLCV columns

    Args:
        data: Input DataFrame

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise ValueError(f"Missing required OHLCV columns: {missing_cols}")

    if (data["high"] < data["low"]).any():
        raise ValueError("Invalid OHLC data: high < low detected")

    if (data["high"] < data["open"]).any() or (data["high"] < data["close"]).any():
        raise ValueError("Invalid OHLC data: high < open/close detected")

    if (data["low"] > data["open"]).any() or (data["low"] > data["close"]).any():
        raise ValueError("Invalid OHLC data: low > open/close detected")

    if (data["volume"] < 0).any():
        raise ValueError("Invalid volume data: negative values detected")

    return True


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    Calculate returns from price series

    Args:
        prices: Price series
        method: Return calculation method ('simple', 'log')

    Returns:
        Returns series
    """
    if method == "simple":
        return prices.pct_change()
    elif method == "log":
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown return method: {method}")

