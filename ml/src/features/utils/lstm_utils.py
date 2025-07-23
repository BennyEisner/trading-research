#!/usr/bin/env python3

"""
LSTM-specific utilities for feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple, Optional, Dict


def prepare_lstm_sequences(data: pd.DataFrame,
                          feature_columns: List[str],
                          target_column: str,
                          sequence_length: int = 60,
                          prediction_horizon: int = 1,
                          stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training with advanced options
    
    Args:
        data: DataFrame with features and target
        feature_columns: List of feature column names
        target_column: Target column name
        sequence_length: Length of input sequences
        prediction_horizon: Steps ahead to predict (1 = next period)
        stride: Step size between sequences (1 = every period)
        
    Returns:
        Tuple of (X_sequences, y_targets)
    """
    # Validate inputs
    if len(data) < sequence_length + prediction_horizon:
        raise ValueError(f"Not enough data. Need at least {sequence_length + prediction_horizon} samples")
    
    # Get available features
    available_features = [col for col in feature_columns if col in data.columns]
    if not available_features:
        raise ValueError("No specified features found in data")
        
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Prepare feature and target data
    feature_data = data[available_features].fillna(0).values
    target_data = data[target_column].fillna(0).values
    
    # Create sequences
    X, y = [], []
    
    for i in range(0, len(data) - sequence_length - prediction_horizon + 1, stride):
        # Input sequence
        X_seq = feature_data[i:i + sequence_length]
        
        # Target (prediction_horizon steps ahead)
        y_target = target_data[i + sequence_length + prediction_horizon - 1]
        
        X.append(X_seq)
        y.append(y_target)
    
    return np.array(X), np.array(y)


def create_time_series_splits(data: pd.DataFrame,
                             n_splits: int = 5,
                             test_size_ratio: float = 0.2) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series cross-validation splits
    
    Args:
        data: DataFrame with time series data
        n_splits: Number of CV splits
        test_size_ratio: Ratio of data to use for testing in each split
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(data) * test_size_ratio))
    return list(tscv.split(data))


def create_lstm_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create LSTM-specific features that capture temporal patterns
    
    Args:
        data: Input DataFrame with basic features
        
    Returns:
        DataFrame with additional LSTM-optimized features
    """
    result_data = data.copy()
    
    # Sequence statistics
    if "daily_return" in data.columns:
        # Return sequence statistics
        result_data["return_sequence_sum_5"] = data["daily_return"].rolling(5).sum()
        result_data["return_sequence_sum_10"] = data["daily_return"].rolling(10).sum()
        result_data["return_sequence_std_5"] = data["daily_return"].rolling(5).std()
        result_data["return_sequence_std_10"] = data["daily_return"].rolling(10).std()
        
        # Return momentum features
        result_data["return_momentum_3"] = data["daily_return"].rolling(3).mean()
        result_data["return_momentum_5"] = data["daily_return"].rolling(5).mean()
        result_data["return_acceleration"] = result_data["return_momentum_3"] - result_data["return_momentum_5"]
    
    # Volatility persistence features
    if "volatility_5d" in data.columns:
        result_data["volatility_persistence"] = data["volatility_5d"].rolling(10).std()
        result_data["volatility_trend"] = data["volatility_5d"].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
    
    # Volume pattern features
    if "volume" in data.columns:
        result_data["volume_trend_5"] = data["volume"].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        result_data["volume_acceleration"] = result_data["volume_trend_5"].diff()
    
    # Price pattern features
    if "close" in data.columns:
        # Price momentum and acceleration
        result_data["price_momentum_3"] = data["close"].pct_change(3)
        result_data["price_momentum_7"] = data["close"].pct_change(7)
        result_data["price_acceleration"] = result_data["price_momentum_3"] - result_data["price_momentum_7"]
        
        # Support and resistance levels
        result_data["support_5"] = data["close"].rolling(5).min()
        result_data["resistance_5"] = data["close"].rolling(5).max()
        result_data["support_distance"] = (data["close"] - result_data["support_5"]) / data["close"]
        result_data["resistance_distance"] = (result_data["resistance_5"] - data["close"]) / data["close"]
    
    # Regime transition indicators
    for col in ["volatility_regime", "trend_regime", "volume_regime"]:
        if col in data.columns:
            result_data[f"{col}_change"] = (data[col] != data[col].shift(1)).astype(int)
            result_data[f"{col}_stability"] = data[col].rolling(5).std()
    
    # Memory features (temporal correlation patterns)
    memory_features = ["daily_return", "volume_ratio"]
    for feature in memory_features:
        if feature in data.columns:
            # Correlation with time (trend detection)
            def safe_time_corr(x):
                try:
                    if len(x) > 1 and x.std() > 0:
                        return np.corrcoef(x, range(len(x)))[0, 1]
                    else:
                        return 0
                except:
                    return 0
            
            result_data[f"{feature}_memory_5"] = data[feature].rolling(5).apply(safe_time_corr)
            result_data[f"{feature}_memory_10"] = data[feature].rolling(10).apply(safe_time_corr)
    
    return result_data


def validate_lstm_data(X: np.ndarray, 
                      y: np.ndarray,
                      feature_names: List[str]) -> Dict[str, any]:
    """
    Validate LSTM input data and provide diagnostics
    
    Args:
        X: Feature sequences array (samples, timesteps, features)
        y: Target array (samples,)
        feature_names: List of feature names
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'issues': [],
        'statistics': {}
    }
    
    # Check array shapes
    if len(X.shape) != 3:
        results['issues'].append(f"X should be 3D array, got shape {X.shape}")
        results['is_valid'] = False
        
    if len(y.shape) != 1:
        results['issues'].append(f"y should be 1D array, got shape {y.shape}")
        results['is_valid'] = False
        
    if X.shape[0] != y.shape[0]:
        results['issues'].append(f"X and y sample counts don't match: {X.shape[0]} vs {y.shape[0]}")
        results['is_valid'] = False
        
    if len(feature_names) != X.shape[2]:
        results['issues'].append(f"Feature names count doesn't match X features: {len(feature_names)} vs {X.shape[2]}")
        
    # Check for NaN/Inf values
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        results['issues'].append(f"Found {nan_count} NaN values in X")
        results['is_valid'] = False
        
    if np.isnan(y).any():
        nan_count = np.isnan(y).sum()
        results['issues'].append(f"Found {nan_count} NaN values in y")
        results['is_valid'] = False
        
    if np.isinf(X).any():
        inf_count = np.isinf(X).sum()
        results['issues'].append(f"Found {inf_count} infinite values in X")
        
    if np.isinf(y).any():
        inf_count = np.isinf(y).sum()
        results['issues'].append(f"Found {inf_count} infinite values in y")
        
    # Collect statistics
    if results['is_valid']:
        results['statistics'] = {
            'n_samples': X.shape[0],
            'sequence_length': X.shape[1],
            'n_features': X.shape[2],
            'X_mean': float(np.mean(X)),
            'X_std': float(np.std(X)),
            'X_min': float(np.min(X)),
            'X_max': float(np.max(X)),
            'y_mean': float(np.mean(y)),
            'y_std': float(np.std(y)),
            'y_min': float(np.min(y)),
            'y_max': float(np.max(y)),
            'feature_names': feature_names
        }
        
        # Per-feature statistics
        feature_stats = {}
        for i, feature in enumerate(feature_names):
            feature_data = X[:, :, i]
            feature_stats[feature] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data))
            }
        results['statistics']['feature_stats'] = feature_stats
    
    return results


def create_prediction_windows(data: pd.DataFrame,
                            window_size: int = 30,
                            overlap: float = 0.5) -> List[pd.DataFrame]:
    """
    Create overlapping prediction windows for model evaluation
    
    Args:
        data: Input DataFrame
        window_size: Size of each window
        overlap: Overlap ratio between windows (0-1)
        
    Returns:
        List of DataFrame windows
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError("Overlap must be between 0 and 1 (exclusive)")
        
    step_size = int(window_size * (1 - overlap))
    windows = []
    
    start = 0
    while start + window_size <= len(data):
        window = data.iloc[start:start + window_size].copy()
        windows.append(window)
        start += step_size
        
    return windows