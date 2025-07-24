#!/usr/bin/env python3

"""
Robust Feature Scaling with Outlier Handling
Fixes the critical scaling issues preventing model learning
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Optional, Tuple


class RobustFeatureScaler:
    """
    Feature scaler with robust outlier handling before scaling
    
    This fixes the critical issue where extreme outliers in features
    (e.g., momentum_oscillator ranging [-418, 294]) break StandardScaler
    and prevent gradient-based learning.
    """
    
    def __init__(self, 
                 outlier_method: str = "quantile",
                 outlier_threshold: float = 0.01,
                 scaler_method: str = "standard"):
        """
        Initialize robust scaler
        
        Args:
            outlier_method: 'quantile' or 'zscore'
            outlier_threshold: For quantile: percentile to clip (0.01 = 1%-99%)
                              For zscore: number of standard deviations (3.0)
            scaler_method: 'standard', 'robust', or 'minmax'
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.scaler_method = scaler_method
        
        # Storage for fitted parameters
        self.scaler = None
        self.outlier_bounds = {}
        self.feature_stats = {}
        
    def fit_transform(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """
        Fit outlier bounds and scaler, then transform features
        
        Args:
            X: Feature array (samples, features) or (samples, timesteps, features)
            feature_names: Optional feature names for logging
            
        Returns:
            Robustly scaled feature array
        """
        original_shape = X.shape
        
        # Handle 3D input (LSTM format)
        if len(original_shape) == 3:
            n_samples, n_timesteps, n_features = original_shape
            X_reshaped = X.reshape(-1, n_features)
        else:
            X_reshaped = X
            n_features = X.shape[1]
        
        # Initialize scaler
        if self.scaler_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler method: {self.scaler_method}")
            
        # Calculate outlier bounds for each feature
        X_clipped = X_reshaped.copy()
        
        for feature_idx in range(n_features):
            feature_name = feature_names[feature_idx] if (feature_names and feature_idx < len(feature_names)) else f"feature_{feature_idx}"
            feature_values = X_reshaped[:, feature_idx]
            
            # Calculate bounds based on method
            if self.outlier_method == "quantile":
                lower_bound = np.percentile(feature_values, self.outlier_threshold * 100)
                upper_bound = np.percentile(feature_values, (1 - self.outlier_threshold) * 100)
            elif self.outlier_method == "zscore":
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                lower_bound = mean_val - self.outlier_threshold * std_val
                upper_bound = mean_val + self.outlier_threshold * std_val
            else:
                raise ValueError(f"Unknown outlier method: {self.outlier_method}")
            
            # Store bounds
            self.outlier_bounds[feature_name] = (lower_bound, upper_bound)
            
            # Apply clipping
            X_clipped[:, feature_idx] = np.clip(feature_values, lower_bound, upper_bound)
            
            # Store statistics for debugging
            original_range = (np.min(feature_values), np.max(feature_values))
            clipped_range = (np.min(X_clipped[:, feature_idx]), np.max(X_clipped[:, feature_idx]))
            outliers_clipped = np.sum((feature_values < lower_bound) | (feature_values > upper_bound))
            
            self.feature_stats[feature_name] = {
                'original_range': original_range,
                'clipped_range': clipped_range,
                'outliers_clipped': outliers_clipped,
                'bounds': (lower_bound, upper_bound)
            }
        
        # Fit and transform with scaler
        X_scaled = self.scaler.fit_transform(X_clipped)
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            X_scaled = X_scaled.reshape(original_shape)
            
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted outlier bounds and scaler
        
        Args:
            X: Feature array to transform
            
        Returns:
            Robustly scaled feature array
        """
        if self.scaler is None:
            raise ValueError("Must call fit_transform first")
            
        original_shape = X.shape
        
        # Handle 3D input
        if len(original_shape) == 3:
            X_reshaped = X.reshape(-1, X.shape[-1])
        else:
            X_reshaped = X
            
        # Apply outlier clipping using fitted bounds
        X_clipped = X_reshaped.copy()
        
        for feature_idx, (feature_name, (lower_bound, upper_bound)) in enumerate(self.outlier_bounds.items()):
            if feature_idx < X_reshaped.shape[1]:  # Safety check
                X_clipped[:, feature_idx] = np.clip(X_reshaped[:, feature_idx], lower_bound, upper_bound)
        
        # Apply fitted scaler
        X_scaled = self.scaler.transform(X_clipped)
        
        # Restore original shape
        if len(original_shape) == 3:
            X_scaled = X_scaled.reshape(original_shape)
            
        return X_scaled
    
    def get_outlier_report(self) -> pd.DataFrame:
        """
        Get detailed report of outlier handling and scaling
        
        Returns:
            DataFrame with outlier statistics for each feature
        """
        if not self.feature_stats:
            return pd.DataFrame()
            
        report_data = []
        for feature_name, stats in self.feature_stats.items():
            report_data.append({
                'feature': feature_name,
                'original_min': stats['original_range'][0],
                'original_max': stats['original_range'][1],
                'clipped_min': stats['clipped_range'][0],
                'clipped_max': stats['clipped_range'][1],
                'lower_bound': stats['bounds'][0],
                'upper_bound': stats['bounds'][1],
                'outliers_clipped': stats['outliers_clipped']
            })
            
        return pd.DataFrame(report_data)
    
    def print_scaling_summary(self):
        """Print summary of outlier handling and scaling"""
        if not self.feature_stats:
            print("No scaling statistics available")
            return
            
        print(f"\n🛠️  ROBUST SCALING SUMMARY")
        print("=" * 50)
        print(f"Outlier method: {self.outlier_method}")
        print(f"Scaler method: {self.scaler_method}")
        
        total_outliers = sum(stats['outliers_clipped'] for stats in self.feature_stats.values())
        print(f"Total outliers clipped: {total_outliers}")
        
        print(f"\nFeature-wise outlier handling:")
        for feature_name, stats in self.feature_stats.items():
            orig_range = stats['original_range']
            clip_range = stats['clipped_range']
            outliers = stats['outliers_clipped']
            
            if outliers > 0:
                print(f"  {feature_name}: {outliers} outliers")
                print(f"    Original: [{orig_range[0]:.3f}, {orig_range[1]:.3f}]")
                print(f"    Clipped:  [{clip_range[0]:.3f}, {clip_range[1]:.3f}]")


def create_robust_scaler_for_pipeline() -> RobustFeatureScaler:
    """
    Create a robust scaler configured for the financial ML pipeline
    
    Returns:
        Configured RobustFeatureScaler for production use
    """
    return RobustFeatureScaler(
        outlier_method="quantile",
        outlier_threshold=0.005,  # Clip extreme 0.5% on each side
        scaler_method="standard"
    )