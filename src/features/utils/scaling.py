#!/usr/bin/env python3

"""
Feature scaling utilities for financial time series
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Union


class FeatureScaler:
    """
    Comprehensive feature scaling for financial time series data
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
        self.excluded_features = ["open", "high", "low", "close", "volume"]  # Keep raw prices
        
    def fit_transform(self, 
                     data: pd.DataFrame,
                     method: str = "robust",
                     exclude_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit scalers and transform features
        
        Args:
            data: Input DataFrame
            method: Scaling method ('robust', 'standard', 'minmax')
            exclude_features: Additional features to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        exclude_list = self.excluded_features.copy()
        if exclude_features:
            exclude_list.extend(exclude_features)
            
        # Identify numeric features to scale
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        features_to_scale = [col for col in numeric_columns if col not in exclude_list]
        
        result_data = data.copy()
        
        # Choose scaler
        if method == "robust":
            scaler_class = RobustScaler
        elif method == "standard":
            scaler_class = StandardScaler
        elif method == "minmax":
            scaler_class = MinMaxScaler
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        # Scale each feature
        for feature in features_to_scale:
            if feature in data.columns and data[feature].notna().sum() > 10:
                try:
                    scaler = scaler_class()
                    
                    # Fit and transform
                    feature_values = data[[feature]].fillna(0)
                    scaled_values = scaler.fit_transform(feature_values)
                    result_data[feature] = scaled_values.flatten()
                    
                    # Store scaler and stats
                    self.scalers[feature] = scaler
                    self.feature_stats[feature] = {
                        'method': method,
                        'mean': data[feature].mean(),
                        'std': data[feature].std(),
                        'min': data[feature].min(),
                        'max': data[feature].max(),
                        'missing_count': data[feature].isna().sum()
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not scale feature {feature}: {e}")
                    
        return result_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted scalers
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with scaled features
        """
        if not self.scalers:
            raise ValueError("Scalers not fitted. Call fit_transform first.")
            
        result_data = data.copy()
        
        for feature, scaler in self.scalers.items():
            if feature in data.columns:
                try:
                    feature_values = data[[feature]].fillna(0)
                    scaled_values = scaler.transform(feature_values)
                    result_data[feature] = scaled_values.flatten()
                except Exception as e:
                    print(f"Warning: Could not transform feature {feature}: {e}")
                    
        return result_data
    
    def inverse_transform(self, 
                         data: pd.DataFrame, 
                         features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inverse transform scaled features back to original scale
        
        Args:
            data: DataFrame with scaled features
            features: Specific features to inverse transform (None for all)
            
        Returns:
            DataFrame with inverse transformed features
        """
        if not self.scalers:
            raise ValueError("Scalers not fitted.")
            
        result_data = data.copy()
        features_to_inverse = features or list(self.scalers.keys())
        
        for feature in features_to_inverse:
            if feature in self.scalers and feature in data.columns:
                try:
                    scaler = self.scalers[feature]
                    feature_values = data[[feature]]
                    original_values = scaler.inverse_transform(feature_values)
                    result_data[feature] = original_values.flatten()
                except Exception as e:
                    print(f"Warning: Could not inverse transform feature {feature}: {e}")
                    
        return result_data
    
    def get_feature_stats(self) -> pd.DataFrame:
        """
        Get statistics for all scaled features
        
        Returns:
            DataFrame with feature statistics
        """
        if not self.feature_stats:
            return pd.DataFrame()
            
        stats_data = []
        for feature, stats in self.feature_stats.items():
            stats_data.append({
                'feature': feature,
                'method': stats['method'],
                'original_mean': stats['mean'],
                'original_std': stats['std'],
                'original_min': stats['min'],
                'original_max': stats['max'],
                'missing_count': stats['missing_count']
            })
            
        return pd.DataFrame(stats_data)
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers to file"""
        import pickle
        
        scaler_data = {
            'scalers': self.scalers,
            'feature_stats': self.feature_stats,
            'excluded_features': self.excluded_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
            
    def load_scalers(self, filepath: str):
        """Load scalers from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
            
        self.scalers = scaler_data['scalers']
        self.feature_stats = scaler_data['feature_stats']
        self.excluded_features = scaler_data['excluded_features']


def scale_features_by_group(data: pd.DataFrame, 
                           feature_groups: Dict[str, List[str]],
                           group_methods: Dict[str, str]) -> pd.DataFrame:
    """
    Scale different feature groups using different methods
    
    Args:
        data: Input DataFrame
        feature_groups: Dictionary mapping group names to feature lists
        group_methods: Dictionary mapping group names to scaling methods
        
    Returns:
        DataFrame with scaled features
    """
    result_data = data.copy()
    
    for group_name, features in feature_groups.items():
        method = group_methods.get(group_name, "robust")
        
        # Get available features for this group
        available_features = [f for f in features if f in data.columns]
        
        if available_features:
            scaler = FeatureScaler()
            
            # Scale only features from this group
            group_data = data[available_features]
            scaled_group = scaler.fit_transform(group_data, method=method, exclude_features=[])
            
            # Update result with scaled features
            for feature in available_features:
                if feature in scaled_group.columns:
                    result_data[feature] = scaled_group[feature]
                    
    return result_data