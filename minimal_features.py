#!/usr/bin/env python3
"""
Minimal feature engineering for testing
"""

import numpy as np
import pandas as pd

class MinimalFeatureEngineer:
    """Minimal feature engineering for testing"""
    
    def calculate_all_features(self, data):
        """Calculate basic features only"""
        print(f"Calculating minimal features")
        data = data.copy()
        
        # Basic price features
        data['price_change'] = data['close'].pct_change()
        data['high_low_ratio'] = data['high'] / data['low']
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Simple moving averages
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['price_to_sma5'] = data['close'] / data['sma_5']
        data['price_to_sma20'] = data['close'] / data['sma_20']
        
        # Basic technical indicators
        # Simple RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        data['volatility_20d'] = data['daily_return'].rolling(20).std()
        
        # Fill NaN values
        data = data.bfill().ffill()
        
        return data
    
    def select_stable_features(self, data):
        """Select stable features"""
        feature_columns = [
            'price_change', 'high_low_ratio', 'volume_ratio',
            'price_to_sma5', 'price_to_sma20', 'rsi', 'volatility_20d'
        ]
        
        # Only return features that exist and have no NaN
        available_features = []
        for col in feature_columns:
            if col in data.columns and not data[col].isna().all():
                available_features.append(col)
        
        return available_features