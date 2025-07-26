#!/usr/bin/env python3

"""
Test data generators for validation framework testing
Provides controlled synthetic data for testing validation logic
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class TestDataGenerator:
    """Generate controlled synthetic data for testing validation frameworks"""
    
    @staticmethod
    def create_perfect_predictions(n_samples: int = 1000, noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create returns where predictions perfectly match actual (should pass all tests)"""
        np.random.seed(42)
        actual_returns = np.random.normal(0.001, 0.02, n_samples)
        predictions = actual_returns + np.random.normal(0, noise_level, n_samples)
        return actual_returns, predictions
    
    @staticmethod
    def create_random_predictions(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Create completely random predictions (should fail significance tests)"""
        np.random.seed(42)
        actual_returns = np.random.normal(0.001, 0.02, n_samples)
        predictions = np.random.normal(0.0, 0.02, n_samples)
        return actual_returns, predictions
    
    @staticmethod
    def create_biased_predictions(n_samples: int = 1000, bias: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
        """Create predictions with some signal but imperfect (borderline significance)"""
        np.random.seed(42)
        actual_returns = np.random.normal(0.001, 0.02, n_samples)
        predictions = actual_returns * bias + np.random.normal(0, 0.015, n_samples)
        return actual_returns, predictions
    
    @staticmethod
    def create_ohlcv_data(n_samples: int = 1000, start_price: float = 100.0) -> pd.DataFrame:
        """Create realistic OHLCV data for pipeline validation"""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # Generate price series with random walk + trend
        returns = np.random.normal(0.0005, 0.02, n_samples)
        prices = [start_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data with proper relationships
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate open and close first
            open_price = price * (1 + np.random.normal(0, 0.005))
            close = price
            
            # Ensure high and low respect OHLC relationships
            min_price = min(open_price, close)
            max_price = max(open_price, close)
            
            # High should be at least the max of open/close, plus some upward movement
            high_premium = abs(np.random.normal(0, 0.01))
            high = max_price * (1 + high_premium)
            
            # Low should be at most the min of open/close, minus some downward movement  
            low_discount = abs(np.random.normal(0, 0.01))
            low = min_price * (1 - low_discount)
            
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_sequences_3d(n_samples: int = 1000, sequence_length: int = 60, n_features: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create 3D sequences for LSTM validation"""
        np.random.seed(42)
        
        # Generate feature time series
        all_features = np.random.normal(0, 1, (n_samples + sequence_length, n_features))
        
        # Add some autocorrelation to make it realistic
        for i in range(1, len(all_features)):
            all_features[i] = 0.1 * all_features[i-1] + 0.9 * all_features[i]
        
        # Create sequences
        X = []
        y = []
        for i in range(sequence_length, n_samples + sequence_length):
            X.append(all_features[i-sequence_length:i])
            # Target is next period's first feature (like returns)
            y.append(all_features[i, 0])
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def create_problematic_data() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Create various problematic datasets to test validation robustness"""
        np.random.seed(42)
        
        problems = {}
        
        # Base actual returns
        actual = np.random.normal(0.001, 0.02, 500)
        
        # All predictions are zero
        pred_zeros = np.zeros_like(actual)
        problems['all_zeros'] = (actual, pred_zeros)
        
        # All predictions identical (but not zero)
        pred_identical = np.full_like(actual, 0.001)
        problems['all_identical'] = (actual, pred_identical)
        
        # Contains NaN values
        pred_nan = actual.copy() + np.random.normal(0, 0.01, len(actual))
        pred_nan[100:110] = np.nan
        problems['contains_nan'] = (actual, pred_nan)
        
        # Contains infinite values
        pred_inf = actual.copy() + np.random.normal(0, 0.01, len(actual))
        pred_inf[200:205] = np.inf
        problems['contains_inf'] = (actual, pred_inf)
        
        # Extreme outliers
        pred_outliers = actual.copy() + np.random.normal(0, 0.01, len(actual))
        pred_outliers[300:305] = 100.0  # Extreme values
        problems['extreme_outliers'] = (actual, pred_outliers)
        
        # Very small dataset
        small_actual = actual[:10]
        small_pred = actual[:10] + np.random.normal(0, 0.001, 10)
        problems['very_small'] = (small_actual, small_pred)
        
        return problems
    
    @staticmethod
    def create_time_series_with_autocorrelation(n_samples: int = 1000, alpha: float = 0.3) -> np.ndarray:
        """Create AR(1) time series with known autocorrelation"""
        np.random.seed(42)
        series = np.zeros(n_samples)
        series[0] = np.random.normal(0, 1)
        
        for i in range(1, n_samples):
            series[i] = alpha * series[i-1] + np.random.normal(0, 1)
        
        return series
    
    @staticmethod
    def create_regime_labels(n_samples: int = 1000, regime_prob: float = 0.2) -> np.ndarray:
        """Create synthetic regime labels (0=stable, 1=transition)"""
        np.random.seed(42)
        return np.random.choice([0, 1], size=n_samples, p=[1-regime_prob, regime_prob])
    
    @staticmethod
    def create_feature_data(n_samples: int = 1000, n_features: int = 15) -> pd.DataFrame:
        """Create feature-like data for pipeline testing"""
        np.random.seed(42)
        
        # Generate different types of features
        feature_data = {}
        
        # Normal features
        for i in range(n_features // 3):
            feature_data[f'normal_feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Skewed features
        for i in range(n_features // 3):
            feature_data[f'skewed_feature_{i}'] = np.random.exponential(2, n_samples)
        
        # Features with some NaN values
        for i in range(n_features - 2 * (n_features // 3)):
            data = np.random.normal(0, 1, n_samples)
            nan_indices = np.random.choice(n_samples, size=n_samples//20, replace=False)
            data[nan_indices] = np.nan
            feature_data[f'sparse_feature_{i}'] = data
        
        return pd.DataFrame(feature_data)
    
    @staticmethod
    def create_training_history(n_epochs: int = 100, stable: bool = True) -> Dict:
        """Create training history for validation testing"""
        np.random.seed(42)
        
        if stable:
            # Stable training - loss decreases
            initial_loss = 1.0
            losses = []
            val_losses = []
            
            for epoch in range(n_epochs):
                # Training loss decreases with noise
                loss = initial_loss * np.exp(-epoch * 0.02) + np.random.normal(0, 0.01)
                val_loss = loss + np.random.normal(0, 0.005)  # Slightly higher than training
                
                losses.append(max(loss, 0.001))  # Don't go negative
                val_losses.append(max(val_loss, 0.001))
            
            return {'loss': losses, 'val_loss': val_losses}
        
        else:
            # Unstable training - exploding gradients
            losses = []
            val_losses = []
            
            for epoch in range(n_epochs):
                if epoch < 20:
                    # Normal start
                    loss = 1.0 - epoch * 0.02 + np.random.normal(0, 0.01)
                else:
                    # Explosion
                    loss = (epoch - 20) ** 1.5 + np.random.normal(0, 0.1)
                
                val_loss = loss + np.random.normal(0, 0.1)
                losses.append(loss)
                val_losses.append(val_loss)
            
            return {'loss': losses, 'val_loss': val_losses}


class ValidationTestFixtures:
    """Pre-built test fixtures for common validation scenarios"""
    
    @classmethod
    def get_perfect_model_scenario(cls) -> Dict:
        """Perfect model that should pass all validation tests"""
        actual, predictions = TestDataGenerator.create_perfect_predictions(noise_level=0.001)
        X, y = TestDataGenerator.create_sequences_3d(n_samples=800)
        
        return {
            'actual_returns': actual,
            'predictions': predictions,
            'sequences_X': X,
            'sequences_y': y,
            'expected_validation_result': True,
            'description': 'Perfect predictions with minimal noise'
        }
    
    @classmethod
    def get_random_model_scenario(cls) -> Dict:
        """Random model that should fail validation tests"""
        actual, predictions = TestDataGenerator.create_random_predictions()
        X, y = TestDataGenerator.create_sequences_3d(n_samples=800)
        
        return {
            'actual_returns': actual,
            'predictions': predictions,
            'sequences_X': X,
            'sequences_y': y,
            'expected_validation_result': False,
            'description': 'Completely random predictions'
        }
    
    @classmethod
    def get_marginal_model_scenario(cls) -> Dict:
        """Marginally predictive model for borderline testing"""
        actual, predictions = TestDataGenerator.create_biased_predictions(bias=0.4)
        X, y = TestDataGenerator.create_sequences_3d(n_samples=800)
        
        return {
            'actual_returns': actual,
            'predictions': predictions,
            'sequences_X': X,
            'sequences_y': y,
            'expected_validation_result': None,  # Could go either way
            'description': 'Marginally predictive model'
        }
    
    @classmethod
    def get_clean_pipeline_data(cls) -> Dict:
        """Clean data that should pass pipeline validation"""
        ohlcv_data = TestDataGenerator.create_ohlcv_data(n_samples=1000)
        feature_data = TestDataGenerator.create_feature_data(n_samples=900, n_features=12)
        
        return {
            'ohlcv_data': ohlcv_data,
            'feature_data': feature_data,
            'feature_columns': feature_data.columns.tolist(),
            'expected_validation_result': True,
            'description': 'Clean synthetic OHLCV and feature data'
        }
    
    @classmethod
    def get_problematic_pipeline_data(cls) -> Dict:
        """Problematic data that should trigger validation warnings"""
        # Create base clean data
        ohlcv_data = TestDataGenerator.create_ohlcv_data(n_samples=1000)
        feature_data = TestDataGenerator.create_feature_data(n_samples=900, n_features=12)
        
        # Introduce problems
        # Invalid OHLC relationships
        ohlcv_data.loc[100, 'high'] = ohlcv_data.loc[100, 'low'] - 1
        
        # Negative prices
        ohlcv_data.loc[200, 'close'] = -5
        
        # Add constant feature
        feature_data['constant_feature'] = 1.0
        
        # Add highly correlated features
        feature_data['corr_feature_1'] = feature_data.iloc[:, 0] + np.random.normal(0, 0.001, len(feature_data))
        feature_data['corr_feature_2'] = feature_data['corr_feature_1'] + np.random.normal(0, 0.001, len(feature_data))
        
        return {
            'ohlcv_data': ohlcv_data,
            'feature_data': feature_data,
            'feature_columns': feature_data.columns.tolist(),
            'expected_validation_result': False,
            'description': 'Data with quality issues for testing validation detection'
        }