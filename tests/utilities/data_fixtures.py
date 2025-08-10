#!/usr/bin/env python3
"""
Test Data Fixtures
Consolidated test data generation and fixtures for all test types
"""

import os
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TestDataGenerator:
    """Generate standardized test data for different testing scenarios"""

    @staticmethod
    def generate_ohlcv_data(
        symbol: str = "AAPL", days: int = 100, start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Generate realistic OHLCV data for testing"""

        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)

        dates = pd.date_range(start=start_date, periods=days, freq="D")

        # Generate realistic price movements
        np.random.seed(42)  # Deterministic for testing
        returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility

        # Starting price
        initial_price = 100.0
        prices = [initial_price]

        for i in range(1, days):
            prices.append(prices[-1] * (1 + returns[i]))

        # Generate OHLC from close prices
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + np.random.uniform(0, 0.02))
            low = price * (1 - np.random.uniform(0, 0.02))
            open_price = prices[i - 1] if i > 0 else price
            volume = np.random.randint(1000000, 10000000)

            data.append(
                {"Date": dates[i], "open": open_price, "high": high, "low": low, "close": price, "volume": volume}
            )

        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)
        return df

    @staticmethod
    def generate_multi_ticker_data(tickers: List[str] = None, days: int = 100) -> Dict[str, pd.DataFrame]:
        """Generate OHLCV data for multiple tickers"""

        if tickers is None:
            tickers = ["AAPL", "MSFT", "GOOG"]

        ticker_data = {}
        base_date = datetime.now() - timedelta(days=days)

        for i, ticker in enumerate(tickers):
            # Slight variation in start dates to test alignment
            start_date = base_date + timedelta(days=i)
            ticker_data[ticker] = TestDataGenerator.generate_ohlcv_data(
                symbol=ticker, days=days - i, start_date=start_date  # Slightly different lengths
            )

        return ticker_data

    @staticmethod
    def generate_feature_matrix(
        samples: int = 100, features: int = 17, sequence_length: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate feature matrix and targets for testing"""

        np.random.seed(42)

        # Generate sequences
        X = np.random.randn(samples, sequence_length, features).astype(np.float32)

        # Generate targets with some correlation to features
        # Use first feature as basis for target generation
        target_correlation = 0.3
        noise_level = 0.7

        feature_influence = np.mean(X[:, -5:, 0], axis=1)  # Last 5 days of first feature
        noise = np.random.randn(samples)

        y = (target_correlation * feature_influence + noise_level * noise + 0.5).astype(
            np.float32
        )  # Shift to positive range

        # Clip to [0, 1] range for pattern confidence scores
        y = np.clip(y, 0.0, 1.0)

        return X, y

    @staticmethod
    def generate_correlation_test_data(correlation: float = 0.5, samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data with specific correlation for testing correlation metrics"""

        np.random.seed(42)

        # Generate correlated data
        x = np.random.randn(samples)
        noise = np.random.randn(samples)

        y = correlation * x + np.sqrt(1 - correlation**2) * noise

        # Scale to [0, 1] range
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
        y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))

        return x_scaled.astype(np.float32), y_scaled.astype(np.float32)


class TestConfigurationFixtures:
    """Standard test configurations and settings"""

    @staticmethod
    def get_minimal_config() -> Dict[str, Any]:
        """Minimal configuration for fast tests"""
        return {
            "model": {
                "lookback_window": 5,
                "prediction_horizon": 1,
                "model_params": {"units": [8, 4], "dropout_rate": 0.1, "learning_rate": 0.01, "batch_size": 16},
            },
            "training": {"epochs": 2, "patience": 1},
        }

    @staticmethod
    def get_standard_config() -> Dict[str, Any]:
        """Standard configuration for integration tests"""
        return {
            "model": {
                "lookback_window": 20,
                "prediction_horizon": 3,
                "model_params": {"units": [32, 16], "dropout_rate": 0.3, "learning_rate": 0.001, "batch_size": 32},
            },
            "training": {"epochs": 5, "patience": 3},
        }


class MockModelFixtures:
    """Mock models and components for testing"""

    @staticmethod
    def create_mock_lstm_model(input_shape: Tuple[int, int] = (20, 17), output_dim: int = 1):
        """Create a minimal LSTM model for testing"""
        import tensorflow as tf

        model = tf.keras.Sequential(
            [tf.keras.layers.LSTM(8, input_shape=input_shape), tf.keras.layers.Dense(output_dim, activation="sigmoid")]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        return model

    @staticmethod
    def create_mock_predictions(samples: int = 100, correlation: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Create mock predictions with specific correlation to targets"""

        np.random.seed(42)

        # Generate targets
        y_true = np.random.uniform(0, 1, samples).astype(np.float32)

        # Generate predictions with desired correlation
        noise = np.random.randn(samples)
        y_pred = correlation * y_true + (1 - correlation) * noise

        # Scale predictions to [0, 1] and add some realistic variance
        y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
        y_pred = y_pred * 0.8 + 0.1  # Keep in [0.1, 0.9] range

        return y_true, y_pred.astype(np.float32)


class TempFileFixtures:
    """Temporary file management for testing"""

    @staticmethod
    def create_temp_config_file(config_data: Dict[str, Any]) -> str:
        """Create temporary configuration file"""
        import json

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)

        json.dump(config_data, temp_file, indent=2)
        temp_file.close()

        return temp_file.name

    @staticmethod
    def create_temp_csv_data(df: pd.DataFrame) -> str:
        """Create temporary CSV file with data"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

        df.to_csv(temp_file.name)
        temp_file.close()

        return temp_file.name

    @staticmethod
    def cleanup_temp_file(file_path: str):
        """Clean up temporary file"""
        try:
            os.unlink(file_path)
        except OSError:
            pass


# Standard test fixtures that can be imported
STANDARD_TEST_DATA = TestDataGenerator.generate_multi_ticker_data(["AAPL", "MSFT"], days=50)
MINIMAL_CONFIG = TestConfigurationFixtures.get_minimal_config()
CORRELATION_TEST_DATA = TestDataGenerator.generate_correlation_test_data(0.5, 100)

# Test constants
TEST_TICKERS = ["AAPL", "MSFT", "GOOG"]
TEST_FEATURES = 17
TEST_SEQUENCE_LENGTH = 20
TEST_SAMPLES = 100

