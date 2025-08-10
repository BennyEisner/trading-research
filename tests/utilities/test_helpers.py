#!/usr/bin/env python3
"""
Test Helper Functions
Common utilities and helper functions for all test types
"""

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TestAssertions:
    """Custom assertions for financial data testing"""

    @staticmethod
    def assert_correlation_close(actual: float, expected: float, tolerance: float = 0.05, msg: str = None):
        """Assert that correlation values are close within tolerance"""
        diff = abs(actual - expected)
        if diff > tolerance:
            raise AssertionError(
                f"Correlation {actual:.6f} not close to expected {expected:.6f} "
                f"(diff: {diff:.6f}, tolerance: {tolerance:.6f}). {msg or ''}"
            )

    @staticmethod
    def assert_array_properties(
        arr: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        shape: Optional[Tuple] = None,
        dtype: Optional[type] = None,
        no_nan: bool = True,
        no_inf: bool = True,
    ):
        """Assert array has expected properties"""

        if shape is not None:
            assert arr.shape == shape, f"Expected shape {shape}, got {arr.shape}"

        if dtype is not None:
            assert arr.dtype == dtype, f"Expected dtype {dtype}, got {arr.dtype}"

        if no_nan:
            assert not np.any(np.isnan(arr)), "Array contains NaN values"

        if no_inf:
            assert not np.any(np.isinf(arr)), "Array contains infinite values"

        if min_val is not None:
            assert np.all(arr >= min_val), f"Array contains values < {min_val}"

        if max_val is not None:
            assert np.all(arr <= max_val), f"Array contains values > {max_val}"

    @staticmethod
    def assert_dataframe_properties(
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        no_null: bool = False,
    ):
        """Assert DataFrame has expected properties"""

        if expected_columns is not None:
            missing_cols = set(expected_columns) - set(df.columns)
            assert not missing_cols, f"Missing columns: {missing_cols}"

        if min_rows is not None:
            assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"

        if max_rows is not None:
            assert len(df) <= max_rows, f"Expected at most {max_rows} rows, got {len(df)}"

        if no_null:
            null_counts = df.isnull().sum()
            assert null_counts.sum() == 0, f"DataFrame contains null values: {null_counts[null_counts > 0]}"


class TestMetrics:
    """Test-specific metrics and calculations"""

    @staticmethod
    def calculate_test_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate correlation for testing with proper error handling"""

        if len(y_true) < 2:
            return 0.0

        # Check for constant arrays
        if np.var(y_true) < 1e-10 or np.var(y_pred) < 1e-10:
            return 0.0

        correlation = np.corrcoef(y_true, y_pred)[0, 1]

        # Handle NaN case
        if np.isnan(correlation):
            return 0.0

        return correlation

    @staticmethod
    def calculate_pattern_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        """Calculate pattern detection accuracy"""

        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)

        return np.mean(y_true_binary == y_pred_binary)

    @staticmethod
    def analyze_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Comprehensive prediction analysis for debugging"""

        return {
            "correlation": TestMetrics.calculate_test_correlation(y_true, y_pred),
            "mae": np.mean(np.abs(y_true - y_pred)),
            "mse": np.mean((y_true - y_pred) ** 2),
            "pattern_accuracy": TestMetrics.calculate_pattern_accuracy(y_true, y_pred),
            "pred_mean": np.mean(y_pred),
            "pred_std": np.std(y_pred),
            "pred_min": np.min(y_pred),
            "pred_max": np.max(y_pred),
            "target_mean": np.mean(y_true),
            "target_std": np.std(y_true),
        }


class TestTimer:
    """Simple test timing utilities"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        import time

        self.start_time = time.time()

    def stop(self):
        import time

        self.end_time = time.time()

    def elapsed(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @contextmanager
    def measure(self):
        """Context manager for timing code blocks"""
        self.start()
        try:
            yield self
        finally:
            self.stop()


class TestEnvironment:
    """Test environment management"""

    @staticmethod
    @contextmanager
    def temporary_directory():
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    @contextmanager
    def modified_environment(**env_vars):
        """Temporarily modify environment variables"""
        old_env = {}

        # Save old values and set new ones
        for key, value in env_vars.items():
            old_env[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)

        try:
            yield
        finally:
            # Restore old values
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    @staticmethod
    @contextmanager
    def suppress_tensorflow_warnings():
        """Suppress TensorFlow warnings during tests"""
        import warnings

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
            warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
            try:
                yield
            finally:
                os.environ.pop("TF_CPP_MIN_LOG_LEVEL", None)


class TestReporting:
    """Test reporting and debugging utilities"""

    @staticmethod
    def format_test_results(results: Dict[str, Any], title: str = "Test Results") -> str:
        """Format test results for readable output"""

        lines = [f"\n{'='*60}", f"{title:^60}", f"{'='*60}"]

        for key, value in results.items():
            if isinstance(value, float):
                lines.append(f"{key:30}: {value:10.6f}")
            elif isinstance(value, dict):
                lines.append(f"{key:30}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        lines.append(f"  {sub_key:28}: {sub_value:10.6f}")
                    else:
                        lines.append(f"  {sub_key:28}: {sub_value}")
            else:
                lines.append(f"{key:30}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def print_array_summary(arr: np.ndarray, name: str = "Array"):
        """Print summary statistics for an array"""

        print(f"\n{name} Summary:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  Mean:  {np.mean(arr):.6f}")
        print(f"  Std:   {np.std(arr):.6f}")
        print(f"  Min:   {np.min(arr):.6f}")
        print(f"  Max:   {np.max(arr):.6f}")

        if np.any(np.isnan(arr)):
            print(f"  NaN count: {np.sum(np.isnan(arr))}")
        if np.any(np.isinf(arr)):
            print(f"  Inf count: {np.sum(np.isinf(arr))}")


def skip_test_if_no_gpu():
    """Decorator to skip tests if no GPU is available"""

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            try:
                import tensorflow as tf

                if not tf.config.list_physical_devices("GPU"):
                    print(f"Skipping {test_func.__name__}: No GPU available")
                    return
            except ImportError:
                print(f"Skipping {test_func.__name__}: TensorFlow not available")
                return

            return test_func(*args, **kwargs)

        return wrapper

    return decorator


def slow_test(test_func):
    """Decorator to mark tests as slow (can be skipped in fast test runs)"""
    test_func._slow_test = True
    return test_func


def requires_data(data_type: str):
    """Decorator to mark tests that require specific data"""

    def decorator(test_func):
        test_func._requires_data = data_type
        return test_func

    return decorator

