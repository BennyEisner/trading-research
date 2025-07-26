#!/usr/bin/env python3

"""
Unit tests for PipelineValidator
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from validation.pipeline_validator import PipelineValidator, create_pipeline_validator


class TestPipelineValidatorBasic(unittest.TestCase):
    """Test core pipeline validator functionality"""

    def setUp(self):
        """Set up validator with default settings"""
        self.validator = PipelineValidator()

    def test_initialization(self):
        """Test validator initializes with correct defaults and custom values"""

        # Default initialization
        validator = PipelineValidator()
        self.assertEqual(validator.extreme_price_threshold, 0.5)
        self.assertEqual(validator.correlation_threshold, 0.95)

        # Custom initialization
        custom_validator = PipelineValidator(extreme_price_threshold=0.3, correlation_threshold=0.9)
        self.assertEqual(custom_validator.extreme_price_threshold, 0.3)
        self.assertEqual(custom_validator.correlation_threshold, 0.9)

    def test_helper_function(self):
        """Test create_pipeline_validator helper function"""
        validator = create_pipeline_validator(outlier_std_threshold=5.0)

        self.assertIsInstance(validator, PipelineValidator)
        self.assertEqual(validator.outlier_std_threshold, 5.0)


class TestRawDataValidation(unittest.TestCase):
    """Test raw OHLCV data validation"""

    def setUp(self):
        self.validator = PipelineValidator()

        # Create clean test data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.normal(0, 2, 100))

        self.clean_data = pd.DataFrame(
            {
                "date": dates,
                "open": prices + np.random.normal(0, 0.5, 100),
                "high": prices + np.random.uniform(0, 2, 100),
                "low": prices - np.random.uniform(0, 2, 100),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, 100),
            }
        )

        # Ensure OHLC relationships are valid
        for i in range(len(self.clean_data)):
            row = self.clean_data.iloc[i]
            max_price = max(row["open"], row["close"])
            min_price = min(row["open"], row["close"])

            self.clean_data.at[i, "high"] = max(row["high"], max_price)
            self.clean_data.at[i, "low"] = min(row["low"], min_price)

    def test_clean_data_passes(self):
        """Test that clean data passes validation"""
        is_valid, issues = self.validator.validate_raw_data(self.clean_data, "TEST")

        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

        # Check results are stored
        self.assertIn("TEST_raw_data", self.validator.validation_results)

    def test_detects_missing_columns(self):
        """Test detection of missing required columns"""
        incomplete_data = self.clean_data.drop(columns=["volume", "high"])

        is_valid, issues = self.validator.validate_raw_data(incomplete_data, "INCOMPLETE")

        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required columns" in issue for issue in issues))

    def test_detects_invalid_ohlc(self):
        """Test detection of invalid OHLC relationships"""
        bad_data = self.clean_data.copy()

        # Make high < low (invalid)
        bad_data.loc[10, "high"] = bad_data.loc[10, "low"] - 5

        is_valid, issues = self.validator.validate_raw_data(bad_data, "BAD_OHLC")

        self.assertFalse(is_valid)
        self.assertTrue(any("OHLC" in issue for issue in issues))

    def test_detects_negative_values(self):
        """Test detection of negative prices/volumes"""
        bad_data = self.clean_data.copy()

        # Negative price and volume
        bad_data.loc[5, "close"] = -10
        bad_data.loc[15, "volume"] = -1000

        is_valid, issues = self.validator.validate_raw_data(bad_data, "NEGATIVE")

        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)

    def test_empty_data_handling(self):
        """Test handling of empty DataFrame"""
        empty_data = pd.DataFrame()

        is_valid, issues = self.validator.validate_raw_data(empty_data, "EMPTY")

        self.assertFalse(is_valid)
        self.assertIn("Empty DataFrame", issues[0])


class TestFeatureDataValidation(unittest.TestCase):
    """Test feature data validation"""

    def setUp(self):
        self.validator = PipelineValidator()

        # Create clean feature data
        np.random.seed(42)
        self.clean_features = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 200),
                "feature_2": np.random.normal(2, 1.5, 200),
                "feature_3": np.random.uniform(-1, 1, 200),
                "feature_4": np.random.exponential(2, 200),
                "feature_5": np.random.normal(0, 0.5, 200),
            }
        )
        self.feature_columns = self.clean_features.columns.tolist()

    def test_clean_features_pass(self):
        """Test that clean features pass validation"""
        is_valid, issues = self.validator.validate_feature_data(self.clean_features, self.feature_columns)

        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

    def test_detects_constant_features(self):
        """Test detection of constant features"""
        bad_features = self.clean_features.copy()
        bad_features["constant_feature"] = 5.0  # All same value

        feature_cols = bad_features.columns.tolist()

        is_valid, issues = self.validator.validate_feature_data(bad_features, feature_cols)

        self.assertFalse(is_valid)
        self.assertTrue(any("Constant features" in issue for issue in issues))

    def test_detects_high_nan_features(self):
        """Test detection of features with high NaN counts"""
        bad_features = self.clean_features.copy()

        # Create feature with 50% NaN values
        bad_features["high_nan_feature"] = np.random.normal(0, 1, 200)
        bad_features.loc[:99, "high_nan_feature"] = np.nan

        feature_cols = bad_features.columns.tolist()

        is_valid, issues = self.validator.validate_feature_data(bad_features, feature_cols)

        self.assertFalse(is_valid)
        self.assertTrue(any("High NaN counts" in issue for issue in issues))

    def test_detects_infinite_values(self):
        """Test detection of infinite values"""
        bad_features = self.clean_features.copy()
        bad_features.loc[50:54, "feature_1"] = np.inf

        is_valid, issues = self.validator.validate_feature_data(bad_features, self.feature_columns)

        self.assertFalse(is_valid)
        self.assertTrue(any("Infinite values" in issue for issue in issues))

    def test_edge_cases(self):
        """Test edge cases"""
        # Empty feature list
        is_valid_empty, issues_empty = self.validator.validate_feature_data(self.clean_features, [])
        self.assertFalse(is_valid_empty)

        # Missing feature columns
        is_valid_missing, issues_missing = self.validator.validate_feature_data(
            self.clean_features, ["nonexistent_feature"]
        )
        self.assertFalse(is_valid_missing)


class TestSequenceValidation(unittest.TestCase):
    """Test LSTM sequence validation"""

    def setUp(self):
        self.validator = PipelineValidator()

        # Create clean 3D sequences
        np.random.seed(42)
        self.X = np.random.normal(0, 1, (500, 60, 10))
        self.y = np.random.normal(0, 0.02, 500)

    def test_clean_sequences_pass(self):
        """Test that clean sequences pass validation"""
        is_valid, issues = self.validator.validate_sequences(self.X, self.y)

        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

        # Check results stored
        result = self.validator.validation_results["sequence_validation"]
        self.assertEqual(result["sequence_shape"], self.X.shape)
        self.assertEqual(result["target_shape"], self.y.shape)

    def test_detects_wrong_dimensions(self):
        """Test detection of wrong array dimensions"""
        # Wrong X dimensions (2D instead of 3D)
        X_2d = np.random.random((100, 60))
        y_1d = np.random.random(100)

        is_valid, issues = self.validator.validate_sequences(X_2d, y_1d)

        self.assertFalse(is_valid)
        self.assertTrue(any("3D" in issue for issue in issues))

    def test_detects_mismatched_lengths(self):
        """Test detection of mismatched array lengths"""
        X_mismatch = self.X[:400]  # 400 samples
        y_mismatch = self.y[:350]  # 350 samples

        is_valid, issues = self.validator.validate_sequences(X_mismatch, y_mismatch)

        self.assertFalse(is_valid)
        self.assertTrue(any("mismatch" in issue for issue in issues))

    def test_detects_nan_values(self):
        """Test detection of NaN values"""
        X_nan = self.X.copy()
        y_nan = self.y.copy()

        X_nan[10:15, :, 0] = np.nan
        y_nan[50:55] = np.nan

        is_valid, issues = self.validator.validate_sequences(X_nan, y_nan)

        self.assertFalse(is_valid)
        nan_issues = [issue for issue in issues if "NaN" in issue]
        self.assertGreater(len(nan_issues), 0)

    def test_empty_arrays(self):
        """Test handling of empty arrays"""
        is_valid, issues = self.validator.validate_sequences(np.array([]), np.array([]))

        self.assertFalse(is_valid)
        self.assertIn("Empty arrays", issues[0])


class TestPredictionValidation(unittest.TestCase):
    """Test model prediction validation"""

    def setUp(self):
        self.validator = PipelineValidator()

        # Create reasonable predictions
        np.random.seed(42)
        self.actual = np.random.normal(0.001, 0.02, 300)
        self.good_predictions = self.actual * 0.7 + np.random.normal(0, 0.01, 300)

    def test_good_predictions_pass(self):
        """Test that reasonable predictions pass validation"""
        is_valid, issues = self.validator.validate_model_predictions(self.actual, self.good_predictions)

        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

        # Check metrics calculated
        result = self.validator.validation_results["prediction_validation"]
        self.assertGreater(result["directional_accuracy"], 40.0)  # Better than min threshold
        self.assertGreater(abs(result["correlation"]), 0.1)

    def test_detects_nan_predictions(self):
        """Test detection of NaN predictions"""
        bad_predictions = self.good_predictions.copy()
        bad_predictions[100:110] = np.nan

        is_valid, issues = self.validator.validate_model_predictions(self.actual, bad_predictions)

        self.assertFalse(is_valid)
        self.assertTrue(any("NaN predictions" in issue for issue in issues))

    def test_detects_identical_predictions(self):
        """Test detection of identical predictions"""
        identical_predictions = np.full_like(self.actual, 0.001)

        is_valid, issues = self.validator.validate_model_predictions(self.actual, identical_predictions)

        self.assertFalse(is_valid)
        self.assertTrue(any("identical" in issue or "Very low prediction variance" in issue for issue in issues))

    def test_detects_poor_directional_accuracy(self):
        """Test detection of poor directional accuracy"""
        # Create predictions with opposite signs (terrible directional accuracy)
        terrible_predictions = -self.actual

        is_valid, issues = self.validator.validate_model_predictions(self.actual, terrible_predictions)

        self.assertFalse(is_valid)
        self.assertTrue(any("directional accuracy" in issue for issue in issues))

    def test_edge_cases(self):
        """Test edge cases"""
        # Empty arrays
        is_valid_empty, issues_empty = self.validator.validate_model_predictions(np.array([]), np.array([]))
        self.assertFalse(is_valid_empty)

        # Mismatched lengths
        is_valid_mismatch, issues_mismatch = self.validator.validate_model_predictions(
            self.actual, self.good_predictions[:250]
        )
        self.assertFalse(is_valid_mismatch)


class TestTrainingStabilityValidation(unittest.TestCase):
    """Test training stability validation"""

    def setUp(self):
        self.validator = PipelineValidator()

    def test_stable_training_passes(self):
        """Test that stable training history passes"""
        # Create decreasing loss history with very low variance at the end
        stable_history = {"loss": [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.32, 0.31, 0.305, 0.301]}

        is_valid, issues = self.validator.validate_training_stability(stable_history)

        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

    def test_detects_loss_explosion(self):
        """Test detection of exploding loss"""
        exploding_history = {"loss": [1.0, 0.8, 0.7, 2.0, 10.0, 50.0]}  # Loss explodes

        is_valid, issues = self.validator.validate_training_stability(exploding_history)

        self.assertFalse(is_valid)
        self.assertTrue(any("explosion" in issue.lower() for issue in issues))

    def test_detects_nan_losses(self):
        """Test detection of NaN losses"""
        nan_history = {"loss": [1.0, 0.8, np.nan, 0.6, np.nan]}

        is_valid, issues = self.validator.validate_training_stability(nan_history)

        self.assertFalse(is_valid)
        self.assertTrue(any("NaN" in issue for issue in issues))

    def test_edge_cases(self):
        """Test edge cases"""
        # No history
        is_valid_no_history, issues_no_history = self.validator.validate_training_stability({})
        self.assertFalse(is_valid_no_history)

        # Empty loss list
        is_valid_empty, issues_empty = self.validator.validate_training_stability({"loss": []})
        self.assertFalse(is_valid_empty)


class TestValidationSummary(unittest.TestCase):
    """Test validation summary and logging functionality"""

    def setUp(self):
        self.validator = PipelineValidator()

    def test_validation_summary(self):
        """Test validation summary generation"""
        # Run some validations
        clean_data = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50),
                "open": np.random.uniform(95, 105, 50),
                "high": np.random.uniform(100, 110, 50),
                "low": np.random.uniform(90, 100, 50),
                "close": np.random.uniform(95, 105, 50),
                "volume": np.random.randint(1000000, 5000000, 50),
            }
        )

        self.validator.validate_raw_data(clean_data, "TEST")

        summary = self.validator.get_validation_summary()

        # Check summary structure
        required_keys = ["timestamp", "validation_results", "overall_status", "total_issues"]
        for key in required_keys:
            self.assertIn(key, summary)

        self.assertIn(summary["overall_status"], ["PASS", "FAIL"])
        self.assertIsInstance(summary["total_issues"], int)

    def test_logging_functionality(self):
        """Test logging methods don't crash"""
        # Test basic logging
        try:
            self.validator.log("Test message", "INFO")
            self.validator.log("Warning", "WARNING")
        except Exception as e:
            self.fail(f"Logging failed: {e}")

        # Test summary logging
        try:
            clean_data = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=10),
                    "open": [100] * 10,
                    "high": [102] * 10,
                    "low": [98] * 10,
                    "close": [101] * 10,
                    "volume": [1000000] * 10,
                }
            )

            self.validator.validate_raw_data(clean_data, "TEST")
            summary = self.validator.log_validation_summary()

            self.assertIsInstance(summary, dict)
        except Exception as e:
            self.fail(f"Summary logging failed: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
