#!/usr/bin/env python3

"""
Unit tests for RobustTimeSeriesValidator
"""

import sys
import unittest
from pathlib import Path

import numpy as np

current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from validation.robust_time_series_validator import RobustTimeSeriesValidator, create_robust_validator


class TestRobustValidatorBasic(unittest.TestCase):
    """Test basic functionality and initialization"""

    def setUp(self):
        """Set up validator with test settings"""
        self.validator = RobustTimeSeriesValidator(block_size=5, n_bootstrap=50, random_state=42)

    def test_initialization(self):
        """Test validator initializes correctly"""
        validator = RobustTimeSeriesValidator(block_size=10, n_bootstrap=100, random_state=123)

        self.assertEqual(validator.block_size, 10)
        self.assertEqual(validator.n_bootstrap, 100)
        self.assertEqual(validator.random_state, 123)
        self.assertEqual(validator.validation_results, {})

    def test_helper_function(self):
        """test create_robust_validator helper function"""
        validator = create_robust_validator(block_size=7, n_bootstrap=200)

        self.assertIsInstance(validator, RobustTimeSeriesValidator)
        self.assertEqual(validator.block_size, 7)
        self.assertEqual(validator.n_bootstrap, 200)


class TestMetricCalculations(unittest.TestCase):
    """Test metric calculation methods"""

    def setUp(self):
        self.validator = RobustTimeSeriesValidator(random_state=42)

    def test_directional_accuracy(self):
        """Test directional accuracy calculation"""
        # Perfect predictions
        returns = np.array([0.01, -0.02, 0.03, -0.01])
        predictions = np.array([0.015, -0.025, 0.025, -0.015])

        accuracy = self.validator._directional_accuracy(returns, predictions)
        self.assertEqual(accuracy, 1.0)

        # Opposite predictions
        opposite_pred = np.array([-0.015, 0.025, -0.025, 0.015])
        accuracy_opposite = self.validator._directional_accuracy(returns, opposite_pred)
        self.assertEqual(accuracy_opposite, 0.0)

        # Half correct
        half_pred = np.array([0.015, 0.025, 0.025, 0.015])
        accuracy_half = self.validator._directional_accuracy(returns, half_pred)
        self.assertEqual(accuracy_half, 0.5)

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        predictions = np.array([0.015, -0.025, 0.025, -0.015, 0.006])

        sharpe = self.validator._sharpe_ratio(returns, predictions)

        self.assertIsInstance(sharpe, float)
        self.assertFalse(np.isnan(sharpe))

        # Zero std case
        constant_returns = np.array([0.01, 0.01, 0.01, 0.01])
        constant_pred = np.array([1.0, 1.0, 1.0, 1.0])

        sharpe_zero = self.validator._sharpe_ratio(constant_returns, constant_pred)
        self.assertEqual(sharpe_zero, 0.0)


class TestMovingBlockBootstrap(unittest.TestCase):
    """Test moving block bootstrap functionality"""

    def setUp(self):
        self.validator = RobustTimeSeriesValidator(
            block_size=5, n_bootstrap=30, random_state=42  # Small for fast testing
        )

    def test_create_moving_blocks(self):
        """Test moving block creation"""
        data = np.arange(20)  # [0, 1, 2, ..., 19]

        bootstrap_sample = self.validator._create_moving_blocks(data, block_size=5)

        # Should be reasonable length
        self.assertGreater(len(bootstrap_sample), 10)

        # Values should be from original data
        unique_values = set(bootstrap_sample)
        original_values = set(data)
        self.assertTrue(unique_values.issubset(original_values))

    def test_bootstrap_with_perfect_predictions(self):
        """Test bootstrap with perfect predictions"""
        np.random.seed(42)
        actual = np.random.normal(0, 0.02, 100)
        predictions = actual + np.random.normal(0, 0.001, 100)  # Nearly perfect

        results = self.validator.moving_block_bootstrap(actual, predictions)

        # Check structure
        required_keys = ["actual_performance", "bootstrap_mean", "p_value", "significant"]
        for key in required_keys:
            self.assertIn(key, results)

        # Perfect predictions should have high performance
        self.assertGreater(results["actual_performance"], 0.8)

        # Should be significant
        self.assertLess(results["p_value"], 0.05)
        self.assertTrue(results["significant"])

    def test_bootstrap_with_random_predictions(self):
        """Test bootstrap with random predictions"""
        np.random.seed(42)
        actual = np.random.normal(0, 0.02, 100)
        predictions = np.random.normal(0, 0.02, 100)  # Random

        results = self.validator.moving_block_bootstrap(actual, predictions)

        # Random predictions should have ~50% accuracy
        self.assertGreater(results["actual_performance"], 0.3)
        self.assertLess(results["actual_performance"], 0.7)

        # Should not be significant
        self.assertGreater(results["p_value"], 0.05)
        self.assertFalse(results["significant"])


class TestPermutationTest(unittest.TestCase):
    """Test conditional permutation testing"""

    def setUp(self):
        self.validator = RobustTimeSeriesValidator(n_bootstrap=30, random_state=42)  # Small for fast testing

    def test_permutation_with_good_predictions(self):
        """Test permutation test with good predictions"""
        np.random.seed(42)
        actual = np.random.normal(0, 0.02, 100)
        predictions = actual * 0.8 + np.random.normal(0, 0.01, 100)  # Good signal

        results = self.validator.conditional_permutation_test(actual, predictions)

        # Check structure
        required_keys = ["actual_performance", "null_mean", "p_value", "significant"]
        for key in required_keys:
            self.assertIn(key, results)

        # Good predictions should outperform null
        self.assertGreater(results["actual_performance"], results["null_mean"])

        # Should be significant
        self.assertLess(results["p_value"], 0.1)  # More lenient for small sample

    def test_permutation_with_random_predictions(self):
        """Test permutation test with random predictions"""
        np.random.seed(42)
        actual = np.random.normal(0, 0.02, 100)
        predictions = np.random.normal(0, 0.02, 100)  # Random

        results = self.validator.conditional_permutation_test(actual, predictions)

        # Random predictions should be similar to null
        performance_diff = abs(results["actual_performance"] - results["null_mean"])
        self.assertLess(performance_diff, 0.15)  # Should be close

        # Should not be significant
        self.assertGreater(results["p_value"], 0.05)


class TestModelSignificanceValidation(unittest.TestCase):
    """Test comprehensive model significance validation"""

    def setUp(self):
        self.validator = RobustTimeSeriesValidator(n_bootstrap=25, random_state=42)  # Small for fast testing

    def test_comprehensive_validation_structure(self):
        """Test that comprehensive validation has correct structure"""
        np.random.seed(42)
        actual = np.random.normal(0, 0.02, 150)
        predictions = actual * 0.7 + np.random.normal(0, 0.01, 150)

        results = self.validator.validate_model_significance(
            actual, predictions, test_types=["bootstrap", "permutation"]
        )

        # Check main structure
        main_keys = ["timestamp", "data_summary", "overall_assessment"]
        for key in main_keys:
            self.assertIn(key, results)

        # Check data summary
        data_summary = results["data_summary"]
        self.assertEqual(data_summary["n_samples"], 150)
        self.assertIsInstance(data_summary["returns_mean"], float)
        self.assertIsInstance(data_summary["returns_std"], float)

        # Check overall assessment
        overall = results["overall_assessment"]
        assessment_keys = ["all_p_values", "min_p_value", "all_significant", "recommendation"]
        for key in assessment_keys:
            self.assertIn(key, overall)

        # P-values should be valid
        self.assertEqual(len(overall["all_p_values"]), 2)  # bootstrap + permutation
        for p_val in overall["all_p_values"]:
            self.assertGreaterEqual(p_val, 0.0)
            self.assertLessEqual(p_val, 1.0)

        # Recommendation should be valid
        self.assertIn(overall["recommendation"], ["PASS", "FAIL"])

    def test_single_test_validation(self):
        """Test validation with single test type"""
        np.random.seed(42)
        actual = np.random.normal(0, 0.02, 80)
        predictions = actual + np.random.normal(0, 0.005, 80)  # Good predictions

        # Bootstrap only
        results_bootstrap = self.validator.validate_model_significance(actual, predictions, test_types=["bootstrap"])

        self.assertIn("moving_block_bootstrap", results_bootstrap)
        self.assertNotIn("conditional_permutation", results_bootstrap)
        self.assertEqual(len(results_bootstrap["overall_assessment"]["all_p_values"]), 1)

        # Permutation only
        results_permutation = self.validator.validate_model_significance(
            actual, predictions, test_types=["permutation"]
        )

        self.assertNotIn("moving_block_bootstrap", results_permutation)
        self.assertIn("conditional_permutation", results_permutation)
        self.assertEqual(len(results_permutation["overall_assessment"]["all_p_values"]), 1)

    def test_perfect_vs_random_predictions(self):
        """Test validation distinguishes between perfect and random predictions"""
        np.random.seed(42)
        actual = np.random.normal(0, 0.02, 100)

        # Perfect predictions
        perfect_pred = actual + np.random.normal(0, 0.001, 100)
        results_perfect = self.validator.validate_model_significance(actual, perfect_pred, test_types=["bootstrap"])

        # Random predictions
        random_pred = np.random.normal(0, 0.02, 100)
        results_random = self.validator.validate_model_significance(actual, random_pred, test_types=["bootstrap"])

        # Perfect should have lower p-value than random
        p_perfect = results_perfect["overall_assessment"]["min_p_value"]
        p_random = results_random["overall_assessment"]["min_p_value"]

        self.assertLess(p_perfect, p_random)

        # perfect should pass, random should fail
        self.assertEqual(results_perfect["overall_assessment"]["recommendation"], "PASS")
        self.assertEqual(results_random["overall_assessment"]["recommendation"], "FAIL")


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.validator = RobustTimeSeriesValidator(n_bootstrap=10, random_state=42)

    def test_small_datasets(self):
        """Test behavior with small datasets"""
        small_actual = np.array([0.01, -0.01, 0.02])
        small_pred = np.array([0.015, -0.015, 0.025])

        # Should not crash
        results = self.validator.moving_block_bootstrap(small_actual, small_pred)
        self.assertIn("p_value", results)
        self.assertIsInstance(results["p_value"], float)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        valid_data = np.array([0.01, -0.01, 0.02, -0.005])

        # Empty arrays - should handle gracefully and return valid results
        try:
            result = self.validator.moving_block_bootstrap(np.array([]), np.array([]))
            # Should return a dictionary with expected keys even if data is problematic
            self.assertIn("p_value", result)
        except (ValueError, IndexError):
            # This is also acceptable behavior for empty arrays
            pass

        # Mismatched lengths - should handle gracefully or raise exception
        try:
            result = self.validator.moving_block_bootstrap(valid_data, np.array([0.01, 0.02]))
            # If it returns a result, it should have the expected structure
            self.assertIn("p_value", result)
        except (ValueError, IndexError):
            # This is also acceptable behavior for mismatched arrays
            pass

    def test_custom_metric_function(self):
        """Test using custom metric function"""
        np.random.seed(42)
        actual = np.random.normal(0, 0.02, 50)
        predictions = actual * 0.8 + np.random.normal(0, 0.01, 50)

        # Custom correlation metric
        def correlation_metric(returns, preds):
            if len(returns) < 2:
                return 0.0
            corr_matrix = np.corrcoef(returns, preds)
            return abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        results = self.validator.moving_block_bootstrap(actual, predictions, correlation_metric)

        self.assertIn("actual_performance", results)
        self.assertGreaterEqual(results["actual_performance"], 0.0)
        self.assertLessEqual(results["actual_performance"], 1.0)


class TestUtilityMethods(unittest.TestCase):
    """Test utility and summary methods"""

    def setUp(self):
        self.validator = RobustTimeSeriesValidator(n_bootstrap=20, random_state=42)

    def test_print_validation_summary(self):
        """Test validation summary printing"""
        # Run validation to populate results
        np.random.seed(42)
        actual = np.random.normal(0, 0.02, 60)
        predictions = actual * 0.6 + np.random.normal(0, 0.01, 60)

        self.validator.validate_model_significance(actual, predictions)

        try:
            self.validator.print_validation_summary()
        except Exception as e:
            self.fail(f"print_validation_summary raised {e}")

    def test_empty_validator_summary(self):
        """Test summary with no validation results"""
        empty_validator = RobustTimeSeriesValidator()

        # Should handle empty results gracefully
        try:
            empty_validator.print_validation_summary()
        except Exception as e:
            self.fail(f"print_validation_summary with no results raised {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

