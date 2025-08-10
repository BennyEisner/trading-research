#!/usr/bin/env python3

"""
Core functionality tests for EnhancedRobustnessTests
Tests basic robustness validation methods with clean, focused tests
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add the src directory to the path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from validation.enhanced_robustness_tests import EnhancedRobustnessTests, create_robustness_tester


class TestEnhancedRobustnessBasic(unittest.TestCase):
    """Test core functionality of enhanced robustness testing"""

    def setUp(self):
        """Set up test environment"""
        self.tester = EnhancedRobustnessTests()

    def test_initialization(self):
        """Test tester initializes with correct defaults"""
        self.assertEqual(self.tester.significance_level, 0.05)
        self.assertEqual(self.tester.eigenvalue_threshold, 0.1)
        self.assertEqual(self.tester.min_sample_size, 30)
        self.assertEqual(self.tester.test_results, {})

    def test_custom_initialization(self):
        """Test tester initializes with custom parameters"""
        tester = EnhancedRobustnessTests(significance_level=0.01, eigenvalue_threshold=0.2, min_sample_size=50)

        self.assertEqual(tester.significance_level, 0.01)
        self.assertEqual(tester.eigenvalue_threshold, 0.2)
        self.assertEqual(tester.min_sample_size, 50)

    def test_create_robustness_tester_helper(self):
        """Test helper function creates tester correctly"""
        tester = create_robustness_tester(significance_level=0.01)

        self.assertIsInstance(tester, EnhancedRobustnessTests)
        self.assertEqual(tester.significance_level, 0.01)


class TestDirectionalAccuracy(unittest.TestCase):
    """Test directional accuracy calculations"""

    def setUp(self):
        self.tester = EnhancedRobustnessTests()

    def test_perfect_predictions(self):
        """Test directional accuracy with perfect predictions"""
        returns = np.array([0.01, -0.02, 0.03, -0.01])
        predictions = np.array([0.015, -0.025, 0.025, -0.015])

        accuracy = self.tester._directional_accuracy(returns, predictions)
        self.assertEqual(accuracy, 1.0)

    def test_no_correct_predictions(self):
        """Test directional accuracy when no predictions match"""
        returns = np.array([0.01, -0.02, 0.03, -0.01])
        predictions = np.array([-0.015, 0.025, -0.025, 0.015])

        accuracy = self.tester._directional_accuracy(returns, predictions)
        self.assertEqual(accuracy, 0.0)

    def test_half_correct_predictions(self):
        """Test directional accuracy with half correct"""
        returns = np.array([0.01, -0.02, 0.03, -0.01])
        predictions = np.array([0.015, 0.025, 0.025, 0.015])  # First 2 wrong direction

        accuracy = self.tester._directional_accuracy(returns, predictions)
        self.assertEqual(accuracy, 0.5)


class TestAutocorrelationPreservation(unittest.TestCase):
    """Test autocorrelation preservation validation"""

    def setUp(self):
        self.tester = EnhancedRobustnessTests()
        np.random.seed(42)  # For reproducible tests

    def test_similar_series_pass(self):
        """Test that similar series pass autocorrelation test"""
        # Create autocorrelated series
        original = np.cumsum(np.random.normal(0, 1, 200))

        # Create similar series with small noise
        similar = original + np.random.normal(0, 0.01, len(original))

        results = self.tester.autocorrelation_preservation_test(original, similar, tolerance=0.2)

        self.assertTrue(results["acf_preserved"])
        self.assertEqual(results["validation"], "PASS")
        self.assertLess(results["avg_acf_difference"], results["tolerance"])

    def test_different_series_fail(self):
        """Test that different series fail autocorrelation test"""
        original = np.cumsum(np.random.normal(0, 1, 200))
        permuted = np.random.permutation(original)

        results = self.tester.autocorrelation_preservation_test(original, permuted)

        self.assertFalse(results["acf_preserved"])
        self.assertEqual(results["validation"], "FAIL")
        self.assertGreater(results["avg_acf_difference"], results["tolerance"])

    def test_result_structure(self):
        """Test that results have correct structure"""
        original = np.random.normal(0, 1, 100)
        permuted = np.random.permutation(original)

        results = self.tester.autocorrelation_preservation_test(original, permuted)

        expected_keys = [
            "acf_preserved",
            "avg_acf_difference",
            "max_acf_difference",
            "tolerance",
            "validation",
            "interpretation",
        ]

        for key in expected_keys:
            self.assertIn(key, results)


class TestRegimeTransitionTesting(unittest.TestCase):
    """Test regime transition performance validation"""

    def setUp(self):
        self.tester = EnhancedRobustnessTests()
        np.random.seed(42)

    def test_performance_degradation_detection(self):
        """Test detection of performance degradation during transitions"""
        n_samples = 200
        actual_returns = np.random.normal(0.001, 0.02, n_samples)

        # Create regime labels (80% stable, 20% transition)
        regime_labels = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

        # Create predictions that perform worse during transitions
        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            if regime_labels[i] == 0:  # Stable
                predictions[i] = actual_returns[i] * 0.8 + np.random.normal(0, 0.005)
            else:  # Transition
                predictions[i] = actual_returns[i] * 0.3 + np.random.normal(0, 0.02)

        results = self.tester.regime_transition_testing(predictions, actual_returns, regime_labels)

        # Should detect degradation
        self.assertGreater(results["stable_accuracy"], results["transition_accuracy"])
        self.assertGreater(results["degradation_pct"], 0)

    def test_result_structure(self):
        """Test that results have correct structure"""
        n_samples = 100
        predictions = np.random.normal(0, 0.01, n_samples)
        actual_returns = np.random.normal(0.001, 0.02, n_samples)
        regime_labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

        results = self.tester.regime_transition_testing(predictions, actual_returns, regime_labels)

        expected_keys = [
            "stable_accuracy",
            "transition_accuracy",
            "degradation_pct",
            "acceptable_degradation",
            "validation",
            "interpretation",
        ]

        for key in expected_keys:
            self.assertIn(key, results)


class TestMarketStressTesting(unittest.TestCase):
    """Test market stress performance validation"""

    def setUp(self):
        self.tester = EnhancedRobustnessTests()
        np.random.seed(42)

    def test_stress_detection(self):
        """Test detection of performance during market stress"""
        n_samples = 300

        # Create returns with varying volatility
        base_returns = np.random.normal(0.001, 0.01, n_samples)

        # Add high volatility periods
        stress_periods = np.random.choice(n_samples, size=n_samples // 5, replace=False)
        for i in stress_periods:
            base_returns[i] = np.random.normal(0, 0.05)

        # Create predictions that perform worse during stress
        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            if i in stress_periods:
                # Make stress performance much worse by using opposite correlation
                predictions[i] = -base_returns[i] * 0.5 + np.random.normal(0, 0.02)
            else:
                predictions[i] = base_returns[i] * 0.8 + np.random.normal(0, 0.005)

        results = self.tester.market_stress_testing(predictions, base_returns)

        # Should detect worse performance during stress (allow some tolerance for randomness)
        stress_much_worse = results["stress_accuracy"] < results["normal_accuracy"] - 5.0
        relative_performance_worse = results["relative_performance"] < 95.0
        
        # Test passes if either condition is met (due to randomness in synthetic data)
        self.assertTrue(stress_much_worse or relative_performance_worse, 
                       f"Expected stress performance degradation. Stress: {results['stress_accuracy']:.1f}%, "
                       f"Normal: {results['normal_accuracy']:.1f}%, Relative: {results['relative_performance']:.1f}%")

    def test_result_structure(self):
        """Test that results have correct structure"""
        n_samples = 100
        predictions = np.random.normal(0, 0.01, n_samples)
        returns = np.random.normal(0, 0.02, n_samples)

        results = self.tester.market_stress_testing(predictions, returns)

        expected_keys = [
            "stress_accuracy",
            "normal_accuracy",
            "stress_acceptable",
            "relative_performance",
            "validation",
        ]

        for key in expected_keys:
            self.assertIn(key, results)


class TestMultipleTestingCorrection(unittest.TestCase):
    """Test multiple testing correction methods"""

    def setUp(self):
        self.tester = EnhancedRobustnessTests()

    def test_fdr_correction(self):
        """Test FDR correction with known p-values"""
        p_values = [0.01, 0.03, 0.05, 0.07, 0.12]

        results = self.tester.multiple_testing_correction(p_values, method="fdr_bh")

        self.assertEqual(results["original_p_values"], p_values)
        self.assertEqual(len(results["corrected_p_values"]), len(p_values))
        self.assertEqual(results["method"], "fdr_bh")

        # Correction should be more conservative
        original_sig = results["n_significant_original"]
        corrected_sig = results["n_significant_corrected"]
        self.assertLessEqual(corrected_sig, original_sig)

    def test_bonferroni_correction(self):
        """Test Bonferroni correction"""
        p_values = [0.01, 0.02, 0.03]

        results = self.tester.multiple_testing_correction(p_values, method="bonferroni")

        self.assertEqual(results["method"], "bonferroni")
        self.assertIn("alpha_bonferroni", results)
        self.assertIn("alpha_sidak", results)

    def test_result_structure(self):
        """Test correction results have correct structure"""
        p_values = [0.02, 0.05, 0.08]

        results = self.tester.multiple_testing_correction(p_values)

        expected_keys = [
            "original_p_values",
            "corrected_p_values",
            "rejected",
            "method",
            "alpha",
            "n_significant_original",
            "n_significant_corrected",
        ]

        for key in expected_keys:
            self.assertIn(key, results)


if __name__ == "__main__":
    unittest.main(verbosity=2)

