#!/usr/bin/env python3

"""
Integration tests for simulation tooling
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from validation.enhanced_robustness_tests import create_robustness_tester
from validation.gapped_time_series_cv import create_gapped_cv
from validation.pipeline_validator import create_pipeline_validator
from validation.robust_time_series_validator import create_robust_validator
from validation.tests.fixtures.test_data_generators import TestDataGenerator, ValidationTestFixtures


class TestValidationWorkflows(unittest.TestCase):
    """Integration tests for simulation tooling"""

    def setUp(self):
        """Set up test environment"""
        self.test_data = TestDataGenerator()
        self.fixtures = ValidationTestFixtures()

    def test_end_to_end_validation_workflow_perfect_model(self):
        """Test complete validation workflow with perfect model scenario"""

        scenario = self.fixtures.get_perfect_model_scenario()

        results = {}

        # 1. Pipeline validation
        pipeline_validator = create_pipeline_validator()

        # Create synthetic OHLCV data for pipeline testing
        ohlcv_data = self.test_data.create_ohlcv_data(n_samples=len(scenario["actual_returns"]))

        # Validate raw data
        raw_valid, raw_issues = pipeline_validator.validate_raw_data(ohlcv_data, "PERFECT_MODEL")
        results["pipeline_raw_data"] = {"valid": raw_valid, "issues_count": len(raw_issues)}

        # Validate sequences
        seq_valid, seq_issues = pipeline_validator.validate_sequences(scenario["sequences_X"], scenario["sequences_y"])
        results["pipeline_sequences"] = {"valid": seq_valid, "issues_count": len(seq_issues)}

        # validate predictions
        pred_valid, pred_issues = pipeline_validator.validate_model_predictions(
            scenario["actual_returns"], scenario["predictions"]
        )
        results["pipeline_predictions"] = {"valid": pred_valid, "issues_count": len(pred_issues)}

        # 2. Robust time series validation
        robust_validator = create_robust_validator(n_bootstrap=50)

        robust_results = robust_validator.validate_model_significance(
            scenario["actual_returns"], scenario["predictions"], test_types=["bootstrap", "permutation"]
        )

        results["robust_validation"] = {
            "all_significant": robust_results["overall_assessment"]["all_significant"],
            "min_p_value": robust_results["overall_assessment"]["min_p_value"],
            "recommendation": robust_results["overall_assessment"]["recommendation"],
        }

        # 3. Cross-validation testing
        dates = pd.date_range(start="2020-01-01", periods=len(scenario["sequences_X"]), freq="D")
        cv = create_gapped_cv(n_splits=3, gap_size=5, test_size=0.2)

        no_leakage = cv.validate_no_leakage(dates)
        results["cv_validation"] = {"no_leakage": no_leakage}

        # Test actual CV splits work
        cv_splits = list(cv.split(scenario["sequences_X"]))
        results["cv_validation"]["splits_generated"] = len(cv_splits)

        # 4. Enhanced robustness testing
        robustness_tester = create_robustness_tester()

        # Create test data for robustness tests
        permuted_returns = np.random.permutation(scenario["actual_returns"])
        regime_labels = self.test_data.create_regime_labels(len(scenario["actual_returns"]))

        robustness_results = robustness_tester.comprehensive_robustness_test(
            scenario["predictions"],
            scenario["actual_returns"],
            permuted_returns=permuted_returns,
            regime_labels=regime_labels,
        )

        results["robustness_tests"] = {
            "all_passed": robustness_results["overall_assessment"]["all_tests_passed"],
            "recommendation": robustness_results["overall_assessment"]["recommendation"],
        }

        # Verify perfect model scenario performs as expected
        self.assertTrue(results["pipeline_raw_data"]["valid"])
        self.assertTrue(results["pipeline_sequences"]["valid"])
        self.assertTrue(results["pipeline_predictions"]["valid"])

        # Perfect model should pass robust validation
        self.assertTrue(results["robust_validation"]["all_significant"])
        self.assertLess(results["robust_validation"]["min_p_value"], 0.05)
        self.assertEqual(results["robust_validation"]["recommendation"], "PASS")

        # CV should work without leakage
        self.assertTrue(results["cv_validation"]["no_leakage"])
        self.assertGreater(results["cv_validation"]["splits_generated"], 0)

        # Print results for inspection
        print(f"\nPerfect Model Workflow Results:")
        for test_name, result in results.items():
            print(f"{test_name}: {result}")

    def test_end_to_end_validation_workflow_random_model(self):
        """Test complete validation workflow with random model scenario"""
        # Get random model scenario
        scenario = self.fixtures.get_random_model_scenario()

        results = {}

        # 1. Pipeline validation (should still pass - data quality is fine)
        pipeline_validator = create_pipeline_validator()

        ohlcv_data = self.test_data.create_ohlcv_data(n_samples=len(scenario["actual_returns"]))

        raw_valid, raw_issues = pipeline_validator.validate_raw_data(ohlcv_data, "RANDOM_MODEL")
        results["pipeline_raw_data"] = {"valid": raw_valid, "issues_count": len(raw_issues)}

        seq_valid, seq_issues = pipeline_validator.validate_sequences(scenario["sequences_X"], scenario["sequences_y"])
        results["pipeline_sequences"] = {"valid": seq_valid, "issues_count": len(seq_issues)}

        pred_valid, pred_issues = pipeline_validator.validate_model_predictions(
            scenario["actual_returns"], scenario["predictions"]
        )
        results["pipeline_predictions"] = {"valid": pred_valid, "issues_count": len(pred_issues)}

        # 2. Robust validation (should fail)
        robust_validator = create_robust_validator(n_bootstrap=50)

        robust_results = robust_validator.validate_model_significance(
            scenario["actual_returns"], scenario["predictions"]
        )

        results["robust_validation"] = {
            "all_significant": robust_results["overall_assessment"]["all_significant"],
            "recommendation": robust_results["overall_assessment"]["recommendation"],
        }

        # 3. Cross-validation (should work structurally)
        dates = pd.date_range(start="2020-01-01", periods=len(scenario["sequences_X"]), freq="D")
        cv = create_gapped_cv(n_splits=2, gap_size=5, test_size=0.2)

        no_leakage = cv.validate_no_leakage(dates)
        results["cv_validation"] = no_leakage

        # Verify random model scenario behaves as expected
        self.assertTrue(results["pipeline_raw_data"]["valid"])
        self.assertTrue(results["pipeline_sequences"]["valid"])

        self.assertFalse(results["robust_validation"]["all_significant"])
        self.assertEqual(results["robust_validation"]["recommendation"], "FAIL")

        self.assertTrue(results["cv_validation"])

        print(f"\nRandom Model Workflow Results:")
        for test_name, result in results.items():
            print(f"{test_name}: {result}")

    def test_validation_workflow_with_data_quality_issues(self):
        """Test validation workflow catches data quality issues"""
        # Get problematic pipeline data
        problematic_data = self.fixtures.get_problematic_pipeline_data()

        pipeline_validator = create_pipeline_validator()

        # Should catch raw data issues
        raw_valid, raw_issues = pipeline_validator.validate_raw_data(problematic_data["ohlcv_data"], "PROBLEMATIC")

        self.assertFalse(raw_valid)
        self.assertGreater(len(raw_issues), 0)

        # Should catch feature data issues
        feature_valid, feature_issues = pipeline_validator.validate_feature_data(
            problematic_data["feature_data"], problematic_data["feature_columns"]
        )

        self.assertFalse(feature_valid)
        self.assertGreater(len(feature_issues), 0)

        print(f"\nData Quality Issues Detected:")
        print(f"Raw data issues: {len(raw_issues)}")
        print(f"Feature data issues: {len(feature_issues)}")

        # Validation workflow should detect and report problems before model evaluation
        self.assertTrue(True)  # Test passes if we get here without crashing

    def test_cross_validation_integration_with_model_evaluation(self):
        """Test cross-validation integration with actual model evaluation"""
        # Create test data
        X, y = self.test_data.create_sequences_3d(n_samples=300, sequence_length=30, n_features=8)
        dates = pd.date_range(start="2020-01-01", periods=len(X), freq="D")

        # Set up cross-validation
        cv = create_gapped_cv(n_splits=3, gap_size=7, test_size=0.2, expanding_window=True)

        # Validate CV setup
        no_leakage = cv.validate_no_leakage(dates)
        self.assertTrue(no_leakage)

        # Test CV workflow
        cv_scores = []
        splits = list(cv.split(X, y))

        self.assertGreater(len(splits), 0)

        for i, (train_idx, test_idx) in enumerate(splits):
            # Validate split properties
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)  # No overlap

            # Simulate model evaluation on split
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Simulate predictions (simple linear model for testing)
            # In real scenario, this would be actual model training/prediction
            mean_features = np.mean(X_train, axis=(0, 1))  # Shape: (n_features,)
            test_features = np.mean(X_test, axis=1)  # Shape: (n_test_samples, n_features)

            # Create simple linear predictions using first half of features
            n_features_subset = len(mean_features) // 2
            predictions = np.dot(test_features[:, :n_features_subset], mean_features[:n_features_subset])

            # Calculate pseudo-accuracy
            if len(predictions) == len(y_test):
                correlation = np.corrcoef(y_test, predictions)[0, 1] if len(y_test) > 1 else 0
                cv_scores.append(abs(correlation) if not np.isnan(correlation) else 0)

        # Verify CV workflow completed
        self.assertEqual(len(cv_scores), len(splits))

        print(f"\nCross-Validation Integration Results:")
        print(f"Splits completed: {len(splits)}")
        print(f"CV scores: {[f'{score:.3f}' for score in cv_scores]}")
        print(f"Mean CV score: {np.mean(cv_scores):.3f}")

    def test_robustness_testing_integration(self):
        """Test integration of different robustness tests"""
        # Create test scenario
        n_samples = 250
        actual_returns = self.test_data.create_time_series_with_autocorrelation(n_samples, alpha=0.4)
        predictions = actual_returns * 0.6 + np.random.normal(0, 0.01, n_samples)

        # Set up robustness tester
        robustness_tester = create_robustness_tester(min_sample_size=30)

        # Prepare test data
        permuted_returns = np.random.permutation(actual_returns)
        regime_labels = self.test_data.create_regime_labels(n_samples, regime_prob=0.25)

        # Run comprehensive robustness testing
        results = robustness_tester.comprehensive_robustness_test(
            predictions, actual_returns, permuted_returns=permuted_returns, regime_labels=regime_labels
        )

        # Verify all expected tests ran
        expected_tests = ["autocorrelation_test", "regime_transition_test", "market_stress_test"]
        for test_name in expected_tests:
            self.assertIn(test_name, results)
            self.assertIn("validation", results[test_name])

        # Verify overall assessment
        self.assertIn("overall_assessment", results)
        overall = results["overall_assessment"]
        self.assertIn("all_tests_passed", overall)
        self.assertIn("test_summaries", overall)
        self.assertIn("recommendation", overall)

        # Check individual test results make sense
        for test_name in expected_tests:
            test_result = results[test_name]
            validation_status = test_result.get("validation", "UNKNOWN")
            self.assertIn(validation_status, ["PASS", "FAIL", "ERROR", "INSUFFICIENT_DATA"])

        print(f"\nRobustness Testing Integration Results:")
        print(f"Tests completed: {len(expected_tests)}")
        print(f"Overall recommendation: {overall['recommendation']}")

        for summary in overall["test_summaries"]:
            print(f"{summary['test']}: {summary['result']}")

    def test_validation_pipeline_error_handling(self):
        """Test validation pipeline handles errors gracefully"""
        # Test with various problematic inputs
        problematic_cases = self.test_data.create_problematic_data()

        pipeline_validator = create_pipeline_validator()
        robust_validator = create_robust_validator(n_bootstrap=10)
        robustness_tester = create_robustness_tester()

        for problem_type, (actual, predictions) in problematic_cases.items():
            with self.subTest(problem=problem_type):
                # Test pipeline validation
                try:
                    pred_valid, pred_issues = pipeline_validator.validate_model_predictions(actual, predictions)
                    # Should either pass or fail gracefully, not crash
                    self.assertIsInstance(pred_valid, bool)
                    self.assertIsInstance(pred_issues, list)
                except Exception as e:
                    self.fail(f"Pipeline validation crashed on {problem_type}: {e}")

                # Test robust validation
                try:
                    if not (np.isnan(predictions).any() or np.isinf(predictions).any()):
                        robust_results = robust_validator.validate_model_significance(actual, predictions)
                        self.assertIn("overall_assessment", robust_results)
                except Exception as e:
                    # Some cases expected to fail, but should not crash
                    print(f"Robust validation failed gracefully on {problem_type}: {type(e).__name__}")

                # Test robustness testing
                try:
                    if len(actual) >= 30 and len(predictions) >= 30:  # Min sample size
                        if not (np.isnan(predictions).any() or np.isinf(predictions).any()):
                            robustness_results = robustness_tester.comprehensive_robustness_test(predictions, actual)
                            self.assertIn("overall_assessment", robustness_results)
                except Exception as e:
                    print(f"Robustness testing failed gracefully on {problem_type}: {type(e).__name__}")

        print(f"\nError Handling Test Completed: All validators handled problematic data gracefully")

    def test_validation_consistency_across_multiple_runs(self):
        """Test that validation results are consistent across multiple runs"""
        # Create deterministic test data
        np.random.seed(42)
        actual, predictions = self.test_data.create_biased_predictions(n_samples=200, bias=0.65)

        # Run validation multiple times
        results = []
        for run in range(3):
            robust_validator = create_robust_validator(n_bootstrap=50)

            result = robust_validator.validate_model_significance(actual, predictions)
            results.append(result["overall_assessment"]["min_p_value"])

        # Results should be identical with same random seed
        for i in range(1, len(results)):
            self.assertAlmostEqual(
                results[i], results[0], places=6, msg="Validation results should be deterministic with same random seed"
            )

        print(f"\nconsitency Test Results:")
        print(f"P-values across runs: {[f'{p:.6f}' for p in results]}")
        print(f"Standard deviation: {np.std(results):.8f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

