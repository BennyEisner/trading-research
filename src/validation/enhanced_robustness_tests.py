#!/usr/bin/env python3

"""Testing suite for Autocorrelation preservation, regime transition testing, market stress testing"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf


class EnhancedRobustnessTests:
    """Comprehensive testing for time series models"""

    def __init__(self, significance_level: float = 0.05, eigenvalue_threshold: float = 0.1, min_sample_size: int = 30):
        self.significance_level = significance_level
        self.eigenvalue_threshold = eigenvalue_threshold
        self.min_sample_size = min_sample_size
        self.test_results = {}

    def autocorrelation_preservation_test(
        self, original_returns: np.ndarray, permuted_returns: np.ndarray, max_lags: int = 20, tolerance: float = 0.1
    ) -> Dict:
        """Tests to ensure permuted data maintains similar autocorrelation structure to original data"""

        # Input validation
        if len(original_returns) != len(permuted_returns):
            return {"acf_preserved": False, "error": "Array length mismatch", "validation": "ERROR"}

        if len(original_returns) < self.min_sample_size:
            return {
                "acf_preserved": False,
                "error": f"Insufficient samples: {len(original_returns)} < {self.min_sample_size}",
                "validation": "ERROR",
            }

        try:
            # Calculate ACF for both
            original_acf = acf(original_returns, nlags=max_lags, fft=True)
            permuted_acf = acf(permuted_returns, nlags=max_lags, fft=True)

            # Calculate similarity metric
            acf_differences = np.abs(original_acf - permuted_acf)
            avg_difference = np.mean(acf_differences[1:])
            max_difference = np.max(acf_differences[1:])

            # Test to ensure similar structures
            preservation_test = avg_difference < tolerance

            results = {
                "acf_preserved": preservation_test,
                "avg_acf_difference": avg_difference,
                "max_acf_difference": max_difference,
                "tolerance": tolerance,
                "original_acf": original_acf,
                "permuted_acf": permuted_acf,
                "acf_differences": acf_differences,
                "validation": "PASS" if preservation_test else "FAIL",
                "interpretation": self._interpret_acf_test(avg_difference, tolerance),
            }

        except Exception as e:
            results = {"acf_preserved": False, "error": str(e), "validation": "ERROR"}

        return results

    def regime_transition_testing(
        self,
        model_predictions: np.ndarray,
        actual_returns: np.ndarray,
        regime_labels: np.ndarray,
        max_degradation: float = 0.3,
    ) -> Dict:
        """Test model performance during regime transitions"""

        # Input validation
        if len(model_predictions) != len(actual_returns) or len(actual_returns) != len(regime_labels):
            return {
                "stable_accuracy": 0,
                "transition_accuracy": 0,
                "degradation_pct": 100,
                "acceptable_degradation": False,
                "validation": "ERROR",
                "error": "Array length mismatch",
            }

        if len(actual_returns) < self.min_sample_size:
            return {
                "stable_accuracy": 0,
                "transition_accuracy": 0,
                "degradation_pct": 100,
                "acceptable_degradation": False,
                "validation": "ERROR",
                "error": f"Insufficient samples: {len(actual_returns)} < {self.min_sample_size}",
            }

        # Separate Stable and transitional periods
        stable_mask = regime_labels == 0
        transition_mask = regime_labels == 1

        if np.sum(stable_mask) == 0 or np.sum(transition_mask) == 0:
            return {
                "stable_accuracy": 0,
                "transition_accuracy": 0,
                "degradation_pct": 100,
                "acceptable_degradation": False,
                "validation": "INSUFFICIENT_DATA",
                "error": "Need both stable and transition periods",
            }

        # Calculate accuracies
        stable_accuracy = self._directional_accuracy(actual_returns[stable_mask], model_predictions[stable_mask])

        transition_accuracy = self._directional_accuracy(
            actual_returns[transition_mask], model_predictions[transition_mask]
        )

        # Calculate degradation
        if stable_accuracy > 0:
            degradation = (stable_accuracy - transition_accuracy) / stable_accuracy
        else:
            degradation = 1.0

        acceptable = degradation <= max_degradation

        results = {
            "stable_accuracy": stable_accuracy,
            "transition_accuracy": transition_accuracy,
            "degradation_pct": degradation * 100,
            "max_degradation_pct": max_degradation * 100,
            "acceptable_degradation": acceptable,
            "stable_samples": np.sum(stable_mask),
            "transition_samples": np.sum(transition_mask),
            "validation": "PASS" if acceptable else "FAIL",
            "interpretation": self._interpret_regime_test(degradation, max_degradation),
        }

        return results

    def market_stress_testing(
        self,
        model_predictions: np.ndarray,
        actual_returns: np.ndarray,
        volatility_threshold: float = None,
        min_stress_accuracy: float = 0.45,
    ) -> Dict:
        """Test model performance during high stress market periods"""

        # Input validation
        if len(model_predictions) != len(actual_returns):
            return {
                "stress_accuracy": 0,
                "normal_accuracy": 0,
                "stress_acceptable": False,
                "validation": "ERROR",
                "error": "Array length mismatch",
            }

        if len(actual_returns) < self.min_sample_size:
            return {
                "stress_accuracy": 0,
                "normal_accuracy": 0,
                "stress_acceptable": False,
                "validation": "ERROR",
                "error": f"Insufficient samples: {len(actual_returns)} < {self.min_sample_size}",
            }

        volatility = pd.Series(actual_returns).rolling(20, min_periods=10).std().dropna()

        # Trim arrays to match volatility length
        n_vol = len(volatility)
        actual_returns_trimmed = actual_returns[-n_vol:]
        model_predictions_trimmed = model_predictions[-n_vol:]

        # Define stress threshold
        if volatility_threshold is None:
            volatility_threshold = np.percentile(volatility, 80)  # Top 20% vol periods

        stress_mask = volatility > volatility_threshold
        normal_mask = ~stress_mask

        if np.sum(stress_mask) == 0:
            return {
                "stress_accuracy": 0,
                "normal_accuracy": 0,
                "stress_acceptable": False,
                "validation": "NO_STRESS_PERIODS",
                "volatility_threshold": volatility_threshold,
            }

        # Calculate accuracies using trimmed arrays
        stress_accuracy = self._directional_accuracy(
            actual_returns_trimmed[stress_mask], model_predictions_trimmed[stress_mask]
        )

        normal_accuracy = self._directional_accuracy(
            actual_returns_trimmed[normal_mask], model_predictions_trimmed[normal_mask]
        )

        stress_acceptable = stress_accuracy >= min_stress_accuracy

        relative_performance = stress_accuracy / normal_accuracy if normal_accuracy > 0 else 0

        results = {
            "stress_accuracy": stress_accuracy * 100,
            "normal_accuracy": normal_accuracy * 100,
            "min_stress_accuracy": min_stress_accuracy * 100,
            "stress_acceptable": stress_acceptable,
            "relative_performance": relative_performance,
            "volatility_threshold": volatility_threshold,
            "stress_periods": np.sum(stress_mask),
            "normal_periods": np.sum(normal_mask),
            "validation": "PASS" if stress_acceptable else "FAIL",
            "interpretation": self._interpret_stress_test(stress_accuracy, min_stress_accuracy),
        }

        return results

    def multiple_testing_correction(self, p_values: List[float], method: str = "fdr_bh", alpha: float = 0.05) -> Dict:
        """Apply multiple testing correction"""

        if len(p_values) == 0:
            return {"error": "No p-values provided"}

        # Apply correction
        rejected, corrected_p, alpha_sidak, alpha_bonf = multipletests(p_values, alpha=alpha, method=method)

        results = {
            "original_p_values": p_values,
            "corrected_p_values": corrected_p.tolist(),
            "rejected": rejected.tolist(),
            "method": method,
            "alpha": alpha,
            "n_significant_original": sum(p < alpha for p in p_values),
            "n_significant_corrected": sum(rejected),
            "alpha_bonferroni": alpha_bonf,
            "alpha_sidak": alpha_sidak,  # Fixed: Include alpha_sidak in results
            "effective_alpha": alpha / len(p_values) if method == "bonferroni" else alpha,
        }
        return results

    def correlation_adjusted_testing(self, test_results: List[float], correlation_matrix: np.ndarray) -> Dict:
        """Adjust for correlation between tests"""

        try:
            # Calculate effective # of independent tests
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            effective_tests = np.sum(eigenvalues > self.eigenvalue_threshold)

            adjusted_alpha = self.significance_level / effective_tests

            results = {
                "original_n_tests": len(test_results),
                "effective_n_tests": effective_tests,
                "eigenvalues": eigenvalues.tolist(),
                "correlation_matrix": correlation_matrix.tolist(),
                "original_alpha": self.significance_level,
                "adjusted_alpha": adjusted_alpha,
                "adjustment_factor": effective_tests / len(test_results),
            }

        except Exception as e:
            results = {"error": str(e), "validation": "ERROR"}

        return results

    def comprehensive_robustness_test(
        self,
        model_predictions: np.ndarray,
        actual_returns: np.ndarray,
        permuted_returns: np.ndarray = None,
        regime_labels: np.ndarray = None,
    ) -> Dict:
        """Run testing suite"""

        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_samples": len(actual_returns),
        }

        # 1. Autocorrelation preservation test
        if permuted_returns is not None:
            print("Running autocorrelation preservation test...")
            acf_results = self.autocorrelation_preservation_test(actual_returns, permuted_returns)
            results["autocorrelation_test"] = acf_results

            if "validation" in acf_results:
                print(f"Result: {acf_results['validation']}")

        # 2. Regime transition test
        if regime_labels is not None:
            print("Running regime transition test...")
            regime_results = self.regime_transition_testing(model_predictions, actual_returns, regime_labels)

            results["regime_transition_test"] = regime_results
            print(f"Result: {regime_results['validation']}")

        # 3. Market stress test
        print("Running stress tests...")
        stress_results = self.market_stress_testing(model_predictions, actual_returns)

        results["market_stress_test"] = stress_results
        print(f"Result: {stress_results['validation']}")

        # Complete Assessment
        all_tests_passed = True
        test_summaries = []

        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and "validation" in test_result:
                passed = test_result["validation"] == "PASS"
                test_summaries.append(
                    {
                        "test": test_name,
                        "result": test_result["validation"],
                        "passed": passed,
                    }
                )
                if not passed:
                    all_tests_passed = False

        results["overall_assessment"] = {
            "all_tests_passed": all_tests_passed,
            "test_summaries": test_summaries,
            "recommendation": "PASS" if all_tests_passed else "INVESTIGATE",
        }

        self.test_results = results
        return results

    def _directional_accuracy(self, returns: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate directional accuracy with proper handling of zero returns"""
        if len(returns) == 0 or len(predictions) == 0:
            return 0.0

        # Handle edge case where model always predicts same sign
        unique_predictions = np.unique(np.sign(predictions))
        if len(unique_predictions) == 1 and unique_predictions[0] == 0:
            return 0.0  # Model predicts zero for everything

        if len(unique_predictions) == 1:
            actual_directions = np.sign(returns)
            return np.mean(actual_directions == unique_predictions[0])

        nonzero_mask = returns != 0
        if np.sum(nonzero_mask) == 0:
            return 0.5  # All returns are zero, accuracy is meaningless

        # Use only non-zero returns for accuracy calculation
        returns_filtered = returns[nonzero_mask]
        predictions_filtered = predictions[nonzero_mask]

        return np.mean(np.sign(returns_filtered) == np.sign(predictions_filtered))

    def _interpret_acf_test(self, avg_difference: float, tolerance: float) -> str:
        """Interpret ACF preservation test results"""
        if avg_difference < tolerance * 0.5:
            return "Great. Autocorrelation structure well preserved"
        elif avg_difference < tolerance:
            return "Good. Autocorrelation structure preserved okay"
        else:
            return f"Poor. Autocorrelation structure NOT preserved: diff={avg_difference:.3f})"

    def _interpret_regime_test(self, degradation: float, max_degradation: float) -> str:
        """Interpret regime transition test results"""
        if degradation < max_degradation * 0.5:
            return "Great. Minimal performance degradation during regime transitions"
        elif degradation < max_degradation:
            return "Good. Some performance degradation during regime transitions"
        else:
            return f"Poor. Excessive degradation during transitions: degradation={degradation*100:.1f})"

    def _interpret_stress_test(self, stress_accuracy: float, min_accuracy: float) -> str:
        """Interpret stress test results"""

        if stress_accuracy / 100 >= min_accuracy + 0.1:
            return "Excellent: Strong performance during market stress"
        elif stress_accuracy / 100 >= min_accuracy:
            return "Acceptable: Adequate performance during stress"
        else:
            return f"Poor: Insufficient performance during stress ({stress_accuracy/100:.3f})"

    def print_robustness_summary(self):
        """Print comprehensive robustness test summary"""
        if not self.test_results:
            print("No test results available. Run comprehensive_robustness_test() first.")
            return

        results = self.test_results

        print("\n" + "=" * 70)
        print("ENHANCED ROBUSTNESS TEST SUMMARY")
        print("=" * 70)

        print(f"Samples tested: {results['n_samples']}")
        print(f"Test timestamp: {results['timestamp']}")

        # Individual test results
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and "validation" in test_result:
                status = "PASSED" if test_result["validation"] == "PASS" else "FAILED"
                print(f"\n{status} {test_name.replace('_', ' ').title()}:")

                if "interpretation" in test_result:
                    print(f"  {test_result['interpretation']}")

        # Overall assessment
        if "overall_assessment" in results:
            overall = results["overall_assessment"]
            print(f"\nOverall Assessment: {overall['recommendation']}")
            print(f"All tests passed: {'YES' if overall['all_tests_passed'] else 'NO'}")

        print("=" * 70)


def create_robustness_tester(
    significance_level: float = 0.05, eigenvalue_threshold: float = 0.1, min_sample_size: int = 30
):
    """Helper function to create robustness tester"""
    return EnhancedRobustnessTests(
        significance_level=significance_level,
        eigenvalue_threshold=eigenvalue_threshold,
        min_sample_size=min_sample_size,
    )

