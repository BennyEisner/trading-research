#!/usr/bin/env python3

"""
Robust Time Series Validation Framework
Implements moving block bootstrap and conditional permutation testing
"""

import warnings
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class RobustTimeSeriesValidator:
    """
    Robust time-series validation using moving block bootstrap that preserves swing trading dependencies while testing for statistical significance
    """

    def __init__(self, block_size: int = 5, n_bootstrap: int = 1000, random_state: int = 42):

        self.block_size = block_size
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        np.random.seed(random_state)
        self.validation_results = {}

    def moving_block_bootstrap(
        self, returns: np.ndarray, predictions: np.ndarray, metric_func: Callable = None
    ) -> Dict:
        """Moving block bootstrap preserving swing trading dependencies"""

        if metric_func is None:
            metric_func = self._directional_accuracy

        actual_performance = metric_func(returns, predictions)

        bootstrap_performances = []
        n_samples = len(returns)

        for _ in range(self.n_bootstrap):

            bootstrap_returns = self._create_moving_blocks(returns, self.block_size)
            bootstrap_predictions = self._create_moving_blocks(predictions, self.block_size)

            # Ensure same length as original
            bootstrap_returns = bootstrap_returns[:n_samples]
            bootstrap_predictions = bootstrap_predictions[:n_samples]

            # Calculate performance on bootstrap sample
            bootstrap_perf = metric_func(bootstrap_returns, bootstrap_predictions)
            bootstrap_performances.append(bootstrap_perf)

        bootstrap_performances = np.array(bootstrap_performances)

        # Calculate p-value ( actual > bootstrap)
        p_value = np.mean(bootstrap_performances >= actual_performance)

        # Calculate confidence intervals
        ci_lower, ci_upper = np.percentile(bootstrap_performances, [2.5, 97.5])

        results = {
            "actual_performance": actual_performance,
            "bootstrap_mean": np.mean(bootstrap_performances),
            "bootstrap_std": np.std(bootstrap_performances),
            "p_value": p_value,
            "confidence_interval": (ci_lower, ci_upper),
            "significant": p_value < 0.05,
            "bootstrap_distribution": bootstrap_performances,
            "test_statistic": (actual_performance - np.mean(bootstrap_performances)) / np.std(bootstrap_performances),
        }

        return results

    def conditional_permutation_test(
        self, returns: np.ndarray, predictions: np.ndarray, volatility_quintiles: int = 5, metric_func: Callable = None
    ) -> Dict:
        """Permute within volatility regimes to preserve market structure while breaking predictive relationships"""

        if metric_func is None:
            metric_func = self._directional_accuracy

        # Calculate vol quintiles
        vol_quintiles = pd.qcut(np.abs(returns), volatility_quintiles, labels=False, duplicates="drop")

        actual_performance = metric_func(returns, predictions)

        # Permutation test
        null_performances = []

        for _ in range(self.n_bootstrap):
            shuffled_returns = returns.copy()

            # Shuffle within each volatility quintile
            unique_quintiles = np.unique(vol_quintiles[~np.isnan(vol_quintiles)])
            for q in unique_quintiles:
                mask = vol_quintiles == q
                shuffled_returns[mask] = np.random.permutation(returns[mask])

            null_performance = metric_func(shuffled_returns, predictions)
            null_performances.append(null_performance)

        null_performances = np.array(null_performances)

        # Calculate pvalue
        p_value = np.mean(null_performances >= actual_performance)

        results = {
            "actual_performance": actual_performance,
            "null_mean": np.mean(null_performances),
            "null_std": np.std(null_performances),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "null_distribution": null_performances,
            "quintiles_used": len(unique_quintiles),
        }

        return results

    def validate_model_significance(
        self, returns: np.ndarray, predictions: np.ndarray, test_types: List[str] = None
    ) -> Dict:
        """Comprehensive significance testing of model predictions"""

        if test_types is None:
            test_types = ["bootstrap", "permutation"]

        results = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "n_samples": len(returns),
                "returns_mean": np.mean(returns),
                "returns_std": np.std(returns),
                "predictions_mean": np.mean(predictions),
                "predictions_std": np.std(predictions),
            },
        }

        # Moving block bootstrap test
        if "bootstrap" in test_types:
            print("Running moving block bootstrap test...")
            bootstrap_results = self.moving_block_bootstrap(returns, predictions)
            results["moving_block_bootstrap"] = bootstrap_results

            print(f"Bootstrap p-value: {bootstrap_results['p_value']:.4f}")
            print(f"Significant: {bootstrap_results['significant']}")

        # Conditional permutation test
        if "permutation" in test_types:
            print("Running conditional permutation test...")
            permutation_results = self.conditional_permutation_test(returns, predictions)
            results["conditional_permutation"] = permutation_results

            print(f"Permutation p-value: {permutation_results['p_value']:.4f}")
            print(f"Significant: {permutation_results['significant']}")

        # Combined assessment
        p_values = []
        if "bootstrap" in test_types:
            p_values.append(bootstrap_results["p_value"])
        if "permutation" in test_types:
            p_values.append(permutation_results["p_value"])

        # Overall significance all tests must be significant
        overall_significant = all(p < 0.05 for p in p_values)

        results["overall_assessment"] = {
            "all_p_values": p_values,
            "min_p_value": min(p_values) if p_values else 1.0,
            "max_p_value": max(p_values) if p_values else 1.0,
            "all_significant": overall_significant,
            "recommendation": "PASS" if overall_significant else "FAIL",
        }

        self.validation_results = results
        return results

    def _create_moving_blocks(self, data: np.ndarray, block_size: int) -> np.ndarray:
        """Create bootstrap sample using overlapping moving blocks"""

        n = len(data)
        if n < block_size:
            return data

        # Number of possible blocks
        n_blocks = n - block_size + 1

        # Calculate number of blocks needed
        n_blocks_needed = int(np.ceil(n / block_size))

        # Randomly select blocks with replacement
        selected_blocks = np.random.choice(n_blocks, size=n_blocks_needed, replace=True)

        # Concatenate selected blocks
        bootstrap_sample = []
        for block_start in selected_blocks:
            block = data[block_start : block_start + block_size]
            bootstrap_sample.extend(block)

        return np.array(bootstrap_sample)

    def _directional_accuracy(self, returns: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate directional accuracy metric"""
        return np.mean(np.sign(returns) == np.sign(predictions))

    def _sharpe_ratio(self, returns: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate Sharpe ratio of strategy returns"""
        # Simulate trading returns
        strategy_returns = np.where(predictions > 0, returns, np.where(predictions < 0, -returns, 0))

        if np.std(strategy_returns) == 0:
            return 0.0

        return np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)

    def print_validation_summary(self):
        """Show summary of validation results"""
        if not self.validation_results:
            print("No validation results available. Run validate_model_significance() first.")
            return

        results = self.validation_results

        print("\n" + "=" * 60)
        print("ROBUST TIME-SERIES VALIDATION SUMMARY")
        print("=" * 60)

        # Data summary
        data_sum = results["data_summary"]
        print(f"Data Summary:")
        print(f"Samples: {data_sum['n_samples']}")
        print(f"Returns: μ={data_sum['returns_mean']:.4f}, σ ={data_sum['returns_std']:.4f}")
        print(f"Predictions: μ={data_sum['predictions_mean']:.4f}, σ={data_sum['predictions_std']:.4f}")

        # Test results
        if "moving_block_bootstrap" in results:
            bootstrap = results["moving_block_bootstrap"]
            print(f"\nMoving Block Bootstrap:")
            print(f"Actual Performance: {bootstrap['actual_performance']:.4f}")
            print(f"Bootstrap Mean: {bootstrap['bootstrap_mean']:.4f}")
            print(f"P-value: {bootstrap['p_value']:.4f}")
            print(f"95% CI: [{bootstrap['confidence_interval'][0]:.4f}, {bootstrap['confidence_interval'][1]:.4f}]")
            print(f"Significant: {'YES' if bootstrap['significant'] else 'NO'}")

        if "conditional_permutation" in results:
            permutation = results["conditional_permutation"]
            print(f"\nConditional Permutation:")
            print(f"Actual Performance: {permutation['actual_performance']:.4f}")
            print(f"Null Mean: {permutation['null_mean']:.4f}")
            print(f"P-value: {permutation['p_value']:.4f}")
            print(f"Significant: {'YES' if permutation['significant'] else 'NO'}")

        # Overall assessment
        overall = results["overall_assessment"]
        print(f"\nOverall Assessment:")
        print(f"All P-values: {[f'{p:.4f}' for p in overall['all_p_values']]}")
        print(f"All Significant: {'YES' if overall['all_significant'] else 'NO'}")
        print(f"Recommendation: {overall['recommendation']}")

        print("=" * 60)


def create_robust_validator(block_size: int = 5, n_bootstrap: int = 1000, random_state: int = 42):
    """Helper function to create robust validator"""
    return RobustTimeSeriesValidator(block_size=block_size, n_bootstrap=n_bootstrap, random_state=random_state)

