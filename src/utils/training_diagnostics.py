#!/usr/bin/env python3

"""Comprehensive checks for data quality, target construction, and model behavior"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TrainingDiagnostics:
    """
    Comprehensive diagnostics for ML training pipeline
    """

    def __init__(self):
        self.diagnostics_results = {}

    def check_target_construction(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Verify targets are properly constructed with correct time shifts

        Args:
            ticker_data: Dict mapping ticker -> DataFrame with features and targets

        Returns:
            Dict with target construction analysis
        """
        print(f"\nTARGET CONSTRUCTION DIAGNOSTICS:")
        print("=" * 60)

        results = {"status": "pass", "issues": [], "stats": {}}

        for ticker, data in ticker_data.items():
            print(f"\n{ticker} Target Analysis:")

            if "daily_return" not in data.columns:
                results["issues"].append(f"{ticker}: Missing daily_return column")
                continue

            returns = data["daily_return"]

            # Basic statistics
            stats = {
                "count": len(returns),
                "mean": np.mean(returns),
                "std": np.std(returns),
                "min": np.min(returns),
                "max": np.max(returns),
                "nan_count": np.isnan(returns).sum(),
            }

            results["stats"][ticker] = stats

            print(f"Count: {stats['count']:,}")
            print(f"Mean: {stats['mean']:.6f}")
            print(f"Std:  {stats['std']:.6f}")
            print(f"Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"NaN Count: {stats['nan_count']}")

            # Check for suspicious patterns
            if abs(stats["mean"]) > 0.01:
                results["issues"].append(f"{ticker}: Mean return {stats['mean']:.6f} seems high for daily returns")
                print(f"WARNING: Mean return {stats['mean']:.6f} is unusually high")

            if stats["std"] > 0.1:
                results["issues"].append(f"{ticker}: Volatility {stats['std']:.6f} seems high")
                print(f"WARNING: High volatility {stats['std']:.6f}")

            if stats["nan_count"] > 0:
                results["issues"].append(f"{ticker}: {stats['nan_count']} NaN values in targets")
                print(f"WARNING: {stats['nan_count']} NaN values in targets")

            # Check for data leakage patterns
            if "close" in data.columns:
                close_prices = data["close"]
                manual_returns = close_prices.pct_change()

                # Compare with existing daily_return
                correlation = np.corrcoef(returns[1:], manual_returns[1:])[0, 1]
                if correlation < 0.99:
                    results["issues"].append(
                        f"{ticker}: daily_return doesn't match price changes (corr={correlation:.3f})"
                    )
                    print(f" WARNING: daily_return correlation with price changes: {correlation:.3f}")
                else:
                    print(f"daily_return properly calculated from prices (corr={correlation:.3f})")

        # Cross-ticker consistency
        print(f"\n🔍 CROSS-TICKER CONSISTENCY:")
        means = [results["stats"][ticker]["mean"] for ticker in results["stats"]]
        stds = [results["stats"][ticker]["std"] for ticker in results["stats"]]

        print(f"   Mean returns range: [{min(means):.6f}, {max(means):.6f}]")
        print(f"   Volatility range: [{min(stds):.6f}, {max(stds):.6f}]")

        if max(means) - min(means) > 0.005:
            results["issues"].append("Large differences in mean returns across tickers")
            print(f" WARNING: Large spread in mean returns across tickers")

        if len(results["issues"]) == 0:
            print(f"All targets look consistent and properly constructed")

        return results

    def check_feature_scaling(self, X_train: np.ndarray, feature_names: List[str] = None) -> Dict:
        """
        Verify feature scaling is working correctly

        Args:
            X_train: Training features array (samples, timesteps, features)
            feature_names: Optional list of feature names

        Returns:
            Dict with feature scaling analysis
        """
        print(f"\nFEATURE SCALING DIAGNOSTICS:")
        print("=" * 60)

        results = {"status": "pass", "issues": [], "feature_stats": {}}

        if len(X_train.shape) != 3:
            results["issues"].append(f"Expected 3D array, got shape {X_train.shape}")
            return results

        n_samples, n_timesteps, n_features = X_train.shape
        print(f"   Shape: {n_samples:,} samples × {n_timesteps} timesteps × {n_features} features")

        # Check each feature across all samples and timesteps
        for feature_idx in range(n_features):
            feature_name = (
                feature_names[feature_idx]
                if feature_names and feature_idx < len(feature_names)
                else f"feature_{feature_idx}"
            )

            # Get all values for this feature across samples and timesteps
            feature_values = X_train[:, :, feature_idx].flatten()

            stats = {
                "mean": np.mean(feature_values),
                "std": np.std(feature_values),
                "min": np.min(feature_values),
                "max": np.max(feature_values),
                "nan_count": np.isnan(feature_values).sum(),
            }

            results["feature_stats"][feature_name] = stats

            # Check for scaling issues
            if abs(stats["mean"]) > 2.0:
                results["issues"].append(f"{feature_name}: Mean {stats['mean']:.3f} suggests poor centering")

            if stats["std"] > 5.0 or stats["std"] < 0.1:
                results["issues"].append(f"{feature_name}: Std {stats['std']:.3f} suggests poor scaling")

            if abs(stats["min"]) > 10.0 or abs(stats["max"]) > 10.0:
                results["issues"].append(
                    f"{feature_name}: Range [{stats['min']:.3f}, {stats['max']:.3f}] suggests poor normalization"
                )

            if stats["nan_count"] > 0:
                results["issues"].append(f"{feature_name}: {stats['nan_count']} NaN values")

        # Summary statistics
        all_means = [stats["mean"] for stats in results["feature_stats"].values()]
        all_stds = [stats["std"] for stats in results["feature_stats"].values()]
        all_mins = [stats["min"] for stats in results["feature_stats"].values()]
        all_maxs = [stats["max"] for stats in results["feature_stats"].values()]

        print(f"\n SCALING SUMMARY:")
        print(f"Mean range: [{min(all_means):.3f}, {max(all_means):.3f}]")
        print(f"Std range: [{min(all_stds):.3f}, {max(all_stds):.3f}]")
        print(f"Min range: [{min(all_mins):.3f}, {max(all_mins):.3f}]")
        print(f"Max range: [{min(all_maxs):.3f}, {max(all_maxs):.3f}]")

        # Check if features look standardized
        if max(all_means) - min(all_means) < 0.1 and max(all_stds) - min(all_stds) < 0.2:
            print(f"Features appear properly standardized")
        else:
            results["issues"].append("Features don't appear to be consistently standardized")
            print(f" WARNING: Features may not be properly standardized")

        return results

    def analyze_prediction_behavior(self, y_true: np.ndarray, y_pred: np.ndarray, split_name: str = "Unknown") -> Dict:
        """
        Analyze model prediction behavior for signs of issues

        Args:
            y_true: True target values
            y_pred: Predicted values
            split_name: Name of the data split (train/val/test)

        Returns:
            Dict with prediction analysis
        """
        print(f"\nPREDICTION BEHAVIOR ANALYSIS - {split_name.upper()}:")
        print("=" * 60)

        results = {"split": split_name, "status": "pass", "issues": [], "metrics": {}}

        # Basic statistics
        true_stats = {"mean": np.mean(y_true), "std": np.std(y_true), "min": np.min(y_true), "max": np.max(y_true)}

        pred_stats = {"mean": np.mean(y_pred), "std": np.std(y_pred), "min": np.min(y_pred), "max": np.max(y_pred)}

        results["metrics"]["true"] = true_stats
        results["metrics"]["pred"] = pred_stats

        print(f"   TRUE VALUES:")
        print(f"     Mean: {true_stats['mean']:.6f}, Std: {true_stats['std']:.6f}")
        print(f"     Range: [{true_stats['min']:.6f}, {true_stats['max']:.6f}]")

        print(f"   PREDICTIONS:")
        print(f"     Mean: {pred_stats['mean']:.6f}, Std: {pred_stats['std']:.6f}")
        print(f"     Range: [{pred_stats['min']:.6f}, {pred_stats['max']:.6f}]")

        # Check for problematic prediction patterns
        if pred_stats["std"] < true_stats["std"] * 0.1:
            results["issues"].append("Model predicting near-constant values (very low variance)")
            print(f" WARNING: Model predictions have very low variance ({pred_stats['std']:.6f})")

        if pred_stats["std"] > true_stats["std"] * 3.0:
            results["issues"].append("Model predictions have excessive variance")
            print(f"WARNING: Model predictions have excessive variance ({pred_stats['std']:.6f})")

        if abs(pred_stats["mean"] - true_stats["mean"]) > abs(true_stats["mean"]) * 2:
            results["issues"].append("Model predictions biased away from true mean")
            print(
                f"WARNING: Prediction mean ({pred_stats['mean']:.6f}) differs significantly from true mean ({true_stats['mean']:.6f})"
            )

        # Calculate correlation
        try:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            results["metrics"]["correlation"] = correlation
            print(f"   Correlation: {correlation:.6f}")

            if np.isnan(correlation):
                results["issues"].append("Correlation is NaN (likely constant predictions)")
                print(f" WARNING: Correlation is NaN")
            elif correlation < 0.01:
                results["issues"].append(f"Very low correlation ({correlation:.6f})")
                print(f" WARNING: Very low correlation ({correlation:.6f})")
        except:
            results["issues"].append("Could not calculate correlation")
            results["metrics"]["correlation"] = None

        # Directional accuracy
        direction_correct = np.sum(np.sign(y_true) == np.sign(y_pred))
        direction_accuracy = direction_correct / len(y_true)
        results["metrics"]["directional_accuracy"] = direction_accuracy

        print(f"   Directional Accuracy: {direction_accuracy:.3f} ({direction_correct}/{len(y_true)})")

        if direction_accuracy < 0.52:
            results["issues"].append(f"Directional accuracy ({direction_accuracy:.3f}) barely better than random")
            print(f" WARNING: Directional accuracy barely better than random")

        if len(results["issues"]) == 0:
            print(f"Prediction behavior looks reasonable")

        return results

    def comprehensive_training_check(
        self,
        ticker_data: Dict[str, pd.DataFrame],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str] = None,
    ) -> Dict:
        """
        Run all diagnostic checks before training

        Returns:
            Dict with comprehensive diagnostic results
        """
        print(f"\nCOMPREHENSIVE TRAINING DIAGNOSTICS:")
        print("=" * 80)

        results = {
            "target_construction": self.check_target_construction(ticker_data),
            "feature_scaling": self.check_feature_scaling(X_train, feature_names),
            "timestamp": pd.Timestamp.now(),
        }

        # Summary
        all_issues = []
        for check_name, check_results in results.items():
            if isinstance(check_results, dict) and "issues" in check_results:
                all_issues.extend(check_results["issues"])

        print(f"\nDIAGNOSTIC SUMMARY:")
        print("=" * 50)
        if len(all_issues) == 0:
            print("ALL CHECKS PASSED - Ready for training")
            results["overall_status"] = "pass"
        else:
            print(f"{len(all_issues)} ISSUES FOUND:")
            for i, issue in enumerate(all_issues, 1):
                print(f"   {i}. {issue}")
            results["overall_status"] = "issues_found"

        results["total_issues"] = len(all_issues)
        self.diagnostics_results = results

        return results

    def save_diagnostics_report(self, filepath: str = "training_diagnostics_report.txt"):
        """
        Save diagnostics results to file

        Args:
            filepath: Path to save the report
        """
        if not self.diagnostics_results:
            print("No diagnostics results to save")
            return

        with open(filepath, "w") as f:
            f.write("TRAINING DIAGNOSTICS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {self.diagnostics_results['timestamp']}\n\n")

            f.write(f"Overall Status: {self.diagnostics_results['overall_status']}\n")
            f.write(f"Total Issues: {self.diagnostics_results['total_issues']}\n\n")

            # Add detailed results
            for check_name, results in self.diagnostics_results.items():
                if isinstance(results, dict) and "issues" in results:
                    f.write(f"\n{check_name.upper()} RESULTS:\n")
                    f.write("-" * 30 + "\n")
                    if results["issues"]:
                        for issue in results["issues"]:
                            f.write(f"  - {issue}\n")
                    else:
                        f.write("No issues found\n")

        print(f"Diagnostics report saved to: {filepath}")

