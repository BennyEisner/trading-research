#!/usr/bin/env python3

"""
Comprehensive pipeline validation and data quality checks
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class PipelineValidator:
    """Comprehensive validation for ML pipeline components"""

    def __init__(
        self,
        logger=None,
        extreme_price_threshold: float = 0.5,
        outlier_std_threshold: float = 6.0,
        correlation_threshold: float = 0.95,
        min_directional_accuracy: float = 40.0,
        high_nan_threshold: float = 0.1,
        gap_multiplier: float = 7.0,
    ):
        self.logger = logger
        self.validation_results = {}

        # Configurable thresholds
        self.extreme_price_threshold = extreme_price_threshold
        self.outlier_std_threshold = outlier_std_threshold
        self.correlation_threshold = correlation_threshold
        self.min_directional_accuracy = min_directional_accuracy
        self.high_nan_threshold = high_nan_threshold
        self.gap_multiplier = gap_multiplier

    def log(self, message, level="INFO"):
        """Log message with fallback to print"""
        if self.logger:
            if hasattr(self.logger, "info") and level == "INFO":
                self.logger.info(message)
            elif hasattr(self.logger, "warning") and level == "WARNING":
                self.logger.warning(message)
            elif hasattr(self.logger, "error") and level == "ERROR":
                self.logger.error(message)
            else:
                self.logger.log(message)
        else:
            print(f"[{level}] {message}")

    def validate_raw_data(self, data: pd.DataFrame, ticker: str) -> Tuple[bool, List[str]]:
        """Validate raw OHLCV data quality"""
        if data.empty:
            return False, ["Empty DataFrame provided"]

        issues = []

        # Work with a copy to avoid mutating input data
        data_copy = data.copy()

        # Check required columns
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data_copy.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

        if issues:  # Cant continue without basic columns
            return False, issues

        # Check data types - work with copy
        try:
            data_copy["date"] = pd.to_datetime(data_copy["date"])
        except Exception as e:
            issues.append(f"Invalid date format: {e}")

        # Check for OHLC consistency
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if data_copy[col].dtype not in ["float64", "int64"]:
                try:
                    data_copy[col] = pd.to_numeric(data_copy[col], errors="coerce")
                except Exception as e:
                    issues.append(f"Cannot convert {col} to numeric: {e}")

        # Validate OHLC relationships
        invalid_ohlc = (
            (data_copy["high"] < data_copy["low"])
            | (data_copy["high"] < data_copy["open"])
            | (data_copy["high"] < data_copy["close"])
            | (data_copy["low"] > data_copy["open"])
            | (data_copy["low"] > data_copy["close"])
        )

        if invalid_ohlc.sum() > 0:
            issues.append(f"Invalid OHLC relationships in {invalid_ohlc.sum()} records")

        # Check for negative prices or volumes
        for col in price_cols + ["volume"]:
            if (data_copy[col] <= 0).sum() > 0:
                negative_count = (data_copy[col] <= 0).sum()
                issues.append(f"Non-positive values in {col}: {negative_count} records")

        # Check for extreme price movements
        if len(data_copy) > 1:
            price_changes = data_copy["close"].pct_change().abs()
            # Filter out NaN values from pct_change
            valid_changes = price_changes.dropna()
            if len(valid_changes) > 0:
                extreme_changes = (valid_changes > self.extreme_price_threshold).sum()
                if extreme_changes > 0:
                    issues.append(
                        f"Extreme price movements (>{self.extreme_price_threshold*100}%): {extreme_changes} occurrences"
                    )

        # Check data continuity (gaps in time series)
        if len(data_copy) > 1:
            data_sorted = data_copy.sort_values("date")
            date_diffs = data_sorted["date"].diff()
            # Remove first NaT value from diff calculation
            valid_diffs = date_diffs.dropna()

            if len(valid_diffs) > 0:
                median_gap = valid_diffs.median()
                if pd.notna(median_gap):
                    large_gaps = (valid_diffs > median_gap * self.gap_multiplier).sum()
                    if large_gaps > 0:
                        issues.append(f"Large time gaps detected: {large_gaps} instances")

        # Data quality summary
        total_records = len(data_copy)
        date_range = f"{data_copy['date'].min()} to {data_copy['date'].max()}"

        self.validation_results[f"{ticker}_raw_data"] = {
            "total_records": total_records,
            "date_range": date_range,
            "issues": issues,
            "is_valid": len(issues) == 0,
        }

        if issues:
            self.log(f"Data quality issues for {ticker}: {'; '.join(issues)}", "WARNING")
        else:
            self.log(f"Raw data validation passed for {ticker}: {total_records} records")

        return len(issues) == 0, issues

    def validate_feature_data(self, data: pd.DataFrame, feature_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate engineered features"""
        if data.empty:
            return False, ["Empty DataFrame provided"]

        issues = []

        if not feature_columns:
            issues.append("No features provided for validation")
            return False, issues

        # Validate that all feature columns exist
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            issues.append(f"Missing feature columns: {missing_features}")
            return False, issues

        feature_data = data[feature_columns].copy()

        # Check for NaN values
        nan_counts = feature_data.isna().sum()
        high_nan_features = nan_counts[nan_counts > len(data) * self.high_nan_threshold].index.tolist()
        if high_nan_features:
            issues.append(f"High NaN counts (>{self.high_nan_threshold*100}%) in features: {high_nan_features}")

        # Check for infinite values
        inf_counts = {}
        for col in feature_columns:
            if np.isinf(feature_data[col]).sum() > 0:
                inf_counts[col] = np.isinf(feature_data[col]).sum()

        if inf_counts:
            issues.append(f"Infinite values detected: {inf_counts}")

        # Check for constant features zero variance)
        constant_features = []
        for col in feature_columns:
            if feature_data[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            issues.append(f"Constant features (zero variance): {constant_features}")

        # Check for extreme outliers
        outlier_features = {}
        for col in feature_columns:
            if feature_data[col].dtype in ["float64", "int64"]:
                col_data = feature_data[col].dropna()
                if len(col_data) > 0 and col_data.std() > 0:
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    extreme_outliers = (z_scores > self.outlier_std_threshold).sum()
                    if extreme_outliers > 0:
                        outlier_features[col] = extreme_outliers

        if outlier_features:
            issues.append(f"Extreme outliers detected: {outlier_features}")

        # Check feature correlation (highly correlated features)
        try:
            # Only compute correlations for numeric columns with sufficient variance
            numeric_features = []
            for col in feature_columns:
                if feature_data[col].dtype in ["float64", "int64"]:
                    col_data = feature_data[col].dropna()
                    if len(col_data) > 1 and col_data.std() > 1e-10:
                        numeric_features.append(col)

            if len(numeric_features) > 1:
                correlation_matrix = feature_data[numeric_features].corr()
                high_correlations = []

                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if pd.notna(corr_value) and abs(corr_value) > self.correlation_threshold:
                            high_correlations.append(
                                (correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value)
                            )

                if high_correlations:
                    issues.append(
                        f"Highly correlated features (>{self.correlation_threshold}): {len(high_correlations)} pairs"
                    )

        except Exception as e:
            issues.append(f"Could not compute correlations: {e}")

        # Feature distributionnalysis
        skewed_features = []
        for col in feature_columns:
            if feature_data[col].dtype in ["float64", "int64"]:
                col_data = feature_data[col].dropna()
                if len(col_data) > 3:  # Need at least 3 points for skewness
                    try:
                        skewness = col_data.skew()
                        if pd.notna(skewness) and abs(skewness) > 3:  # Highly skewed
                            skewed_features.append(col)
                    except:
                        continue

        if len(skewed_features) > len(feature_columns) * 0.5:
            issues.append(f"Many highly skewed features: {len(skewed_features)}/{len(feature_columns)}")

        self.validation_results["feature_validation"] = {
            "total_features": len(feature_columns),
            "nan_features": len(high_nan_features),
            "infinite_features": len(inf_counts),
            "constant_features": len(constant_features),
            "outlier_features": len(outlier_features),
            "skewed_features": len(skewed_features),
            "issues": issues,
            "is_valid": len(issues) == 0,
        }

        if issues:
            self.log(f"Feature validation issues: {'; '.join(issues)}", "WARNING")
        else:
            self.log(f"Feature validation passed: {len(feature_columns)} features")

        return len(issues) == 0, issues

    def validate_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate LSTM sequences"""
        issues = []

        # Basic input validation
        if X.size == 0 or y.size == 0:
            return False, ["Empty arrays provided"]

        # Basic shape validation
        if len(X.shape) != 3:
            issues.append(f"Expected 3D sequences, got shape {X.shape}")

        if len(y.shape) != 1:
            issues.append(f"Expected 1D targets, got shape {y.shape}")

        if len(X) != len(y):
            issues.append(f"Sequence count mismatch: X={len(X)}, y={len(y)}")

        if len(issues) > 0:  # Cant continue with shape issues
            return False, issues

        # Check for NaN values
        nan_sequences = np.isnan(X).any(axis=(1, 2)).sum()
        nan_targets = np.isnan(y).sum()

        if nan_sequences > 0:
            issues.append(f"NaN values in {nan_sequences} sequences")

        if nan_targets > 0:
            issues.append(f"NaN values in {nan_targets} targets")

        # Check for infinite values
        inf_sequences = np.isinf(X).any(axis=(1, 2)).sum()
        inf_targets = np.isinf(y).sum()

        if inf_sequences > 0:
            issues.append(f"Infinite values in {inf_sequences} sequences")

        if inf_targets > 0:
            issues.append(f"Infinite values in {inf_targets} targets")

        # Check sequence diversity - corrected logic
        sequence_stds = np.std(X, axis=1)

        # Check sequences where ALL features have zero variance
        all_features_zero_var = (sequence_stds == 0).all(axis=1).sum()

        # Check sequences where ANY feature has zero variance
        any_feature_zero_var = (sequence_stds == 0).any(axis=1).sum()

        if all_features_zero_var > len(X) * 0.1:  # More than 10% of sequences have zero variance in ALL features
            issues.append(f"High number of sequences with zero variance in all features: {all_features_zero_var}")
        elif any_feature_zero_var == len(X):  # Every sequence has at least one zero-variance feature
            self.log(
                "Note: All sequences have some zero-variance features (may be expected for categorical features)",
                "INFO",
            )

        # Check target distribution
        target_std = np.std(y)
        if target_std == 0:
            issues.append("All targets are identical (zero variance)")
        elif target_std < 1e-6:
            issues.append(f"Very low target variance: {target_std}")

        if len(y) > 2:
            try:
                # Adaptive extreme detection based on target range
                target_min, target_max = np.min(y), np.max(y)
                target_range = target_max - target_min
                
                # For pattern confidence scores (0-1 range), use different logic
                if target_min >= 0 and target_max <= 1 and target_range > 0.3:
                    # Pattern confidence targets - check for values outside [0, 1]
                    extreme_targets = ((y < 0) | (y > 1)).sum()
                else:
                    # Return prediction targets - use original percentile logic
                    target_percentiles = np.percentile(y, [1, 99])
                    if target_percentiles[0] != 0:
                        extreme_low = (y < target_percentiles[0] * 10).sum()
                    else:
                        extreme_low = 0

                    if target_percentiles[1] != 0:
                        extreme_high = (y > target_percentiles[1] * 10).sum()
                    else:
                        extreme_high = 0

                    extreme_targets = extreme_low + extreme_high

                if extreme_targets > 0:
                    issues.append(f"Extreme target values detected: {extreme_targets}")
            except:
                # Skip extreme target check if percentile calculation fails
                pass

        self.validation_results["sequence_validation"] = {
            "sequence_shape": X.shape,
            "target_shape": y.shape,
            "nan_sequences": nan_sequences,
            "nan_targets": nan_targets,
            "all_zero_variance_sequences": all_features_zero_var,
            "any_zero_variance_sequences": any_feature_zero_var,
            "target_std": target_std,
            "issues": issues,
            "is_valid": len(issues) == 0,
        }

        if issues:
            self.log(f"Sequence validation issues: {'; '.join(issues)}", "WARNING")
        else:
            self.log(f"Sequence validation passed: {X.shape}")

        return len(issues) == 0, issues

    def validate_model_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate model predictions"""
        issues = []

        # Basic input validation
        if y_true.size == 0 or y_pred.size == 0:
            return False, ["Empty prediction arrays provided"]

        # Basic validation
        if len(y_true) != len(y_pred):
            issues.append(f"Prediction count mismatch: true={len(y_true)}, pred={len(y_pred)}")
            return False, issues

        # Check for nan or infinite predictions
        nan_predictions = np.isnan(y_pred).sum()
        inf_predictions = np.isinf(y_pred).sum()

        if nan_predictions > 0:
            issues.append(f"NaN predictions: {nan_predictions}")

        if inf_predictions > 0:
            issues.append(f"Infinite predictions: {inf_predictions}")

        # Check prediction distribution
        pred_std = np.std(y_pred)
        if pred_std == 0:
            issues.append("All predictions are identical")
        elif pred_std < 1e-6:
            issues.append(f"Very low prediction variance: {pred_std}")

        # Check for extreme predictions
        try:
            true_range = np.max(y_true) - np.min(y_true)
            pred_range = np.max(y_pred) - np.min(y_pred)

            if true_range > 0 and pred_range > true_range * 10:
                issues.append(f"Prediction range much larger than true range: {pred_range:.4f} vs {true_range:.4f}")
        except:
            # Skip range check if calculation fails
            pass

        # Statistical validation
        try:
            # Directional accuracy - handle zero returns properly
            y_true_nonzero = y_true[y_true != 0]
            y_pred_nonzero = y_pred[y_true != 0]

            if len(y_true_nonzero) > 0:
                dir_acc = np.mean(np.sign(y_true_nonzero) == np.sign(y_pred_nonzero)) * 100
            else:
                dir_acc = 50.0  # Default if all returns are zero

            # Correlation (handle edge cases)
            correlation = 0.0
            if len(y_true) > 1:
                true_std = np.std(y_true)
                pred_std = np.std(y_pred)
                if true_std > 1e-10 and pred_std > 1e-10:
                    try:
                        corr_matrix = np.corrcoef(y_true, y_pred)
                        if not np.isnan(corr_matrix).any():
                            correlation = corr_matrix[0, 1]
                    except:
                        correlation = 0.0

            # Mean Absolute Error
            mae = np.mean(np.abs(y_true - y_pred))

            if dir_acc < self.min_directional_accuracy:
                issues.append(f"Very low directional accuracy: {dir_acc:.1f}%")

            if abs(correlation) < 0.1:
                issues.append(f"Very low correlation: {correlation:.3f}")

            self.validation_results["prediction_validation"] = {
                "directional_accuracy": dir_acc,
                "correlation": correlation,
                "mae": mae,
                "nan_predictions": nan_predictions,
                "inf_predictions": inf_predictions,
                "prediction_std": pred_std,
                "issues": issues,
                "is_valid": len(issues) == 0,
            }

        except Exception as e:
            issues.append(f"Could not compute prediction metrics: {e}")

        if issues:
            self.log(f"Prediction validation issues: {'; '.join(issues)}", "WARNING")
        else:
            self.log(f"Prediction validation passed")

        return len(issues) == 0, issues

    def validate_training_stability(self, history: Dict) -> Tuple[bool, List[str]]:
        """Validate training stability from history"""
        issues = []

        if not history or "loss" not in history:
            issues.append("No loss history available")
            return False, issues

        losses = history["loss"]

        if not losses or len(losses) == 0:
            issues.append("Empty loss history")
            return False, issues

        # Check for exploding loss
        if len(losses) > 1:
            # Filter out NaN losses for calculations
            valid_losses = [loss for loss in losses if not np.isnan(loss)]

            if len(valid_losses) > 1:
                initial_loss = valid_losses[0]
                final_loss = valid_losses[-1]
                max_loss = max(valid_losses)

                if initial_loss > 0 and final_loss > initial_loss * 5:
                    issues.append(f"Loss explosion: {initial_loss:.4f} → {final_loss:.4f}")

                if initial_loss > 0 and max_loss > initial_loss * 10:
                    issues.append(f"Loss spike detected: max={max_loss:.4f}")

        # Check for NaN losses
        nan_losses = sum(1 for loss in losses if np.isnan(loss))
        if nan_losses > 0:
            issues.append(f"NaN losses encountered: {nan_losses} epochs")

        # Check convergence
        if len(losses) >= 10:
            recent_losses = [loss for loss in losses[-10:] if not np.isnan(loss)]

            if len(recent_losses) > 1:
                loss_std = np.std(recent_losses)
                loss_mean = np.mean(recent_losses)

                if loss_mean > 0 and loss_std > loss_mean * 0.5:
                    issues.append("Training appears unstable (high loss variance)")

        # Validation loss comparison
        if "val_loss" in history:
            val_losses = history["val_loss"]
            if len(val_losses) == len(losses):
                train_val_gaps = []

                for train_loss, val_loss in zip(losses, val_losses):
                    if not (np.isnan(train_loss) or np.isnan(val_loss)):
                        train_val_gaps.append(val_loss - train_loss)

                if len(train_val_gaps) > 0:
                    avg_gap = np.mean(train_val_gaps)
                    avg_train_loss = np.mean([loss for loss in losses if not np.isnan(loss)])

                    if avg_train_loss > 0 and avg_gap > avg_train_loss * 2:
                        issues.append(f"Large train/validation gap indicates overfitting: {avg_gap:.4f}")

        self.validation_results["training_validation"] = {
            "epochs_completed": len(losses),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "nan_losses": nan_losses,
            "issues": issues,
            "is_valid": len(issues) == 0,
        }

        if issues:
            self.log(f"Training stability issues: {'; '.join(issues)}", "WARNING")
        else:
            self.log("Training stability validation passed")

        return len(issues) == 0, issues

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "validation_results": self.validation_results,
            "overall_status": "PASS",
        }

        # Check if any validation failed
        for validation_key, validation_data in self.validation_results.items():
            if isinstance(validation_data, dict) and not validation_data.get("is_valid", True):
                summary["overall_status"] = "FAIL"
                break

        # Count issues
        total_issues = 0
        for validation_data in self.validation_results.values():
            if isinstance(validation_data, dict) and "issues" in validation_data:
                total_issues += len(validation_data["issues"])

        summary["total_issues"] = total_issues

        return summary

    def log_validation_summary(self):
        """Log comprehensive validation summary"""
        summary = self.get_validation_summary()

        self.log("=" * 50)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 50)

        self.log(f"Overall Status: {summary['overall_status']}")
        self.log(f"Total Issues: {summary['total_issues']}")

        for validation_key, validation_data in self.validation_results.items():
            if isinstance(validation_data, dict):
                status = "✓ PASS" if validation_data.get("is_valid", True) else "✗ FAIL"
                issues_count = len(validation_data.get("issues", []))
                self.log(f"{status}: {validation_key} ({issues_count} issues)")

        return summary


def create_pipeline_validator(logger=None, **kwargs):
    """Convenience function to create pipeline validator"""
    return PipelineValidator(logger, **kwargs)

