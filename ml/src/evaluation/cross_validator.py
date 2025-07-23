#!/usr/bin/env python3

"""
Cross-validation and model evaluation utilities
"""

import numpy as np
from sklearn.metrics import mean_absolute_error

# Add import for financial metrics
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.financial_metrics import FinancialMetrics


class TimeSeriesCrossValidator:
    """Time series cross-validation with expanding window"""

    def __init__(self, n_folds=3):
        self.n_folds = n_folds

    def split(self, X, y):
        """Generate train/test splits for time series data"""
        n_samples = len(X)
        test_size = n_samples // (self.n_folds + 1)

        splits = []
        for fold in range(self.n_folds):
            # Create expanding window split
            train_end = n_samples - (self.n_folds - fold) * test_size
            test_start = train_end
            test_end = test_start + test_size

            train_indices = list(range(train_end))
            test_indices = list(range(test_start, test_end))

            splits.append((train_indices, test_indices))

        return splits

    def validate_model(self, model_builder, X, y, config, logger=None):
        """Perform cross-validation on a model"""
        if logger:
            logger.start_phase("Cross Validation")

        fold_results = []
        splits = self.split(X, y)

        for fold, (train_idx, test_idx) in enumerate(splits):
            if logger:
                logger.log(f"Training fold {fold + 1}/{self.n_folds}")

            # Split data
            X_fold_train = X[train_idx]
            y_fold_train = y[train_idx]
            X_fold_test = X[test_idx]
            y_fold_test = y[test_idx]

            # Build and train model
            model = model_builder.build_lstm_model(input_shape=X.shape[1:])
            model = model_builder.compile_model(model)

            # Get callbacks
            callbacks = model_builder.get_callbacks(logger, f"CV Fold {fold + 1}")

            # Train model
            history = model.fit(
                X_fold_train,
                y_fold_train,
                validation_split=0.2,
                epochs=config.get("epochs", 500),
                batch_size=config.get("batch_size", 32),
                callbacks=callbacks,
                verbose=0,
            )

            # Evaluate
            pred = model.predict(X_fold_test, verbose=0).flatten()
            mae = mean_absolute_error(y_fold_test, pred)
            dir_acc = np.mean(np.sign(y_fold_test) == np.sign(pred)) * 100

            fold_results.append(
                {
                    "fold": fold + 1,
                    "mae": mae,
                    "dir_acc": dir_acc,
                    "epochs_trained": len(history.history["loss"]),
                }
            )

            if logger:
                logger.log(f"Fold {fold + 1}: MAE={mae:.4f}, Dir Acc={dir_acc:.1f}%")

        # Calculate summary statistics
        cv_results = self._summarize_cv_results(fold_results)

        if logger:
            logger.log(
                f"CV Results: MAE={cv_results['cv_mae_mean']:.4f}±{cv_results['cv_mae_std']:.4f}, "
                f"Dir Acc={cv_results['cv_dir_acc_mean']:.1f}%±{cv_results['cv_dir_acc_std']:.1f}%"
            )
            logger.end_phase("Cross Validation")

        return cv_results

    def _summarize_cv_results(self, fold_results):
        """Summarize cross-validation results"""
        mae_scores = [r["mae"] for r in fold_results]
        dir_acc_scores = [r["dir_acc"] for r in fold_results]

        return {
            "cv_mae_mean": np.mean(mae_scores),
            "cv_mae_std": np.std(mae_scores),
            "cv_dir_acc_mean": np.mean(dir_acc_scores),
            "cv_dir_acc_std": np.std(dir_acc_scores),
            "fold_results": fold_results,
            "best_fold": max(fold_results, key=lambda x: x["dir_acc"]),
            "worst_fold": min(fold_results, key=lambda x: x["dir_acc"]),
        }


class ModelEvaluator:
    """Comprehensive model evaluation"""

    @staticmethod
    def evaluate_predictions(y_true, y_pred, phase_name="Test"):
        """Calculate comprehensive evaluation metrics"""
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # Directional accuracy
        dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

        # Additional metrics
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

        # Statistical significance of directional accuracy
        n_correct = np.sum(np.sign(y_true) == np.sign(y_pred))
        n_total = len(y_true)

        # Confidence bounds for directional accuracy (binomial)
        p_hat = n_correct / n_total
        margin_error = 1.96 * np.sqrt(p_hat * (1 - p_hat) / n_total)
        dir_acc_lower = max(0, (p_hat - margin_error) * 100)
        dir_acc_upper = min(100, (p_hat + margin_error) * 100)

        # Return stratified by magnitude
        high_magnitude_mask = np.abs(y_true) > np.percentile(np.abs(y_true), 75)
        if np.sum(high_magnitude_mask) > 0:
            high_mag_dir_acc = np.mean(np.sign(y_true[high_magnitude_mask]) == np.sign(y_pred[high_magnitude_mask])) * 100
        else:
            high_mag_dir_acc = 0.0

        # Financial performance metrics
        financial_metrics = {}
        try:
            financial_metrics = FinancialMetrics.evaluate_trading_performance(y_pred, y_true)
        except Exception as e:
            # If financial metrics calculation fails, continue with basic metrics
            pass

        results = {
            "phase": phase_name,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "directional_accuracy": dir_acc,
            "dir_acc_ci_lower": dir_acc_lower,
            "dir_acc_ci_upper": dir_acc_upper,
            "correlation": correlation,
            "high_magnitude_dir_acc": high_mag_dir_acc,
            "n_samples": len(y_true),
            "n_correct_direction": n_correct,
        }
        
        # Add financial metrics if available
        results.update(financial_metrics)
        
        return results

    @staticmethod
    def compare_models(baseline_results, lstm_results, logger=None):
        """Compare model performance"""
        comparison = {}

        # Find best baseline
        best_baseline_name = max(baseline_results.keys(), key=lambda k: baseline_results[k]["dir_acc"])
        best_baseline = baseline_results[best_baseline_name]

        # Calculate improvements
        dir_acc_improvement = lstm_results["directional_accuracy"] - best_baseline["dir_acc"]
        mae_improvement = best_baseline["mae"] - lstm_results["mae"]  # Lower is better

        comparison = {
            "best_baseline": {
                "name": best_baseline_name,
                "dir_acc": best_baseline["dir_acc"],
                "mae": best_baseline["mae"],
            },
            "lstm": {
                "dir_acc": lstm_results["directional_accuracy"],
                "mae": lstm_results["mae"],
            },
            "improvements": {
                "dir_acc_improvement": dir_acc_improvement,
                "mae_improvement": mae_improvement,
                "dir_acc_improvement_pct": (dir_acc_improvement / best_baseline["dir_acc"]) * 100,
                "mae_improvement_pct": (mae_improvement / best_baseline["mae"]) * 100,
            },
        }

        # Performance assessment
        if dir_acc_improvement > 2:
            performance_assessment = "LSTM significantly outperforms baseline"
        elif dir_acc_improvement > -2:
            performance_assessment = "LSTM performance similar to baseline"
        else:
            performance_assessment = "LSTM underperforms baseline"

        comparison["assessment"] = performance_assessment

        if logger:
            logger.log("=== PERFORMANCE COMPARISON ===")
            logger.log(f"Best Baseline ({best_baseline_name}): {best_baseline['dir_acc']:.1f}%")
            logger.log(f"LSTM Model: {lstm_results['directional_accuracy']:.1f}%")
            logger.log(f"Performance Gain: {dir_acc_improvement:.1f}%")
            logger.log(performance_assessment)

        return comparison

    @staticmethod
    def create_performance_summary(train_results, val_results, test_results, cv_results=None):
        """Create comprehensive performance summary"""
        summary = {
            "train": train_results,
            "validation": val_results,
            "test": test_results,
        }

        if cv_results:
            summary["cross_validation"] = cv_results

        # Check for overfitting
        train_val_gap = train_results["directional_accuracy"] - val_results["directional_accuracy"]
        if train_val_gap > 10:
            overfitting_risk = "High"
        elif train_val_gap > 5:
            overfitting_risk = "Moderate"
        else:
            overfitting_risk = "Low"

        summary["overfitting_assessment"] = {
            "train_val_gap": train_val_gap,
            "risk_level": overfitting_risk,
        }

        # Generalization assessment
        if cv_results:
            cv_test_gap = abs(cv_results["cv_dir_acc_mean"] - test_results["directional_accuracy"])
            if cv_test_gap < 2:
                generalization = "Good"
            elif cv_test_gap < 5:
                generalization = "Fair"
            else:
                generalization = "Poor"

            summary["generalization_assessment"] = {
                "cv_test_gap": cv_test_gap,
                "assessment": generalization,
            }

        return summary
