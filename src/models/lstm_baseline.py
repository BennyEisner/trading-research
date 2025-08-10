#!/usr/bin/env python3

"""
LSTM Performance Baseline Framework
Establishes standalone LSTM performance before ensemble integration
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(str(Path(__file__).parent.parent))

from config.config import get_config
from models.shared_backbone_lstm import SharedBackboneLSTMBuilder


class LSTMBaselineValidator:
    """
    Validates LSTM performance in isolation before ensemble integration
    Implements walk-forward validation with performance attribution
    """

    def __init__(self):
        self.config = get_config()
        self.lstm_builder = SharedBackboneLSTMBuilder(self.config.dict())
        self.logger = self._setup_logging()

        # Pattern detection performance targets
        self.targets = {
            "pattern_detection_accuracy": 0.55,  # Minimum for pattern detection specialist
            "momentum_pattern_accuracy": 0.55,  # Individual pattern thresholds
            "volatility_pattern_accuracy": 0.55,
            "trend_pattern_accuracy": 0.55,
            "sharpe_ratio": 0.8,
            "max_drawdown": 0.15,
            "correlation": 0.25,  # Signal quality threshold
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation"""
        logger = logging.getLogger("lstm_baseline")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def validate_standalone_performance(
        self, features: np.ndarray, targets: np.ndarray, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Perform walk-forward LSTM validation

        Args:
            features: Pattern features (n_samples, 30, 17) - Updated for new pattern feature set
            targets: Future returns
            data: Market data with timestamps

        Returns:
            Validation results with ensemble readiness assessment
        """
        self.logger.info("Starting LSTM standalone validation")

        # Walk-forward validation
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_results = []
        models = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
            self.logger.info(f"Processing fold {fold + 1}/{n_splits}")

            # Split data
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = targets[train_idx], targets[test_idx]

            # Train model
            model = self._train_fold_model(X_train, y_train, fold)
            models.append(model)

            # Generate predictions
            predictions = model.predict(X_test, verbose=0).flatten()

            # Calculate metrics
            fold_metrics = self._calculate_fold_metrics(y_test, predictions)
            fold_metrics["fold"] = fold
            fold_results.append(fold_metrics)

            self.logger.info(f"Fold {fold + 1} - Pattern Detection Accuracy: {fold_metrics['pattern_detection_accuracy']:.3f}")

        # Aggregate results
        final_results = self._aggregate_results(fold_results)

        # Ensemble readiness assessment
        readiness = self._assess_ensemble_readiness(final_results)

        validation_results = {
            "fold_results": fold_results,
            "aggregated_metrics": final_results,
            "ensemble_readiness": readiness,
            "models": models,
            "validation_date": datetime.now().isoformat(),
        }

        self.logger.info("LSTM validation completed")
        return validation_results

    def _train_fold_model(self, X_train: np.ndarray, y_train: np.ndarray, fold: int):
        """Train LSTM model for specific fold"""

        # Get model configuration
        model_config = {**self.config.model.model_params, **self.config.model.training_params}

        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])  # (30, 17) - Updated for pattern features
        model = self.lstm_builder.build_model(input_shape, **model_config)

        # Prepare callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0),
        ]

        # Train
        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=model_config.get("epochs", 50),
            batch_size=model_config.get("batch_size", 64),
            callbacks=callbacks,
            verbose=0,
        )

        return model

    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for validation fold - Updated for pattern detection"""

        # Pattern detection accuracy (assuming y_true and y_pred are pattern confidence scores 0-1)
        # Convert to binary classification with 0.5 threshold
        y_true_binary = (y_true > 0.5).astype(int)
        y_pred_binary = (y_pred > 0.5).astype(int)
        pattern_detection_accuracy = accuracy_score(y_true_binary, y_pred_binary)

        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

        # Pattern-based metrics (simplified since we're not doing return prediction)
        # For pattern detection, we focus on pattern recognition accuracy
        
        # Pattern confidence correlation
        pattern_confidence_corr = correlation  # Reuse correlation for pattern confidence
        
        # Pattern precision (how often high confidence predictions are correct)
        high_conf_mask = y_pred > 0.7
        pattern_precision = (
            np.mean(y_true_binary[high_conf_mask] == y_pred_binary[high_conf_mask])
            if np.sum(high_conf_mask) > 0 else 0.0
        )
        
        # Pattern recall (how often true patterns are detected)
        true_pattern_mask = y_true > 0.5
        pattern_recall = (
            np.mean(y_pred_binary[true_pattern_mask] == y_true_binary[true_pattern_mask])
            if np.sum(true_pattern_mask) > 0 else 0.0
        )

        return {
            "pattern_detection_accuracy": pattern_detection_accuracy,
            "correlation": pattern_confidence_corr,
            "pattern_precision": pattern_precision,
            "pattern_recall": pattern_recall,
            "mae": np.mean(np.abs(y_true - y_pred)),
            "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        }

    def _aggregate_results(self, fold_results: list) -> Dict[str, float]:
        """Aggregate metrics across folds"""

        metrics = ["pattern_detection_accuracy", "correlation", "pattern_precision", "pattern_recall", "mae", "rmse"]

        aggregated = {}
        for metric in metrics:
            values = [result[metric] for result in fold_results if not np.isnan(result[metric])]
            if values:
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
            else:
                aggregated[f"{metric}_mean"] = 0.0
                aggregated[f"{metric}_std"] = 0.0

        return aggregated

    def _assess_ensemble_readiness(self, results: Dict[str, float]) -> Dict[str, Any]:
        """Assess if LSTM is ready for ensemble integration"""

        assessments = {}
        for metric, target in self.targets.items():
            actual = results.get(f"{metric}_mean", 0.0)

            if metric == "max_drawdown":
                meets_target = abs(actual) <= target
            else:
                meets_target = actual >= target

            assessments[metric] = {
                "target": target,
                "actual": actual,
                "meets_target": meets_target,
                "gap": actual - target if metric != "max_drawdown" else target - abs(actual),
            }

        # Overall readiness
        ready_for_ensemble = all(assessment["meets_target"] for assessment in assessments.values())

        return {
            "individual_assessments": assessments,
            "ready_for_ensemble": ready_for_ensemble,
            "recommendation": (
                "PROCEED to ensemble integration" if ready_for_ensemble else "IMPROVE model before integration"
            ),
            "key_metrics_summary": {
                "pattern_detection_accuracy": f"{results['pattern_detection_accuracy_mean']:.3f} ± {results['pattern_detection_accuracy_std']:.3f}",
                "pattern_precision": f"{results['pattern_precision_mean']:.3f} ± {results['pattern_precision_std']:.3f}",
                "correlation": f"{results['correlation_mean']:.3f} ± {results['correlation_std']:.3f}",
            },
        }

    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""

        readiness = validation_results["ensemble_readiness"]

        report = f"""
LSTM Standalone Performance Validation Report
============================================

Validation Date: {validation_results['validation_date']}

ENSEMBLE READINESS: {readiness['recommendation']}

KEY METRICS:
- Pattern Detection Accuracy: {readiness['key_metrics_summary']['pattern_detection_accuracy']}
- Sharpe Ratio: {readiness['key_metrics_summary']['sharpe_ratio']}  
- Correlation: {readiness['key_metrics_summary']['correlation']}

TARGET ASSESSMENT:
"""

        for metric, assessment in readiness["individual_assessments"].items():
            status = "PASS" if assessment["meets_target"] else "FAIL"
            gap_text = f"({assessment['gap']:+.3f})" if assessment["gap"] != 0 else ""
            report += f"- {metric.replace('_', ' ').title()}: {assessment['actual']:.3f} vs {assessment['target']:.3f} {gap_text} {status}\n"

        report += f"\nNEXT STEPS:\n"

        if readiness["ready_for_ensemble"]:
            report += "1. Proceed to hierarchical ensemble integration\n"
            report += "2. Implement LSTM as refinement layer\n"
            report += "3. Create component attribution framework\n"
        else:
            report += "1. Improve pattern-focused feature engineering\n"
            report += "2. Optimize model architecture and hyperparameters\n"
            report += "3. Increase data quality and feature selection\n"
            report += "4. Re-validate before ensemble integration\n"

        return report


def create_test_data() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create synthetic test data for validation"""

    n_samples = 1000
    sequence_length = 30
    n_features = 17  # Updated to match new pattern feature count

    # Generate synthetic features
    features = np.random.randn(n_samples, sequence_length, n_features)

    # Generate correlated targets (some predictive signal)
    noise = np.random.randn(n_samples) * 0.015
    signal = np.mean(features[:, -5:, :3], axis=(1, 2)) * 0.005  # Weak signal from recent features
    targets = signal + noise

    # Create synthetic data DataFrame
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    data = pd.DataFrame({"date": dates, "close": 100 + np.cumsum(targets), "returns": targets})

    return features, targets, data


if __name__ == "__main__":
    # Quick test
    features, targets, data = create_test_data()

    validator = LSTMBaselineValidator()
    results = validator.validate_standalone_performance(features, targets, data)

    report = validator.generate_validation_report(results)
    print(report)
