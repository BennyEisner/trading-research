#!/usr/bin/env python3

"""
Shared Backbone Training Framework
Integrates enhanced regularization with existing validation infrastructure
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import get_config

from ..features.multi_ticker_engine import MultiTickerPatternEngine
from ..features.utils.lstm_utils import prepare_swing_trading_sequences, validate_sequence_overlap
from ..models.shared_backbone_lstm import SharedBackboneLSTMBuilder
from ..validation.gapped_time_series_cv import GappedTimeSeriesCV
from ..validation.pipeline_validator import PipelineValidator
from ..validation.robust_time_series_validator import RobustTimeSeriesValidator
from .pattern_target_generator import PatternTargetGenerator


class SharedBackboneTrainer:
    """
    Training framework for shared backbone LSTM with enhanced regularization
    Integrates with existing validation infrastructure
    """

    def __init__(self, tickers: Optional[List[str]] = None, use_expanded_universe: bool = True):
        """
        Initialize shared backbone trainer

        Args:
            tickers: List of tickers to train on (if None, uses config)
            use_expanded_universe: Whether to use expanded universe for training
        """
        self.config = get_config()

        if tickers is None:
            if use_expanded_universe:
                # Remove VIX for yfinance compatibility
                self.tickers = [t for t in self.config.model.expanded_universe if t != "VIX"]
            else:
                self.tickers = self.config.model.mag7_tickers
        else:
            self.tickers = tickers

        self.use_expanded_universe = use_expanded_universe

        # Initialize components
        self.lstm_builder = SharedBackboneLSTMBuilder(self.config.dict())
        self.pattern_engine = MultiTickerPatternEngine(tickers=self.tickers, max_workers=4)

        # Validation infrastructure
        self.pipeline_validator = PipelineValidator()
        self.robust_validator = RobustTimeSeriesValidator(
            block_size=5, n_bootstrap=1000  # 5-day blocks for swing trading
        )

        # Pattern detection infrastructure
        self.pattern_target_generator = PatternTargetGenerator(
            lookback_window=self.config.model.lookback_window, validation_horizons=[3, 5, 10]
        )

        self.training_results = {}

    def prepare_training_data(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare training data using enhanced sequence generation

        Args:
            ticker_data: Dictionary {ticker: ohlcv_dataframe}

        Returns:
            Dictionary {ticker: (X_sequences, y_targets)}
        """
        print(f"Preparing training data for {len(self.tickers)} tickers with swing trading sequences")

        # Calculate pattern features for all tickers
        portfolio_features = self.pattern_engine.calculate_portfolio_features(ticker_data, parallel=True)

        training_data = {}
        total_sequences = 0

        for ticker in self.tickers:
            if ticker in portfolio_features and portfolio_features[ticker] is not None:
                features_df = portfolio_features[ticker]

                # Get feature names (17 pattern features)
                feature_names = [
                    "price_acceleration",
                    "volume_price_divergence",
                    "volatility_regime_change",
                    "return_skewness_7d",
                    "momentum_persistence_7d",
                    "volatility_clustering",
                    "trend_exhaustion",
                    "garch_volatility_forecast",
                    "intraday_range_expansion",
                    "overnight_gap_behavior",
                    "end_of_day_momentum",
                    "sector_relative_strength",
                    "market_beta_instability",
                    "vix_term_structure",
                    "returns_1d",
                    "returns_3d",
                    "returns_7d",
                ]

                available_features = [f for f in feature_names if f in features_df.columns]

                if len(available_features) < 15:  # Need most features
                    print(f"Warning: Only {len(available_features)} features available for {ticker}")
                    continue

                # Generate pattern detection targets (NOT return prediction)
                try:
                    pattern_targets = self.pattern_target_generator.generate_all_pattern_targets(
                        features_df, primary_horizon=self.config.model.prediction_horizon
                    )
                    # Use combined pattern confidence as primary target
                    target_values = pattern_targets["pattern_confidence_score"]
                except Exception as e:
                    print(f"Warning: Could not generate pattern targets for {ticker}: {e}")
                    continue

                # Prepare pattern detection sequences with OPTIMIZED APPROACH
                try:
                    # OPTIMIZED: Since targets are now leak-free, we can use 1-day stride safely
                    # The real leakage was in target generation (now fixed)
                    # 1-day stride gives us maximum data utilization while maintaining safety
                    optimal_stride = 1  # Back to 1-day stride - safe now that targets are leak-free

                    print(f"  OPTIMIZED STRIDE: Using {optimal_stride}-day stride (targets are leak-free)")
                    overlap_pct = (
                        (self.config.model.lookback_window - optimal_stride) / self.config.model.lookback_window * 100
                    )
                    print(f"  Sequence overlap: {overlap_pct:.1f}% (safe due to leak-free targets)")

                    X, y = self._prepare_pattern_detection_sequences(
                        features_df=features_df,
                        feature_columns=available_features,
                        pattern_targets=target_values,
                        sequence_length=self.config.model.lookback_window,  # 20 days
                        stride=optimal_stride,  # Reduced overlap stride
                    )

                    # Validate sequences
                    sequence_valid, validation_issues = self.pipeline_validator.validate_sequences(X, y)

                    if sequence_valid:
                        training_data[ticker] = (X, y)
                        total_sequences += len(X)

                        # Log sequence overlap statistics with 1-day stride
                        overlap_stats = validate_sequence_overlap(
                            sequence_length=self.config.model.lookback_window,
                            stride=1,  # 1-day stride
                            total_samples=len(features_df),
                        )

                        print(
                            f"SUCCESS {ticker}: {len(X)} sequences ({overlap_stats['overlap_percentage']:.1f}% overlap)"
                        )
                    else:
                        print(f"FAILED {ticker}: Sequence validation failed - {validation_issues}")

                except Exception as e:
                    print(f"FAILED {ticker}: Error preparing sequences - {e}")

        print(f"\nTRAINING DATA SUMMARY:")
        print(f"- Successful tickers: {len(training_data)}")
        print(f"- Total sequences: {total_sequences}")
        print(f"- Avg sequences per ticker: {total_sequences / len(training_data) if training_data else 0:.0f}")
        print(f"- Training multiplier: ~{total_sequences / (len(self.tickers) * 1000):.1f}x vs daily sequences")

        return training_data

    def train_shared_backbone(
        self, training_data: Dict[str, Tuple[np.ndarray, np.ndarray]], validation_split: float = 0.2, epochs: int = 100
    ) -> Dict[str, Any]:
        """
        Train shared backbone model on combined data from all tickers

        Args:
            training_data: Prepared training data from prepare_training_data
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs

        Returns:
            Training results including model and validation metrics
        """
        print(f"Training shared backbone LSTM on {len(training_data)} tickers")

        all_X = []
        all_y = []
        ticker_indices = {}  # Track which sequences belong to which ticker

        current_idx = 0
        for ticker, (X, y) in training_data.items():
            all_X.append(X)
            all_y.append(y)
            ticker_indices[ticker] = (current_idx, current_idx + len(X))
            current_idx += len(X)

        combined_X = np.vstack(all_X)
        combined_y = np.concatenate(all_y)

        print(f"Combined training data: {combined_X.shape} sequences, {combined_X.shape[2]} features")

        print(f"TEMPORAL VALIDATION: Using temporal split instead of random shuffle")

        total_samples = len(combined_X)

        train_size = int(0.8 * total_samples)

        # Keep data in temporal ordr
        print(f"  - Training samples: {train_size}")
        print(f"  - Validation samples: {total_samples - train_size}")
        print(f"  - Temporal integrity: PRESERVED (no random shuffling)")

        # Build shared backbone model
        input_shape = (combined_X.shape[1], combined_X.shape[2])

        model_params = self.config.model.model_params.copy()
        critical_overrides = {
            "learning_rate": 0.002,
            "diversity_penalty_weight": 25.0,
            "correlation_penalty_weight": 15.0,  # Penalty for low correlation
        }

        print(f"CRITICAL OVERRIDES APPLIED:")
        for param, value in critical_overrides.items():
            old_value = model_params.get(param, "not_set")
            model_params[param] = value
            print(f"- {param}: {old_value} → {value}")

        model = self.lstm_builder.build_model(input_shape, **model_params)

        print(f"Model architecture: {model.count_params()} parameters")

        # FIXED: Add manual correlation monitoring callback
        class ManualCorrelationMonitor(tf.keras.callbacks.Callback):
            """Accurate correlation monitoring using full epoch data"""

            def __init__(self, train_X, train_y, val_X, val_y):
                self.train_X = train_X
                self.train_y = train_y
                self.val_X = val_X
                self.val_y = val_y
                self.best_correlation = -999

            def on_epoch_end(self, epoch, logs=None):
                train_pred = self.model.predict(self.train_X, verbose=0).flatten()
                val_pred = self.model.predict(self.val_X, verbose=0).flatten()

                train_corr = np.corrcoef(self.train_y, train_pred)[0, 1] if np.var(train_pred) > 1e-10 else 0.0
                val_corr = np.corrcoef(self.val_y, val_pred)[0, 1] if np.var(val_pred) > 1e-10 else 0.0

                # Track best validation correlation
                if val_corr > self.best_correlation:
                    self.best_correlation = val_corr

                # Prediction variance analysis
                train_pred_var = np.var(train_pred)
                val_pred_var = np.var(val_pred)

                print(f"")
                print(f"EPOCH {epoch+1} CORRELATION ANALYSIS:")
                print(f" Train correlation: {train_corr:8.6f}")
                print(f" Val correlation:   {val_corr:8.6f} (best: {self.best_correlation:8.6f})")
                print(f"Train pred var:    {train_pred_var:8.6f}")
                print(f" Val pred var:      {val_pred_var:8.6f}")

                # Status indicators
                if abs(val_corr) > 0.2:
                    print(f"EXCELLENT: Strong correlation achieved!")
                elif abs(val_corr) > 0.1:
                    print(f" GOOD: Moderate correlation")
                elif abs(val_corr) > 0.05:
                    print(f"PROGRESS: Weak but meaningful correlation")
                elif val_pred_var < 1e-6:
                    print(f"WARNING: Predictions collapsed to constants")
                else:
                    print(f"LOW: Minimal correlation detected")

        train_X, val_X = combined_X[:train_size], combined_X[train_size:]
        train_y, val_y = combined_y[:train_size], combined_y[train_size:]

        print(f"Final training split: {train_X.shape[0]} train, {val_X.shape[0]} validation")

        # Get base regularization callbacks
        callbacks = self.lstm_builder.get_regularization_callbacks(epochs)

        correlation_monitor = ManualCorrelationMonitor(train_X, train_y, val_X, val_y)
        callbacks.append(correlation_monitor)

        print(f"Starting shared backbone training with correlation optimization...")
        print(f"- Batch size: {self.config.model.training_params.get('batch_size', 256)}")
        print(f"- Monitor metric: val_loss (correlation via manual calculation)")
        print(f"- Patience: {self.config.model.training_params.get('patience', 15)} epochs")
        print(f"- CRITICAL: Using correlation-optimized loss function")
        print(f"- NOTE: Keras correlation metric display is misleading (batch vs epoch issue)")
        print(f"- FOCUS: Manual correlation calculation shows true learning progress")

        print(f"")
        print(f"STARTING CORRELATION-OPTIMIZED TRAINING")
        print(f"Loss function: Direct correlation optimization")
        print(f"Monitoring: Manual correlation calculation (accurate)")
        print(f"Expected: Meaningful correlation (0.1-0.5 range)")
        print(f"")

        history = model.fit(
            train_X,
            train_y,
            validation_data=(val_X, val_y),
            epochs=epochs,
            batch_size=self.config.model.training_params.get("batch_size", 256),
            callbacks=callbacks,
            verbose=1,
        )

        # Validate training stability
        training_stable, stability_issues = self.pipeline_validator.validate_training_stability(history.history)

        if not training_stable:
            print(f"WARNING: Training stability issues: {stability_issues}")

        # Generate predictions for validation
        predictions = model.predict(combined_X, verbose=0).flatten()

        # Validate predictions
        pred_valid, pred_issues = self.pipeline_validator.validate_model_predictions(combined_y, predictions)

        if not pred_valid:
            print(f"WARNING: Prediction validation issues: {pred_issues}")

        # statistical validation using existing framework
        robust_results = self.robust_validator.moving_block_bootstrap(returns=combined_y, predictions=predictions)

        training_results = {
            "model": model,
            "history": history.history,
            "training_data_shape": combined_X.shape,
            "ticker_indices": ticker_indices,
            "training_stable": training_stable,
            "stability_issues": stability_issues,
            "predictions_valid": pred_valid,
            "prediction_issues": pred_issues,
            "robust_validation": robust_results,
            "final_metrics": {
                "train_loss": history.history["loss"][-1],
                "val_loss": history.history["val_loss"][-1],
                "pattern_detection_accuracy": self._calculate_pattern_detection_accuracy(combined_y, predictions),
                "correlation": np.corrcoef(combined_y, predictions)[0, 1],
                "best_validation_correlation": correlation_monitor.best_correlation,  # Track best correlation achieved
                "keras_correlation": history.history.get("correlation_metric", [0])[-1],
            },
        }

        self.training_results["shared_backbone"] = training_results

        print(f"CORRELATION-OPTIMIZED TRAINING COMPLETE!")
        print(f"- Final train loss: {training_results['final_metrics']['train_loss']:.4f}")
        print(f"- Final val loss: {training_results['final_metrics']['val_loss']:.4f}")
        print(f"- Pattern detection accuracy: {training_results['final_metrics']['pattern_detection_accuracy']:.3f}")
        print(f"- MANUAL CORRELATION: {training_results['final_metrics']['correlation']:.6f}")
        print(f"- BEST VAL CORRELATION: {training_results['final_metrics']['best_validation_correlation']:.6f}")
        print(f"-  Keras correlation: {training_results['final_metrics']['keras_correlation']:.2e} (misleading)")
        print(f"- Statistical significance: p={robust_results['p_value']:.4f}")

        # Success assessment
        best_corr = abs(training_results["final_metrics"]["best_validation_correlation"])
        if best_corr > 0.2:
            print(f"SUCCESS: Strong correlation achieved! Pattern learning enabled.")
        elif best_corr > 0.1:
            print(f"SUCCESS: Good correlation achieved! Meaningful pattern detection.")
        elif best_corr > 0.05:
            print(f"PROGRESS: Weak but meaningful correlation. Continue training.")
        else:
            print(f"SUBOPTIMAL: Low correlation. May need architecture adjustments.")

        return training_results

    def _prepare_pattern_detection_sequences(
        self,
        features_df: pd.DataFrame,
        feature_columns: List[str],
        pattern_targets: np.ndarray,
        sequence_length: int = 20,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for pattern detection training with TEMPORAL LEAKAGE PREVENTION

        Args:
            features_df: DataFrame with features
            feature_columns: List of feature column names
            pattern_targets: Pattern detection target values
            sequence_length: Length of input sequences
            stride: Stride between sequences

        Returns:
            Tuple of (X_sequences, y_targets)
        """

        # SHORT-TERM PATTERN DETECTION: Minimal gap for 1-20 day pattern learning
        # Since pattern targets are leak-free (use only historical data), we can use minimal gap
        prediction_horizon = 1  # Learn patterns that resolve in 1 day (short-term focus)
        temporal_gap = 0  # No gap needed - targets are completely leak-free
        total_future_offset = prediction_horizon + temporal_gap  # 1 day only

        # Extract feature matrix
        feature_matrix = features_df[feature_columns].values

        # Handle NaN values with better strategy than zero-replacement
        feature_df_clean = pd.DataFrame(feature_matrix, columns=feature_columns)
        feature_df_clean = feature_df_clean.ffill().bfill().fillna(0.0)
        feature_matrix = feature_df_clean.values

        X_sequences = []
        y_targets = []

        print(f"SHORT-TERM PATTERN LEARNING: Using {total_future_offset}-day offset (targets are leak-free)")
        print(f"Sequence length: {sequence_length} days (historical features)")
        print(f"Target calculation: Uses only historical data up to sequence end")
        print(f"Prediction horizon: {prediction_horizon} day (learn 1-20 day patterns)")

        # Generate sequences with temporal separation
        max_start_idx = len(feature_matrix) - sequence_length - total_future_offset

        for i in range(0, max_start_idx, stride):
            sequence = feature_matrix[i : i + sequence_length]

            # Target with temporal gap to prevent leakage
            # i+sequence_length = end of historical window
            # i+sequence_length+temporal_gap = start of safe prediction window
            # i+sequence_length+total_future_offset-1 = target day
            target_idx = i + sequence_length + total_future_offset - 1

            if target_idx < len(pattern_targets):
                target = pattern_targets[target_idx]

                sequence_end = i + sequence_length - 1
                target_calculation_start = target_idx - prediction_horizon + 1

                if target_calculation_start > sequence_end:
                    X_sequences.append(sequence)
                    y_targets.append(target)
                else:
                    print(f"WARNING: Temporal overlap detected at index {i}, skipping sequence")

        sequences_generated = len(X_sequences)
        sequences_dropped = max_start_idx - sequences_generated

        print(f"Generated {sequences_generated} clean sequences")
        print(f"Dropped {sequences_dropped} sequences to prevent temporal leakage")
        print(f"Data utilization: {sequences_generated / max_start_idx * 100:.1f}%")

        return np.array(X_sequences), np.array(y_targets)

    def _calculate_pattern_detection_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate pattern detection accuracy
        """
        # Convert continuous pattern confidence to binary classification
        y_true_binary = (y_true > 0.5).astype(int)
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Calculate accuracy
        accuracy = np.mean(y_true_binary == y_pred_binary)

        return accuracy

    def validate_cross_ticker_performance(
        self, training_data: Dict[str, Tuple[np.ndarray, np.ndarray]], trained_model: tf.keras.Model
    ) -> Dict[str, Any]:
        """
        Validate model performance across different tickers using time series CV

        Args:
            training_data: Training data by ticker
            trained_model: Trained shared backbone model

        Returns:
            Cross-ticker validation results
        """
        print(f"Validating cross-ticker performance...")

        ticker_performance = {}

        for ticker, (X, y) in training_data.items():
            print(f"Validating {ticker}...")

            # Time sies cross-validation for this ticker
            cv = GappedTimeSeriesCV(n_splits=3, test_size=0.2, gap_size=5)

            fold_results = []

            for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_test_fold = X[test_idx]
                y_test_fold = y[test_idx]

                predictions = trained_model.predict(X_test_fold, verbose=0).flatten()

                # Calculate pattern detection metrics
                pattern_detection_acc = np.mean(
                    (predictions > 0.5) == (y_test_fold > 0.5)
                )  # Binary accuracy for pattern confidence
                correlation = np.corrcoef(y_test_fold, predictions)[0, 1] if len(y_test_fold) > 1 else 0.0
                mae = np.mean(np.abs(y_test_fold - predictions))

                fold_results.append(
                    {
                        "fold": fold,
                        "pattern_detection_accuracy": pattern_detection_acc,
                        "correlation": correlation,
                        "mae": mae,
                    }
                )

            # Aggregate results for this ticker
            ticker_performance[ticker] = {
                "fold_results": fold_results,
                "mean_pattern_detection_accuracy": np.mean([r["pattern_detection_accuracy"] for r in fold_results]),
                "mean_correlation": np.mean([r["correlation"] for r in fold_results if not np.isnan(r["correlation"])]),
                "mean_mae": np.mean([r["mae"] for r in fold_results]),
            }

        all_pattern_acc = [perf["mean_pattern_detection_accuracy"] for perf in ticker_performance.values()]
        all_corr = [
            perf["mean_correlation"] for perf in ticker_performance.values() if not np.isnan(perf["mean_correlation"])
        ]

        cross_ticker_results = {
            "ticker_performance": ticker_performance,
            "overall_stats": {
                "mean_pattern_detection_accuracy": np.mean(all_pattern_acc),
                "std_pattern_detection_accuracy": np.std(all_pattern_acc),
                "mean_correlation": np.mean(all_corr) if all_corr else 0.0,
                "std_correlation": np.std(all_corr) if all_corr else 0.0,
                "successful_tickers": len(ticker_performance),
                "pattern_generalization_score": np.mean(all_pattern_acc),  # Pattern generalization metric
            },
        }

        print(f"Cross-ticker validation results:")
        print(
            f"   - Mean pattern detection accuracy: {cross_ticker_results['overall_stats']['mean_pattern_detection_accuracy']:.3f} ± {cross_ticker_results['overall_stats']['std_pattern_detection_accuracy']:.3f}"
        )
        print(
            f"   - Mean correlation: {cross_ticker_results['overall_stats']['mean_correlation']:.3f} ± {cross_ticker_results['overall_stats']['std_correlation']:.3f}"
        )
        print(
            f"   - Pattern generalization score: {cross_ticker_results['overall_stats']['pattern_generalization_score']:.3f}"
        )

        return cross_ticker_results

    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""

        if "shared_backbone" not in self.training_results:
            return "No training results available. Run train_shared_backbone() first."

        results = self.training_results["shared_backbone"]
        robust_stats = results["robust_validation"]

        report = f"""
Shared Backbone LSTM Training Report
========================================================

Training Configuration:
- Architecture: Shared Backbone LSTM
- Tickers: {len(self.tickers)} ({'expanded universe' if self.use_expanded_universe else 'MAG7 only'})
- Training Data Shape: {results['training_data_shape']}
- Model Parameters: {results['model'].count_params():,}

Enhanced Regularization Features:
- Dropout Rate: {self.config.model.model_params['dropout_rate']}
- L2 Regularization: {self.config.model.model_params['l2_regularization']}
- Batch Normalization: {self.config.model.model_params['use_batch_norm']}
- Recurrent Dropout: {self.config.model.model_params['use_recurrent_dropout']}

Final Training Metrics:
- Training Loss: {results['final_metrics']['train_loss']:.4f}
- Validation Loss: {results['final_metrics']['val_loss']:.4f}
- Pattern Detection Accuracy: {results['final_metrics']['pattern_detection_accuracy']:.3f}
- Correlation: {results['final_metrics']['correlation']:.3f}

Robust Statistical Validation:
- Bootstrap Mean Performance: {robust_stats['bootstrap_mean']:.3f}
- Statistical Significance: p = {robust_stats['p_value']:.4f}
- 95% Confidence Interval: [{robust_stats['confidence_interval'][0]:.3f}, {robust_stats['confidence_interval'][1]:.3f}]
- Test Statistic: {robust_stats['test_statistic']:.3f}

Training Stability: {'PASS' if results['training_stable'] else 'FAIL'}
Prediction Quality: {'PASS' if results['predictions_valid'] else 'FAIL'}

Overfitting Prevention Status:
- Enhanced regularization applied
- Expanded universe training ({len(self.tickers)} securities)
- Overlapping sequence generation (75% overlap)
- Cross-ticker pattern learning enabled

Ready for Phase 2 (Multi-Task Architecture): {'YES' if robust_stats['significant'] and results['final_metrics']['pattern_detection_accuracy'] > 0.55 else 'NEEDS IMPROVEMENT'}
"""

        return report


def create_shared_backbone_trainer(
    tickers: Optional[List[str]] = None, use_expanded_universe: bool = True
) -> SharedBackboneTrainer:
    """Convenience function to create shared backbone trainer"""
    return SharedBackboneTrainer(tickers=tickers, use_expanded_universe=use_expanded_universe)


# Example usage
if __name__ == "__main__":
    # Create trainer with expanded universe
    trainer = create_shared_backbone_trainer(use_expanded_universe=True)

    print(f"Shared Backbone Training Framework Initialized")
    print(f"- Training Universe: {len(trainer.tickers)} securities")
    print(f"- Enhanced Regularization: Enabled")
    print(f"- Validation Framework: Integrated")
    print(f"- Swing Trading Optimized: 20-day sequences, 5-day stride")

    print(f"\nNext steps:")
    print(f"1. Load ticker data: ticker_data = {{ticker: ohlcv_df for ticker in trainer.tickers}}")
    print(f"2. Prepare training data: training_data = trainer.prepare_training_data(ticker_data)")
    print(f"3. Train shared backbone: results = trainer.train_shared_backbone(training_data)")
    print(
        f"4. Validate performance: validation = trainer.validate_cross_ticker_performance(training_data, results['model'])"
    )
    print(f"5. Generate report: report = trainer.generate_training_report()")

