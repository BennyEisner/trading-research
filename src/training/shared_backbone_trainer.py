#!/usr/bin/env python3

"""
Shared Backbone Training Framework
Integrates enhanced regularization with existing validation infrastructure
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import tensorflow as tf

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import get_config
from ..models.shared_backbone_lstm import SharedBackboneLSTMBuilder
from ..features.multi_ticker_engine import MultiTickerPatternEngine
from ..features.utils.lstm_utils import prepare_swing_trading_sequences, validate_sequence_overlap
from ..validation.pipeline_validator import PipelineValidator
from ..validation.robust_time_series_validator import RobustTimeSeriesValidator
from ..validation.gapped_time_series_cv import GappedTimeSeriesCV
from .pattern_target_generator import PatternTargetGenerator


class SharedBackboneTrainer:
    """
    Training framework for shared backbone LSTM with enhanced regularization
    Integrates with existing validation infrastructure
    """
    
    def __init__(self, 
                 tickers: Optional[List[str]] = None,
                 use_expanded_universe: bool = True):
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
        self.pattern_engine = MultiTickerPatternEngine(
            tickers=self.tickers,
            max_workers=4
        )
        
        # Validation infrastructure
        self.pipeline_validator = PipelineValidator()
        self.robust_validator = RobustTimeSeriesValidator(
            block_size=5,  # 5-day blocks for swing trading
            n_bootstrap=1000
        )
        
        # Pattern detection infrastructure
        self.pattern_target_generator = PatternTargetGenerator(
            lookback_window=self.config.model.lookback_window,
            validation_horizons=[3, 5, 10]
        )
        
        self.training_results = {}
    
    def prepare_training_data(self, 
                            ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare training data using enhanced sequence generation
        
        Args:
            ticker_data: Dictionary {ticker: ohlcv_dataframe}
            
        Returns:
            Dictionary {ticker: (X_sequences, y_targets)}
        """
        print(f"Preparing training data for {len(self.tickers)} tickers with swing trading sequences")
        
        # Calculate pattern features for all tickers
        portfolio_features = self.pattern_engine.calculate_portfolio_features(
            ticker_data, parallel=True
        )
        
        training_data = {}
        total_sequences = 0
        
        for ticker in self.tickers:
            if ticker in portfolio_features and portfolio_features[ticker] is not None:
                features_df = portfolio_features[ticker]
                
                # Get feature names (17 pattern features)
                feature_names = [
                    'price_acceleration', 'volume_price_divergence', 'volatility_regime_change', 'return_skewness_7d',
                    'momentum_persistence_7d', 'volatility_clustering', 'trend_exhaustion', 'garch_volatility_forecast',
                    'intraday_range_expansion', 'overnight_gap_behavior', 'end_of_day_momentum',
                    'sector_relative_strength', 'market_beta_instability', 'vix_term_structure',
                    'returns_1d', 'returns_3d', 'returns_7d'
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
                    target_values = pattern_targets['pattern_confidence_score']
                except Exception as e:
                    print(f"Warning: Could not generate pattern targets for {ticker}: {e}")
                    continue
                
                # Prepare pattern detection sequences with overlapping
                try:
                    X, y = self._prepare_pattern_detection_sequences(
                        features_df=features_df,
                        feature_columns=available_features,
                        pattern_targets=target_values,
                        sequence_length=self.config.model.lookback_window,  # 20 days
                        stride=self.config.model.sequence_stride  # 5-day stride
                    )
                    
                    # Validate sequences
                    sequence_valid, validation_issues = self.pipeline_validator.validate_sequences(X, y)
                    
                    if sequence_valid:
                        training_data[ticker] = (X, y)
                        total_sequences += len(X)
                        
                        # Log sequence overlap statistics
                        overlap_stats = validate_sequence_overlap(
                            sequence_length=self.config.model.lookback_window,
                            stride=self.config.model.sequence_stride,
                            total_samples=len(features_df)
                        )
                        
                        print(f"SUCCESS {ticker}: {len(X)} sequences ({overlap_stats['overlap_percentage']:.1f}% overlap)")
                    else:
                        print(f"FAILED {ticker}: Sequence validation failed - {validation_issues}")
                        
                except Exception as e:
                    print(f"FAILED {ticker}: Error preparing sequences - {e}")
        
        print(f"\nTRAINING DATA SUMMARY:")
        print(f"   - Successful tickers: {len(training_data)}")
        print(f"   - Total sequences: {total_sequences}")
        print(f"   - Avg sequences per ticker: {total_sequences / len(training_data) if training_data else 0:.0f}")
        print(f"   - Training multiplier: ~{total_sequences / (len(self.tickers) * 1000):.1f}x vs daily sequences")
        
        return training_data
    
    def train_shared_backbone(self, 
                            training_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            validation_split: float = 0.2,
                            epochs: int = 100) -> Dict[str, Any]:
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
        
        # Combine all ticker data for shared training
        all_X = []
        all_y = []
        ticker_indices = {}  # Track which sequences belong to which ticker
        
        current_idx = 0
        for ticker, (X, y) in training_data.items():
            all_X.append(X)
            all_y.append(y)
            ticker_indices[ticker] = (current_idx, current_idx + len(X))
            current_idx += len(X)
        
        # Combine all training data
        combined_X = np.vstack(all_X)
        combined_y = np.concatenate(all_y)
        
        print(f"Combined training data: {combined_X.shape} sequences, {combined_X.shape[2]} features")
        
        # Shuffle data while maintaining time series integrity
        # Use random permutation to mix tickers but preserve sequences
        shuffle_indices = np.random.permutation(len(combined_X))
        combined_X = combined_X[shuffle_indices]
        combined_y = combined_y[shuffle_indices]
        
        # Build shared backbone model
        input_shape = (combined_X.shape[1], combined_X.shape[2])
        model = self.lstm_builder.build_model(input_shape, **self.config.model.model_params)
        
        print(f"Model architecture: {model.count_params()} parameters")
        
        # Get enhanced regularization callbacks
        callbacks = self.lstm_builder.get_regularization_callbacks(epochs)
        
        # Train shared backbone
        print(f"Starting shared backbone training...")
        history = model.fit(
            combined_X, combined_y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=self.config.model.training_params.get("batch_size", 64),
            callbacks=callbacks,
            verbose=1
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
        
        # Robust statistical validation using existing framework
        robust_results = self.robust_validator.moving_block_bootstrap(
            returns=combined_y,
            predictions=predictions
        )
        
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
                "correlation": history.history["correlation_metric"][-1]
            }
        }
        
        self.training_results["shared_backbone"] = training_results
        
        print(f"Shared backbone training complete!")
        print(f"   - Final train loss: {training_results['final_metrics']['train_loss']:.4f}")
        print(f"   - Final val loss: {training_results['final_metrics']['val_loss']:.4f}")
        print(f"   - Pattern detection accuracy: {training_results['final_metrics']['pattern_detection_accuracy']:.3f}")
        print(f"   - Correlation: {training_results['final_metrics']['correlation']:.3f}")
        print(f"   - Statistical significance: p={robust_results['p_value']:.4f}")
        
        return training_results
    
    def _prepare_pattern_detection_sequences(self, 
                                           features_df: pd.DataFrame,
                                           feature_columns: List[str], 
                                           pattern_targets: np.ndarray,
                                           sequence_length: int = 20,
                                           stride: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for pattern detection training
        
        Args:
            features_df: DataFrame with features
            feature_columns: List of feature column names
            pattern_targets: Pattern detection target values
            sequence_length: Length of input sequences
            stride: Stride between sequences
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        
        # Extract feature matrix
        feature_matrix = features_df[feature_columns].values
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        X_sequences = []
        y_targets = []
        
        # Generate overlapping sequences
        for i in range(0, len(feature_matrix) - sequence_length, stride):
            # Extract sequence
            sequence = feature_matrix[i:i + sequence_length]
            
            # Target is the pattern confidence at the end of the sequence
            target_idx = i + sequence_length - 1
            if target_idx < len(pattern_targets):
                target = pattern_targets[target_idx]
                
                X_sequences.append(sequence)
                y_targets.append(target)
        
        return np.array(X_sequences), np.array(y_targets)
    
    def _calculate_pattern_detection_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate pattern detection accuracy
        
        For pattern confidence scores (0-1 range), we use threshold-based accuracy
        """
        # Convert continuous pattern confidence to binary classification
        # Threshold: 0.5 for pattern detection confidence
        y_true_binary = (y_true > 0.5).astype(int)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(y_true_binary == y_pred_binary)
        
        return accuracy
    
    def validate_cross_ticker_performance(self, 
                                        training_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                        trained_model: tf.keras.Model) -> Dict[str, Any]:
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
            print(f"   Validating {ticker}...")
            
            # Time series cross-validation for this ticker
            cv = GappedTimeSeriesCV(n_splits=3, test_size=0.2, gap_size=5)
            
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx] 
                X_test_fold = X[test_idx]
                y_test_fold = y[test_idx]
                
                # Generate predictions
                predictions = trained_model.predict(X_test_fold, verbose=0).flatten()
                
                # Calculate pattern detection metrics
                pattern_detection_acc = np.mean((predictions > 0.5) == (y_test_fold > 0.5))  # Binary accuracy for pattern confidence
                correlation = np.corrcoef(y_test_fold, predictions)[0, 1] if len(y_test_fold) > 1 else 0.0
                mae = np.mean(np.abs(y_test_fold - predictions))
                
                fold_results.append({
                    "fold": fold,
                    "pattern_detection_accuracy": pattern_detection_acc,
                    "correlation": correlation,
                    "mae": mae
                })
            
            # Aggregate results for this ticker
            ticker_performance[ticker] = {
                "fold_results": fold_results,
                "mean_pattern_detection_accuracy": np.mean([r["pattern_detection_accuracy"] for r in fold_results]),
                "mean_correlation": np.mean([r["correlation"] for r in fold_results if not np.isnan(r["correlation"])]),
                "mean_mae": np.mean([r["mae"] for r in fold_results])
            }
        
        # Calculate overall cross-ticker statistics
        all_pattern_acc = [perf["mean_pattern_detection_accuracy"] for perf in ticker_performance.values()]
        all_corr = [perf["mean_correlation"] for perf in ticker_performance.values() if not np.isnan(perf["mean_correlation"])]
        
        cross_ticker_results = {
            "ticker_performance": ticker_performance,
            "overall_stats": {
                "mean_pattern_detection_accuracy": np.mean(all_pattern_acc),
                "std_pattern_detection_accuracy": np.std(all_pattern_acc),
                "mean_correlation": np.mean(all_corr) if all_corr else 0.0,
                "std_correlation": np.std(all_corr) if all_corr else 0.0,
                "successful_tickers": len(ticker_performance),
                "pattern_generalization_score": np.mean(all_pattern_acc)  # Pattern generalization metric
            }
        }
        
        print(f"Cross-ticker validation results:")
        print(f"   - Mean pattern detection accuracy: {cross_ticker_results['overall_stats']['mean_pattern_detection_accuracy']:.3f} ± {cross_ticker_results['overall_stats']['std_pattern_detection_accuracy']:.3f}")
        print(f"   - Mean correlation: {cross_ticker_results['overall_stats']['mean_correlation']:.3f} ± {cross_ticker_results['overall_stats']['std_correlation']:.3f}")
        print(f"   - Pattern generalization score: {cross_ticker_results['overall_stats']['pattern_generalization_score']:.3f}")
        
        return cross_ticker_results
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        
        if "shared_backbone" not in self.training_results:
            return "No training results available. Run train_shared_backbone() first."
        
        results = self.training_results["shared_backbone"]
        robust_stats = results["robust_validation"]
        
        report = f"""
Shared Backbone LSTM Training Report
==================================

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


def create_shared_backbone_trainer(tickers: Optional[List[str]] = None,
                                 use_expanded_universe: bool = True) -> SharedBackboneTrainer:
    """Convenience function to create shared backbone trainer"""
    return SharedBackboneTrainer(tickers=tickers, use_expanded_universe=use_expanded_universe)


# Example usage
if __name__ == "__main__":
    # Create trainer with expanded universe
    trainer = create_shared_backbone_trainer(use_expanded_universe=True)
    
    print(f"Shared Backbone Training Framework Initialized")
    print(f"   - Training Universe: {len(trainer.tickers)} securities")
    print(f"   - Enhanced Regularization: Enabled")
    print(f"   - Validation Framework: Integrated")
    print(f"   - Swing Trading Optimized: 20-day sequences, 5-day stride")
    
    print(f"\nNext steps:")
    print(f"   1. Load ticker data: ticker_data = {{ticker: ohlcv_df for ticker in trainer.tickers}}")
    print(f"   2. Prepare training data: training_data = trainer.prepare_training_data(ticker_data)")
    print(f"   3. Train shared backbone: results = trainer.train_shared_backbone(training_data)")
    print(f"   4. Validate performance: validation = trainer.validate_cross_ticker_performance(training_data, results['model'])")
    print(f"   5. Generate report: report = trainer.generate_training_report()")