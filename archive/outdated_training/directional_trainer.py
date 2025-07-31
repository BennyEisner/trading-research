#!/usr/bin/env python3

"""
Directional-focused LSTM trainer for financial time series
Optimized for directional accuracy over magnitude accuracy
"""

import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(".")

import tensorflow as tf

# Import training infrastructure
from src.training.base_trainer import LSTMProductionTrainer
from src.models.multi_scale_lstm import MultiScaleLSTMBuilder
from src.models.directional_loss import DirectionalLoss
from config.config import load_config


class DirectionalEnhancedTrainer(LSTMProductionTrainer):
    """
    Enhanced trainer with directional loss focus
    """

    def __init__(self, config, config_path=None):
        super().__init__(config)
        
        # Load centralized configuration
        self.central_config = load_config(config_path) if config_path else load_config()
        self.directional_config = self._create_directional_config()

    def _create_directional_config(self):
        """Create configuration optimized for directional prediction from central config"""
        # Get base parameters from central config
        base_params = self.central_config.get_model_params()
        
        return {
            "directional_alpha": base_params.get("directional_alpha", 0.05),
            "learning_rate": base_params.get("learning_rate", 0.0005),
            "batch_size": base_params.get("batch_size", 64),
            "dropout_rate": base_params.get("dropout_rate", 0.4),
            "l2_regularization": base_params.get("l2_regularization", 0.005),
            "patience": base_params.get("patience", 5),
            "monitor_metric": base_params.get("monitor_metric", "val_directional_accuracy"),
            "early_stopping_mode": base_params.get("early_stopping_mode", "max"),
        }

    def build_directional_model(self, input_shape, **model_params):
        """Build model with directional loss and enhanced metrics"""

        # Merge directional config with model params
        enhanced_params = {**model_params, **self.directional_config}

        # Build multi-scale LSTM with directional focus
        builder = MultiScaleLSTMBuilder(self.config)
        model = builder.build_directional_focused_model(input_shape, **enhanced_params)

        return model

    def create_directional_callbacks(self):
        """Create callbacks optimized for directional prediction"""

        callbacks = [
            # Early stopping based on directional accuracy
            tf.keras.callbacks.EarlyStopping(
                monitor=self.directional_config["monitor_metric"],
                patience=self.directional_config["patience"],
                restore_best_weights=True,
                mode=self.directional_config["early_stopping_mode"],
                verbose=1,
            ),
            # Reduce learning rate when directional accuracy plateaus
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=self.directional_config["monitor_metric"],
                factor=0.5,
                patience=7,
                mode=self.directional_config["early_stopping_mode"],
                verbose=1,
                min_lr=1e-6,
            ),
            # Model checkpoint based on best directional accuracy
            tf.keras.callbacks.ModelCheckpoint(
                filepath="models/trained/best_directional_lstm_model.keras",
                monitor=self.directional_config["monitor_metric"],
                save_best_only=True,
                mode=self.directional_config["early_stopping_mode"],
                verbose=1,
            ),
            # CSV logger for detailed metrics tracking
            tf.keras.callbacks.CSVLogger("directional_training_log.csv", append=True),
        ]

        return callbacks

    def train_directional_model(self, data_splits):
        """Train model with directional focus"""

        print(f"TRAINING DIRECTIONAL-FOCUSED MULTI-SCALE LSTM")
        print(f"Configuration: directional_alpha={self.directional_config['directional_alpha']}")

        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = data_splits

        # Build directional model
        model = self.build_directional_model(input_shape=X_train.shape[1:], **self.config)

        print(f"Model built with {model.count_params():,} parameters")

        # Create directional callbacks
        callbacks = self.create_directional_callbacks()

        # Train with enhanced monitoring
        print(f"\nStarting directional training...")
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")

        history = model.fit(
            X_train,
            y_train,
            epochs=100,  # More epochs for directional learning
            batch_size=self.directional_config["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate directional performance
        print(f"\nDIRECTIONAL EVALUATION RESULTS")

        # Training set evaluation
        train_results = model.evaluate(X_train, y_train, verbose=0)
        print(f"\nTraining Performance:")
        for i, metric in enumerate(model.metrics_names):
            print(f"  {metric}: {train_results[i]:.4f}")

        # Validation set evaluation
        val_results = model.evaluate(X_val, y_val, verbose=0)
        print(f"\nValidation Performance:")
        for i, metric in enumerate(model.metrics_names):
            print(f"  {metric}: {val_results[i]:.4f}")

        # Test set evaluation
        test_results = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Performance:")
        for i, metric in enumerate(model.metrics_names):
            print(f"  {metric}: {test_results[i]:.4f}")

        # Additional directional analysis
        test_preds = model.predict(X_test, verbose=0)
        self._analyze_directional_performance(y_test, test_preds.flatten())

        return {
            "model": model,
            "history": history,
            "test_results": dict(zip(model.metrics_names, test_results)),
            "feature_names": feature_names,
        }

    def _analyze_directional_performance(self, y_true, y_pred):
        """Detailed directional performance analysis"""

        print(f"\nDETAILED DIRECTIONAL ANALYSIS")

        # Basic directional accuracy
        correct_direction = np.sign(y_true) == np.sign(y_pred)
        dir_accuracy = np.mean(correct_direction)
        print(f"Directional Accuracy: {dir_accuracy:.3f} ({100*dir_accuracy:.1f}%)")

        # Up vs Down accuracy
        up_mask = y_true > 0
        down_mask = y_true < 0

        up_accuracy = np.mean(correct_direction[up_mask]) if np.any(up_mask) else 0
        down_accuracy = np.mean(correct_direction[down_mask]) if np.any(down_mask) else 0

        print(f"Up Move Accuracy:     {up_accuracy:.3f} ({100*up_accuracy:.1f}%) - {np.sum(up_mask)} samples")
        print(f"Down Move Accuracy:   {down_accuracy:.3f} ({100*down_accuracy:.1f}%) - {np.sum(down_mask)} samples")

        # Magnitude-weighted accuracy
        weights = np.abs(y_true)
        weighted_accuracy = np.sum(correct_direction * weights) / np.sum(weights)
        print(f"Magnitude-Weighted Accuracy: {weighted_accuracy:.3f}")

        # Simulated trading performance
        trading_pnl = np.sum(np.sign(y_pred) * y_true)
        buy_hold_pnl = np.sum(y_true)  # Always long

        print(f"\nSimulated Trading P&L: {trading_pnl:.4f}")
        print(f"Buy & Hold P&L:        {buy_hold_pnl:.4f}")
        print(f"Strategy Advantage:    {trading_pnl - buy_hold_pnl:+.4f}")

        # Prediction distribution
        pred_up = np.sum(y_pred > 0)
        pred_down = np.sum(y_pred < 0)
        actual_up = np.sum(y_true > 0)
        actual_down = np.sum(y_true < 0)

        print(f"\nPrediction Distribution:")
        print(f"  Predicted Up:   {pred_up} ({100*pred_up/len(y_pred):.1f}%)")
        print(f"  Predicted Down: {pred_down} ({100*pred_down/len(y_pred):.1f}%)")
        print(f"  Actual Up:      {actual_up} ({100*actual_up/len(y_true):.1f}%)")
        print(f"  Actual Down:    {actual_down} ({100*actual_down/len(y_true):.1f}%)")


def main():
    """Run enhanced directional training"""

    print("ENHANCED DIRECTIONAL-FOCUSED LSTM TRAINING")
    print("=" * 60)

    # Enhanced configuration for directional training
    config = {
        "tickers": ["AAPL", "MSFT", "GOOG", "NVDA", "TSLA", "AMZN", "META"],
        "years_of_data": 20,
        "database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db",
        "lookback_window": 30,
        "target_features": 40,
        "features_per_category": 6,
        "random_seed": 42,
        # Enhanced model parameters for directional learning
        "lstm_units_1": 512,
        "lstm_units_2": 256,
        "lstm_units_3": 128,
        "use_attention": True,
        "dense_layers": [256, 128, 64],
    }

    print(f"Configuration: {json.dumps(config, indent=2)}")

    # Create directional trainer
    trainer = DirectionalEnhancedTrainer(config)

    try:
        # Prepare data (reuse existing data preparation)
        print(f"\nPREPARING TRAINING DATA")
        data_splits = trainer.prepare_comprehensive_training_data()

        # Train directional model
        results = trainer.train_directional_model(data_splits)

        print(f"\nDIRECTIONAL TRAINING COMPLETED SUCCESSFULLY!")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"directional_lstm_results_{timestamp}.json"

        # Extract serializable results
        serializable_results = {
            "timestamp": timestamp,
            "config": config,
            "test_metrics": results["test_results"],
            "feature_count": len(results["feature_names"]),
            "model_parameters": results["model"].count_params(),
        }

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to: {results_file}")

        return results

    except Exception as e:
        print(f"\nDIRECTIONAL TRAINING FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
