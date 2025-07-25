#!/usr/bin/env python3

"""
Simple training script to test refactored /src infrastructure
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_loader import DataLoader
from src.models.multi_scale_lstm import MultiScaleLSTMBuilder
from src.validation.pipeline_validator import PipelineValidator


def create_training_config():
    """Create training config"""
    return {
        # Data config
        "database_url": "sqlite:///./returns.db",
        "tickers": ["AAPL", "MSFT"],
        "years_of_data": 2,
        # Model Architecture
        "lookback_window": 30,
        "lstm_units_1": 256,
        "lstm_units_2": 128,
        "lstm_units_3": 64,
        "dropout_rate": 0.3,
        "l2_regularization": 0.003,
        "use_attention": True,
        "dense_layers": [128, 64],
        # Training Params
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "directional_alpha": 0.4,
        # Feature engineering
        "target_features": 20,
        "prediction_horizon": "daily",
    }


def load_and_validate_data(config):
    """Load data and run validation checks"""
    print("LOADING DATA")

    # Initialize components
    data_loader = DataLoader(config)
    validator = PipelineValidator()

    try:
        print(f"Loading data for tickers: {config['tickers']}")
        X, y, num_features = data_loader.load_multi_ticker_data(
            tickers=config["tickers"], years=config["years_of_data"]
        )
        print(f"Loaded sequences: {X.shape}")
        print(f"Targets: {y.shape}")
        print(f"Feature per timestamp: {num_features}")

        # Validate the sequences
        print("\nVALIDATING DATA")
        is_valid, issues = validator.validate_sequences(X, y)

        if not is_valid:
            print(f"Data validation failed {issues}")
            return None, None, None
        print("Data validation passes")
        return X, y, num_features

    except Exception as e:
        print(f"Data loading failed: {e}")
        return None, None, None


def train_model(X, y, config):
    """Build and train model"""
    print("\n BUILDING MODEL")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Building Model
    builder = MultiScaleLSTMBuilder(config)
    model = builder.build_directional_focused_model(input_shape=X_train.shape[1:], **config)  # (timestamps, features)
    print(f"Model built with {model.count_params():,} parameters")

    # Train model
    print("\n TRAINING MODEL")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=1,
    )

    # Evaluate model
    print("\n EVALUATING MODEL")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    for i, metric in enumerate(model.metrics_names):
        print(f"{metric}: {test_results[i]:.4f}")

    return model, history, (X_test, y_test)


def save_model_and_results(model, history, config):
    """Save trained model in .keras format with metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create models directory
    models_dir = Path("models/trained")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model in .keras format
    model_name = f"simple_lstm_{timestamp}.keras"
    model_path = models_dir / model_name

    print("\n SAVING MODEL")
    model.save(str(model_path))
    print(f"Model saved: {model_path}")

    metadata = {
        "timestamp": timestamp,
        "config": config,
        "model_path": str(model_path),
        "model_parameters": model.count_params(),
        "training_epochs": len(history.history["loss"]),
        "final_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1] if "val_loss" in history.history else None,
    }

    metadata_path = models_dir / f"metadata_{timestamp}.json"
    import json

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {metadata_path}")
    return model_path


def main():
    print("SIMPLE LSTM TRAINING PIPELINE")
    print("=" * 50)

    # Load configuration
    config = create_training_config()
    print(f"Config: {len(config)}")

    # Load and validate data
    X, y, num_features = load_and_validate_data(config)
    if X is None:
        print("Training failed at data loading stage")
        return

    # Train model
    try:
        model, history, test_data = train_model(X, y, config)
        model_path = save_model_and_results(model, history, config)

        print(f"\n Training completed successfully!")
        print(f"Model saved: {model_path}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
