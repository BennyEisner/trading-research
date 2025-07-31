#!/usr/bin/env python3

"""
Base LSTM Production Trainer
Foundation class for all LSTM training strategies
"""

import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(".")

# Import core infrastructure
from src.config.config import Config
from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer
from src.models.multi_scale_lstm import MultiScaleLSTMBuilder
from src.utils.logging_utils import setup_production_logger


class LSTMProductionTrainer:
    """
    Base trainer class providing core functionality for LSTM model training
    All specialized trainers inherit from this class
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_production_logger({"log_file": "training.log"})
        self.training_results = {}

    def prepare_comprehensive_training_data(self) -> Dict[str, Any]:
        """
        Prepare comprehensive training data with advanced feature engineering
        Returns data splits and metadata
        """
        print("Preparing comprehensive training data...")

        config_obj = Config(self.config)
        data_loader = DataLoader(config_obj)

        tickers = self.config.get("tickers", ["AAPL"])
        years_of_data = self.config.get("years_of_data", 2)
        lookback_window = self.config.get("lookback_window", 30)

        # Load all ticker data first for cross-sectional features
        all_ticker_data = {}
        for ticker in tickers:
            try:
                ticker_raw_data = data_loader.load_single_ticker_data(ticker, years_of_data)
                all_ticker_data[ticker] = ticker_raw_data
            except Exception as e:
                print(f"⚠️  Could not load data for {ticker}: {e}")
                continue

        ticker_data = {}
        all_features = []
        all_targets = []

        for ticker in tickers:
            try:
                # Use already loaded data
                raw_data = all_ticker_data[ticker]

                if len(raw_data) < lookback_window + 50:  # Need minimum data
                    print(f"⚠️  Insufficient data for {ticker}, skipping...")
                    continue

                # Feature engineering with proper market data context
                feature_engineer = FeatureEngineer(
                    symbol=ticker,
                    market_data=None,  # Could add SPY data here
                    mag7_data=None,    # Could add MAG7 data here  
                    all_ticker_data=all_ticker_data
                )
                features_df = feature_engineer.create_comprehensive_features(raw_data)

                # Single-horizon target creation (efficient approach)
                primary_horizon = self.config.get("prediction_horizon", 1)  # Single horizon for efficiency
                
                # Calculate target for primary horizon only (no waste)
                horizon_return = features_df["close"].pct_change(primary_horizon).shift(-primary_horizon)
                features_df["target"] = horizon_return
                
                print(f"  Using {primary_horizon}-day returns as target (efficient single-horizon approach)")

                # Remove NaN rows
                features_df = features_df.dropna()

                # Store ticker data
                ticker_data[ticker] = features_df

                # Prepare sequences for this ticker
                feature_columns = [col for col in features_df.columns if col not in ["date", "target", "ticker"]]

                # Create sequences
                for i in range(lookback_window, len(features_df)):
                    sequence = features_df[feature_columns].iloc[i - lookback_window : i].values
                    target = features_df["target"].iloc[i]

                    if not np.isnan(target):
                        all_features.append(sequence)
                        all_targets.append(target)

                print(f"✅ Processed {ticker}: {len(features_df)} records")

            except Exception as e:
                print(f"❌ Failed to process {ticker}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid training data could be prepared")

        # Convert to arrays
        X = np.array(all_features)
        y = np.array(all_targets)

        print(f"📊 Total sequences: {len(X)}")
        print(f"📐 Sequence shape: {X.shape}")

        # Create train/validation/test splits (temporal order preserved)
        n_samples = len(X)
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]

        # Create feature names
        if ticker_data:
            first_ticker_data = list(ticker_data.values())[0]
            feature_names = [col for col in first_ticker_data.columns if col not in ["date", "target", "ticker"]]
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[2])]

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
            "feature_names": feature_names,
            "ticker_data": ticker_data,
            "data_stats": {
                "total_samples": n_samples,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "n_features": X.shape[2],
                "sequence_length": X.shape[1],
            },
        }

    def build_model(self, input_shape: Tuple[int, int], **params) -> Any:
        """
        Build LSTM model using MultiScaleLSTMBuilder
        Override in subclasses for specialized architectures
        """
        builder = MultiScaleLSTMBuilder(self.config)
        return builder.build_multi_scale_model(input_shape, **params)

    def train_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Train model with prepared data
        Override in subclasses for specialized training
        """
        # Prepare data
        data_dict = self.prepare_comprehensive_training_data()

        # Extract data
        X_train, y_train = data_dict["train"]
        X_val, y_val = data_dict["val"]
        X_test, y_test = data_dict["test"]

        # Build model
        model = self.build_model(X_train.shape[1:], **self.config)

        # Compile model
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get("epochs", 50),
            batch_size=self.config.get("batch_size", 32),
            verbose=1,
        )

        # Create metadata
        metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "config": self.config,
            "data_stats": data_dict["data_stats"],
            "model_params": model.count_params(),
            "training_epochs": len(history.history["loss"]),
            "final_loss": history.history["loss"][-1],
            "final_val_loss": history.history["val_loss"][-1],
        }

        return model, history, metadata

