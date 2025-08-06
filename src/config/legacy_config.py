#!/usr/bin/env python3
"""
Configuration management for ML training pipeline
"""

# Default configuration
DEFAULT_CONFIG = {
    "tickers": [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
    ],
    "lookback_window": 24,  # 6 months of weekly data
    "lstm_units": [128, 96, 64, 32],  # 4 layer deep LSTM
    "epochs": 500,
    "batch_size": 32,
    "learning_rate": 0.002,
    "early_stopping_patience": 80,
    "years_of_data": 15,
    "prediction_horizon": "weekly",
    "enable_cross_validation": False,
    "enable_ensemble": True,  # Multiple model ensemble
    "enable_advanced_features": True,  # Complex feature engineering
    "enable_attention": True,  # Attention mechanism
    "log_file": "overnight_training_log.json",
    "database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db",
    "train_ratio": 0.7,
    "validation_ratio": 0.15,
    "cv_folds": 3,
}


class Config:
    """Configuration class with validation and type checking"""

    def __init__(self, custom_config=None):
        self.config = DEFAULT_CONFIG.copy()
        if custom_config:
            self.config.update(custom_config)
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.config["lookback_window"] > 0, "Lookback window must be positive"
        assert self.config["epochs"] > 0, "Epochs must be positive"
        assert self.config["batch_size"] > 0, "Batch size must be positive"
        assert 0 < self.config["learning_rate"] < 1, "Learning rate must be between 0 and 1"
        assert len(self.config["tickers"]) > 0, "Must have at least one ticker"
        assert self.config["prediction_horizon"] in [
            "daily",
            "weekly",
        ], "Invalid prediction horizon"

        # Validate split ratios
        total_ratio = self.config["train_ratio"] + self.config["validation_ratio"]
        assert total_ratio < 1.0, "Train and validation ratios must sum to less than 1"

    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def update(self, updates):
        """Update configuration with new values"""
        self.config.update(updates)
        self._validate_config()

    def to_dict(self):
        """Return configuration as dictionary"""
        return self.config.copy()

    def save_to_file(self, filepath):
        """Save configuration to JSON file"""
        import json

        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, filepath):
        """Load configuration from JSON file"""
        import json

        with open(filepath, "r") as f:
            custom_config = json.load(f)
        return cls(custom_config)
