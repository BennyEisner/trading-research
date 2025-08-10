#!/usr/bin/env python3

"""
Pragmatic Configuration Management
Type-safe configuration using Pydantic for personal trading research
Requires Python 3.12+
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

if sys.version_info < (3, 12):
    raise RuntimeError("This application requires Python 3.12 or higher")

from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration"""

    url: str = Field(default="postgresql://trader:password@localhost:5432/trading_research", alias="DATABASE_URL")
    pool_size: int = Field(default=5)
    max_overflow: int = Field(default=10)
    echo: bool = Field(default=False)

    class Config:
        env_prefix = ""


class ModelConfig(BaseSettings):
    """Model training configuration"""

    # MAG7 core tickers for final specialization
    mag7_tickers: List[str] = Field(default=["AAPL", "MSFT", "GOOG", "NVDA", "TSLA", "AMZN", "META"])

    expanded_universe: List[str] = Field(
        default=[
            # MAG7
            "AAPL",
            "MSFT",
            "GOOG",
            "NVDA",
            "TSLA",
            "AMZN",
            "META",
            # Correlated mega-cap tech
            "CRM",
            "ADBE",
            "NOW",
            "ORCL",
            "NFLX",
            "AMD",
            "INTC",
            "CSCO",
            "AVGO",
            "TXN",
            "QCOM",
            "MU",
            "AMAT",
            "LRCX",
            "KLAC",
            "MRVL",
            "SNPS",
            "CDNS",
            "FTNT",
            "PANW",
            "INTU",
            "UBER",
            "ZM",
            "DDOG",
            # Market indicators
            "QQQ",
            "XLK",
            "SPY",
        ]
    )

    # Default to MAG7 for backward compatibility
    tickers: List[str] = Field(default=["AAPL", "MSFT", "GOOG", "NVDA", "TSLA", "AMZN", "META"])

    lookback_window: int = Field(default=20)
    
    # Stride Configuration for Overlap Control
    training_stride: int = Field(default=5)  # Training sequence stride (was hardcoded to 1)
    validation_stride: int = Field(default=20)  # Validation sequence stride (ensures 0% overlap)
    
    prediction_horizon: int = Field(default=3)  # Optimized for 1-10 day range
    
    # Out-of-Sample Testing Configuration
    out_of_sample_enabled: bool = Field(default=True)
    out_of_sample_gap_months: int = Field(default=6)  # 6-month gap for out-of-sample testing
    temporal_validation_split: float = Field(default=0.8)  # 80% train, 20% validation by time
    
    # Enhanced Leakage Detection
    correlation_monitoring_enabled: bool = Field(default=True)
    early_epoch_correlation_threshold: float = Field(default=0.10)  # Alert if correlation > 10% in early epochs
    leakage_detection_epochs: int = Field(default=3)  # Monitor first 3 epochs for leakage

    target_features: int = Field(default=24)
    random_seed: int = Field(default=42)

    model_size: str = Field(default="medium-large")

    # Model architecture dynamically set based on model_size
    model_params: Dict[str, Any] = Field(default_factory=lambda: {})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_params = self._get_model_params_for_size(self.model_size)

    def _get_model_params_for_size(self, size: str) -> Dict[str, Any]:
        """Get model parameters optimized for specific parameter count targets"""

        base_params = {
            "dropout_rate": 0.45,  # Reduced from 0.55 due to lower sequence overlap
            "l2_regularization": 0.01,  # Increased for overfitting prevention
            "directional_alpha": 0.05,
            "use_batch_norm": True,  
            "use_recurrent_dropout": True,  # Add recurrent dropout
            "learning_rate": 0.0005,  # Conservative for high-overlap training
        }

        if size == "small":  # Target: < 10K parameters
            return {
                **base_params,
                "lstm_units_1": 32,
                "lstm_units_2": 16,
                "dense_units": 8,
                "use_attention": False,
                "model_type": "shared_backbone",
                "adaptation_units": 8,
                "finetune_learning_rate": 0.0002,  
            }
        elif size == "medium-large":  # Target: ~52K parameters for 1-day stride
            return {
                **base_params,
                "lstm_units_1": 80,  # Increased capacity
                "lstm_units_2": 40,
                "dense_units": 20,
                "use_attention": False,
                "model_type": "shared_backbone",
                "adaptation_units": 12,
                "finetune_learning_rate": 0.0002,
            }
        else:  # medium (default)
            return {
                **base_params,
                "lstm_units_1": 64,
                "lstm_units_2": 32,
                "dense_units": 16,
                "use_attention": False,
                "model_type": "shared_backbone",
                "adaptation_units": 8,
                "finetune_learning_rate": 0.0002,
            }

    # Training parameters optimized for 95% overlap sequences
    training_params: Dict[str, Any] = Field(
        default={
            "batch_size": 256,  # Large batch size for gradient stability
            "epochs": 150,
            "learning_rate": 0.0005,  # Conservative for high overlap
            "patience": 15,  # Higher patience for validation monitoring
            "validation_split": 0.2,
            "monitor_metric": "val_loss",  # Focus on validation loss due to overlap
            "early_stopping_mode": "min",  # Minimize validation loss
            "reduce_lr_patience": 8,  # ReduceLROnPlateau patience
            "reduce_lr_factor": 0.5,
            "min_lr": 1e-6,
        }
    )


class BacktestConfig(BaseSettings):
    """Backtesting configuration"""

    # Execution parameters
    transaction_cost: float = Field(default=0.001)  # 10 bps round-trip
    min_position_size: float = Field(default=0.01)  # 1% minimum
    max_position_size: float = Field(default=0.10)  # 10% maximum

    # Risk management
    max_drawdown_limit: float = Field(default=0.15)  # 15% max drawdown
    volatility_target: float = Field(default=0.12)  # 12% annual vol target

    # Walk-forward parameters
    training_window_days: int = Field(default=252)  # 1 year training
    rebalance_frequency_days: int = Field(default=5)  # Weekly rebalance
    walk_forward_window_days: int = Field(default=30)  # Monthly walk-forward


class APIConfig(BaseSettings):
    """API server configuration"""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # Model serving
    model_cache_size: int = Field(default=5)
    prediction_timeout: int = Field(default=30)


class TradingConfig(BaseSettings):
    """Master configuration combining all components"""

    # Sub-onfigurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Global settings
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    data_directory: str = Field(default="./data")
    model_directory: str = Field(default="./models")

    @classmethod
    def from_yaml(cls, config_path: str) -> "TradingConfig":
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        config_data = self.dict()

        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

    def get_database_url(self) -> str:
        """Get database URL for SQLAlchemy"""
        return self.database.url

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for training"""
        return {**self.model.model_params, **self.model.training_params}


# Global configuration instance
config = TradingConfig()


def load_config(config_path: str = None) -> TradingConfig:
    """Load configuration from file or use defaults"""
    if config_path:
        return TradingConfig.from_yaml(config_path)
    else:
        return TradingConfig()


def get_config() -> TradingConfig:
    """Get the global configuration instance"""
    return config
