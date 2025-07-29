#!/usr/bin/env python3

"""
Pragmatic Configuration Management
Type-safe configuration using Pydantic for personal trading research
Requires Python 3.12+
"""

import sys
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path

# Verify Python version
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
    # Core parameters
    tickers: List[str] = Field(default=["AAPL", "MSFT", "GOOG", "NVDA", "TSLA", "AMZN", "META"])
    lookback_window: int = Field(default=30)
    prediction_horizon: int = Field(default=1)  # Primary prediction horizon in days (1=daily, 3=3-day, 4=4-day)
    # Note: Single horizon approach for efficiency. Use prediction_horizon=3 for 3-day returns, etc.
    target_features: int = Field(default=24)
    random_seed: int = Field(default=42)
    
    # Model architecture configurations by size
    model_size: str = Field(default="medium")  # Options: "small", "medium", "large"
    
    # Model architecture - dynamically set based on model_size
    model_params: Dict[str, Any] = Field(default_factory=lambda: {})
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_params = self._get_model_params_for_size(self.model_size)
    
    def _get_model_params_for_size(self, size: str) -> Dict[str, Any]:
        """Get model parameters optimized for specific parameter count targets"""
        
        base_params = {
            'dropout_rate': 0.3,
            'l2_regularization': 0.003,
            'directional_alpha': 0.05,
        }
        
        if size == "small":  # Target: ~500K-800K parameters
            return {
                **base_params,
                'lstm_units_1': 224,      # Increased to hit target range
                'lstm_units_2': 160,      # Increased to hit target range
                'lstm_units_3': 96,       # Keep current
                'use_attention': False,   # Remove attention (major param reduction)
                'dense_layers': [224, 128], # Larger dense layers for target range
                'model_type': 'simplified_lstm'  # Use simpler architecture
            }
        elif size == "medium":  # Target: ~1M-1.5M parameters  
            return {
                **base_params,
                'lstm_units_1': 320,      # Increased to hit target range
                'lstm_units_2': 192,      # Increased to hit target range
                'lstm_units_3': 96,       # Increased to hit target range
                'use_attention': True,    # Keep attention but optimized
                'dense_layers': [192, 96], # Larger dense layers for target range
                'attention_heads': 4,     # Fewer attention heads
                'model_type': 'efficient_multi_scale'
            }
        else:  # "large" - Original architecture
            return {
                **base_params,
                'lstm_units_1': 512,
                'lstm_units_2': 256,
                'lstm_units_3': 128,
                'use_attention': True,
                'dense_layers': [256, 128, 64],
                'model_type': 'full_multi_scale'
            }
    
    # Training parameters
    training_params: Dict[str, Any] = Field(default={
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.0005,
        'patience': 5,  # Fixed: reduced from 15 for faster convergence
        'validation_split': 0.2,
        'monitor_metric': 'val__directional_accuracy',  # Fixed: correct TensorFlow metric name (double underscore)
        'early_stopping_mode': 'max'
    })


class BacktestConfig(BaseSettings):
    """Backtesting configuration"""
    # Execution parameters
    transaction_cost: float = Field(default=0.001)  # 10 bps round-trip
    min_position_size: float = Field(default=0.01)  # 1% minimum
    max_position_size: float = Field(default=0.10)  # 10% maximum
    
    # Risk management
    max_drawdown_limit: float = Field(default=0.15)  # 15% max drawdown
    volatility_target: float = Field(default=0.12)   # 12% annual vol target
    
    # Walk-forward parameters
    training_window_days: int = Field(default=252)    # 1 year training
    rebalance_frequency_days: int = Field(default=5)  # Weekly rebalance
    walk_forward_window_days: int = Field(default=30) # Monthly walk-forward


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
    
    # Sub-configurations
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
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        config_data = self.dict()
        
        with open(config_path, 'w') as f:
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