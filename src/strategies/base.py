#!/usr/bin/env python3

"""
Base classes for trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pydantic import BaseSettings, Field


class StrategyConfig(BaseSettings):
    """Base configuration for all strategies"""
    
    name: str
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Risk management
    max_position_size: float = Field(default=0.1, ge=0.0, le=1.0)
    stop_loss_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    take_profit_pct: Optional[float] = Field(default=None, ge=0.0)
    
    # Signal filtering
    min_signal_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    lookback_period: int = Field(default=20, ge=1)
    
    class Config:
        env_prefix = ""


@dataclass
class StrategySignal:
    """Standardized strategy signal output"""
    
    timestamp: pd.Timestamp
    position: float  # -1 to 1, where 0 is no position
    signal_strength: float  # 0 to 1, confidence in signal
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None 
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    
    All strategies must implement:
    1. generate_signals() - Core signal generation logic
    2. calculate_signal_strength() - Signal confidence calculation
    3. validate_parameters() - Parameter validation
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.validate_parameters()
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with columns:
            - position: -1 (short), 0 (no position), 1 (long)
            - signal_strength: 0.0 to 1.0 confidence score
            - entry_price: Suggested entry price (optional)
            - stop_loss: Stop loss price (optional)
            - take_profit: Take profit price (optional)
        """
        pass
    
    @abstractmethod
    def calculate_signal_strength(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength/confidence for each signal
        
        Args:
            data: Market data DataFrame
            signals: Raw signals DataFrame
            
        Returns:
            Series with signal strength values (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters
        
        Returns:
            True if parameters are valid
            
        Raises:
            ValueError if parameters are invalid
        """
        pass
    
    def get_required_features(self) -> List[str]:
        """
        Get list of required features/indicators for this strategy
        
        Returns:
            List of required column names
        """
        return ['open', 'high', 'low', 'close', 'volume']
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before signal generation (optional override)
        
        Args:
            data: Raw market data
            
        Returns:
            Preprocessed data
        """
        # Check for required features
        required_features = self.get_required_features()
        missing_features = [f for f in required_features if f not in data.columns]
        
        if missing_features:
            raise ValueError(f"Missing required features for {self.name}: {missing_features}")
        
        return data.copy()
    
    def postprocess_signals(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process signals after generation (optional override)
        
        Args:
            signals: Raw signals
            data: Market data
            
        Returns:
            Post-processed signals
        """
        processed = signals.copy()
        
        # Apply minimum signal strength filter
        if self.config.min_signal_strength > 0:
            weak_signals = processed['signal_strength'] < self.config.min_signal_strength
            processed.loc[weak_signals, 'position'] = 0
        
        # Apply position size limits
        processed['position'] = processed['position'].clip(-self.config.max_position_size, 
                                                          self.config.max_position_size)
        
        return processed
    
    def run_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Complete strategy execution pipeline
        
        Args:
            data: Market data with technical indicators
            
        Returns:
            DataFrame with processed signals
        """
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Generate raw signals
        raw_signals = self.generate_signals(processed_data)
        
        # Calculate signal strength
        raw_signals['signal_strength'] = self.calculate_signal_strength(processed_data, raw_signals)
        
        # Post-process signals
        final_signals = self.postprocess_signals(raw_signals, processed_data)
        
        # Add strategy metadata
        final_signals['strategy'] = self.name
        final_signals['timestamp'] = processed_data.index
        
        return final_signals
    
    def calculate_returns(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> pd.Series:
        """
        Calculate strategy returns based on signals and price data
        
        Args:
            signals: Strategy signals DataFrame
            price_data: Price data DataFrame
            
        Returns:
            Series of strategy returns
        """
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")
        
        # Calculate returns
        price_returns = price_data['close'].pct_change()
        strategy_returns = signals['position'].shift(1) * price_returns
        
        return strategy_returns.fillna(0)
    
    def update_performance_metrics(self, returns: pd.Series):
        """
        Update strategy performance metrics
        
        Args:
            returns: Strategy returns series
        """
        self.performance_metrics.update({
            'total_return': returns.sum(),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': (returns > 0).mean(),
            'total_signals': len(returns[returns != 0])
        })
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        return {
            'strategy_name': self.name,
            'enabled': self.config.enabled,
            'weight': self.config.weight,
            **self.performance_metrics
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.config.enabled})"