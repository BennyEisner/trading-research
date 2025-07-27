#!/usr/bin/env python3

"""
MACD Momentum Strategy
"""

from typing import List

import numpy as np
import pandas as pd
from pydantic import Field

from ..base import BaseStrategy, StrategyConfig


class MACDStrategyConfig(StrategyConfig):
    """Configuration for MACD Momentum Strategy"""

    name: str = "macd_momentum_strategy"

    # MACD Parameters
    fast_period: int = Field(default=12, ge=2, le=26)
    slow_period: int = Field(default=26, ge=4, le=52)
    signal_period: int = Field(default=9)

    # Signal Threshold
    signal_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    histogram_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Momentum Parameters
    momentum_confirmation: bool = Field(default=True)
    divergence_detection: bool = Field(default=False)

    # Exit Conditions
    exit_on_opposite_signal: bool = Field(default=True)
    exit_on_signal_line_cross: bool = Field(default=True)

    # Risk Management
    max_holding_period: int = Field(default=20, ge=1, le=100)
    volatility_adjustment: bool = Field(default=True)


class MACDMomentumStrategy(BaseStrategy):
    """MACD Momentum Strategy

    Logic:
        - Long when MACD line crosses ABOVE signal line (bullish crossover)
        - Short when MACD line crosses BELOW signal line (bearish crossover)
        - Additional confirmation from histogram
        - Position size based on histogram strength
    """

    def __init__(self, config: MACDStrategyConfig):
        super().__init__(config)
        self.config: MACDStrategyConfig = config

    def get_required_features(self) -> List[str]:
        """Required features for MACD Strategy"""
        return super().get_required_features() + ["macd", "macd_signal", "macd_histogram", "close", "atr"]

    def validate_parameters(self) -> bool:
        if self.config.fast_period >= self.config.slow_period:
            raise ValueError("Fast period must be smaller than slow period")
        if self.config.signal_period <= 0:
            raise ValueError("Signal Period must be positive")

        return True

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD Momentum Signals"""

        signals = pd.DataFrame(index=data.index)

        # Check for required MACD columns
        required_cols = ["macd", "macd_signal", "macd_histogram"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Required MACD columns not found in data: {missing_cols}")

        macd = data["macd"]
        macd_signal = data["macd_signal"]
        macd_histogram = data["macd_histogram"]

        # Initialize signals
        signals["position"] = 0.0
        signals["entry_price"] = np.nan
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan

        # Detect crossovers
        bullish_cross = self._detect_bullish_crossover(macd, macd_signal)
        bearish_cross = self._detect_bearish_crossover(macd, macd_signal)

        # Apply momentum confirmation if enabled
        if self.config.momentum_confirmation:
            bullish_cross = bullish_cross & (macd_histogram > self.config.histogram_threshold)
            bearish_cross = bearish_cross & (macd_histogram < -self.config.histogram_threshold)

        # Long signals (bullish crossover)
        long_entries = bullish_cross
        signals.loc[long_entries, "position"] = 1.0
        signals.loc[long_entries, "entry_price"] = data.loc[long_entries, "close"]

        # Short signals (bearish crossover)
        short_entries = bearish_cross
        signals.loc[short_entries, "position"] = -1.0
        signals.loc[short_entries, "entry_price"] = data.loc[short_entries, "close"]

        # Generate exit signals
        if self.config.exit_on_opposite_signal:
            # Exit longs on bearish crossover, exit shorts on bullish crossover
            signals.loc[bearish_cross, "position"] = 0.0
            signals.loc[bullish_cross, "position"] = 0.0

        # Stop loss and take profit levels
        if "atr" in data.columns:
            atr = data["atr"]

            # Long positions
            signals.loc[long_entries, "stop_loss"] = data.loc[long_entries, "close"] - 2 * atr[long_entries]
            signals.loc[long_entries, "take_profit"] = data.loc[long_entries, "close"] + 3 * atr[long_entries]

            # Short positions
            signals.loc[short_entries, "stop_loss"] = data.loc[short_entries, "close"] + 2 * atr[short_entries]
            signals.loc[short_entries, "take_profit"] = data.loc[short_entries, "close"] - 3 * atr[short_entries]

        return signals

    def calculate_signal_strength(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """Calculate signal strength based on MACD histogram magnitude"""

        if "macd_histogram" not in data.columns:
            return pd.Series(0.0, index=data.index)

        macd_histogram = data["macd_histogram"]
        strength = pd.Series(0.0, index=data.index)

        # Calculate strength based on histogram magnitude
        # Normalize by rolling standard deviation for relative strength
        hist_std = macd_histogram.rolling(20, min_periods=1).std()
        hist_std = hist_std.replace(0, 1)  # Avoid division by zero

        # Long signal strength
        long_mask = signals["position"] > 0
        if long_mask.any():
            long_strength = (macd_histogram / hist_std).clip(0, 3) / 3  # Normalize to 0-1
            strength[long_mask] = long_strength[long_mask]

        # Short signal strength
        short_mask = signals["position"] < 0
        if short_mask.any():
            short_strength = (-macd_histogram / hist_std).clip(0, 3) / 3  # Normalize to 0-1
            strength[short_mask] = short_strength[short_mask]

        # Volatility adjustment
        if self.config.volatility_adjustment and "atr" in data.columns:
            atr_normalized = data["atr"] / data["close"]
            rolling_mean = atr_normalized.rolling(20, min_periods=1).mean()
            vol_ratio = (atr_normalized / rolling_mean).clip(0, 2)
            vol_adjustment = 1 - (vol_ratio - 1).clip(0, 1)
            strength *= vol_adjustment

        return strength.fillna(0).clip(0, 1)

    def _detect_bullish_crossover(self, macd: pd.Series, macd_signal: pd.Series) -> pd.Series:
        """Detect bullish crossover (MACD crosses above signal)"""
        current_above = macd > macd_signal + self.config.signal_threshold
        prev_below = macd.shift(1, fill_value=False) <= macd_signal.shift(1, fill_value=False) + self.config.signal_threshold
        return (current_above & prev_below).fillna(False)

    def _detect_bearish_crossover(self, macd: pd.Series, macd_signal: pd.Series) -> pd.Series:
        """Detect bearish crossover (MACD crosses below signal)"""
        current_below = macd < macd_signal - self.config.signal_threshold
        prev_above = macd.shift(1, fill_value=False) >= macd_signal.shift(1, fill_value=False) - self.config.signal_threshold
        return (current_below & prev_above).fillna(False)

    def get_strategy_description(self) -> str:
        """Get human-readable strategy description"""
        return (
            f"MACD Momentum Strategy: "
            f"Fast={self.config.fast_period}, Slow={self.config.slow_period}, Signal={self.config.signal_period}, "
            f"Long on bullish crossover, Short on bearish crossover"
        )
