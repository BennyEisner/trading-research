#!/usr/bin/env python3

"""
RSI Mean Reversion Strategy Implementation
"""

from typing import List

import numpy as np
import pandas as pd
from pydantic import Field

from ..base import BaseStrategy, StrategyConfig


class RSIStrategyConfig(StrategyConfig):
    """Configuration for RSI Mean Reversion"""

    name: str = "rsi_mean_reversion"

    # RSI Parameters
    rsi_period: int = Field(default=14, ge=2, le=50)
    oversold_threshold: float = Field(default=30.0, ge=0.0, le=100.0)
    overbought_threshold: float = Field(default=70.0, ge=0.0, le=100.0)

    # Mean reversion parameters
    neutral_zone_lower: float = Field(default=40.0, ge=0.0, le=100.0)
    neutral_zone_upper: float = Field(default=60.0, ge=0.0, le=100.0)

    exit_on_neutral: bool = Field(default=True)
    partial_exit_threshold: float = Field(default=0.5, ge=0.1, le=0.9)

    # Risk Management
    max_holding_period: int = Field(default=10, ge=1, le=50)
    volatility_adjustment: bool = Field(default=True)


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy

    Logic:
        - Long when RSI < oversold_threshold
        - Short when RSI > overbought_threshold
        - Exit when RSI returns to neutral zone
        - Position size based on how extreme RSI is
    """

    def __init__(self, config: RSIStrategyConfig):
        super().__init__(config)
        self.config: RSIStrategyConfig = config

    def get_required_features(self) -> List[str]:
        """Required features for RSI Strategy"""
        return super().get_required_features() + [f"rsi_{self.config.rsi_period}", "rsi", "atr", "close"]

    def validate_parameters(self) -> bool:
        if self.config.oversold_threshold >= self.config.neutral_zone_lower:
            raise ValueError("Oversold threshold should be below neutral zone lower")
        if self.config.overbought_threshold <= self.config.neutral_zone_upper:
            raise ValueError("Overbought threshold should be above neutral zone upper")
        if self.config.neutral_zone_upper <= self.config.neutral_zone_lower:
            raise ValueError("Neutral zone lower must be less than upper")

        return True

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI Mean Reversion Signals"""

        signals = pd.DataFrame(index=data.index)

        rsi_col = f"rsi_{self.config.rsi_period}"
        if rsi_col not in data.columns:
            rsi_col = "rsi"

        if rsi_col not in data.columns:
            raise ValueError(f"RSI columns '{rsi_col}' not found in data")

        rsi = data[rsi_col]

        # Initialize signals
        signals["position"] = 0.0
        signals["entry_price"] = np.nan
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan

        # Generate entry signals (handle NaN values)
        oversold_condition = (rsi < self.config.oversold_threshold).fillna(False)
        overbought_condition = (rsi > self.config.overbought_threshold).fillna(False)

        # Long signals (buy oversold)
        long_entries = oversold_condition & ~oversold_condition.shift(1).fillna(False)
        signals.loc[long_entries, "position"] = 1.0
        signals.loc[long_entries, "entry_price"] = data.loc[long_entries, "close"]

        # Short signals (sell overbought)
        short_entries = overbought_condition & ~overbought_condition.shift(1).fillna(False)
        signals.loc[short_entries, "position"] = -1.0
        signals.loc[short_entries, "entry_price"] = data.loc[short_entries, "close"]

        # Generate exit signals
        if self.config.exit_on_neutral:
            # Exit longs when RSI reverts back to neutral zone (handle NaN values)
            long_exits = (
                (rsi > self.config.neutral_zone_lower) & (rsi.shift(1) <= self.config.neutral_zone_lower)
            ).fillna(False)

            # Exit shorts when RSI reverts back to neutral zone (handle NaN values)
            short_exits = (
                (rsi < self.config.neutral_zone_upper) & (rsi.shift(1) >= self.config.neutral_zone_upper)
            ).fillna(False)

            # Apply exits (SIMPLIFIED VERSION FOR DEVELOPMENT STAGE)
            signals.loc[long_exits, "position"] = 0.0
            signals.loc[short_exits, "position"] = 0.0

        # Stop loss and take profit levels
        if "atr" in data.columns:

            atr = data["atr"]

            signals.loc[long_entries, "stop_loss"] = data.loc[long_entries, "close"] - 2 * atr[long_entries]
            signals.loc[long_entries, "take_profit"] = data.loc[long_entries, "close"] + 3 * atr[long_entries]

            signals.loc[short_entries, "stop_loss"] = data.loc[short_entries, "close"] + 2 * atr[short_entries]
            signals.loc[short_entries, "take_profit"] = data.loc[short_entries, "close"] - 3 * atr[short_entries]

        return signals

    def calculate_signal_strength(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """Calculate signal strength based on RSI extremity"""

        rsi_col = f"rsi_{self.config.rsi_period}"
        if rsi_col not in data.columns:
            rsi_col = "rsi"

        rsi = data[rsi_col]
        strength = pd.Series(0.0, index=data.index)

        # Long signal strength
        long_mask = signals["position"] > 0
        if long_mask.any():
            # Strength  increases as RSI gets more oversold
            long_strength = (self.config.oversold_threshold - rsi) / self.config.oversold_threshold
            strength[long_mask] = long_strength[long_mask].clip(0, 1)

        # Short signal strength
        short_mask = signals["position"] < 0
        if short_mask.any():
            # Strength  increases as RSI gets more overbought
            short_strength = (rsi - self.config.overbought_threshold) / (100 - self.config.overbought_threshold)
            strength[short_mask] = short_strength[short_mask].clip(0, 1)

        # Volatility Adjustment
        if self.config.volatility_adjustment and "atr" in data.columns:
            # Reduce signal strength for increased volatility
            atr_normalized = data["atr"] / data["close"]
            rolling_mean = atr_normalized.rolling(20, min_periods=1).mean()
            vol_ratio = (atr_normalized / rolling_mean).clip(0, 2)  # Cap at 2x mean
            vol_adjustment = 1 - (vol_ratio - 1).clip(0, 1)  # Reduce strength when vol > mean
            strength *= vol_adjustment

        return strength.fillna(0)

    def _calculate_rsi_extremity(self, rsi_value: float) -> float:
        """Calculate how extreme an RSI value is (0 to 1)"""

        if rsi_value <= self.config.oversold_threshold:
            return (self.config.oversold_threshold - rsi_value) / self.config.oversold_threshold

        elif rsi_value >= self.config.overbought_threshold:
            return (rsi_value - self.config.overbought_threshold) / (100 - self.config.overbought_threshold)

        else:
            return 0.0

    def get_strategy_description(self) -> str:
        """Get human-readable strategy description"""
        return (
            f"RSI Mean Reversion Strategy: "
            f"Long when RSI < {self.config.oversold_threshold}, "
            f"Short when RSI > {self.config.overbought_threshold}, "
            f"Exit in neutral zone [{self.config.neutral_zone_lower}-{self.config.neutral_zone_upper}]"
        )
