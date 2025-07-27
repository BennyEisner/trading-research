#!/usr/bin/env python3

"""
Unit tests for RSI Mean Reversion Strategy
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ..indicators.rsi_strategy import RSIMeanReversionStrategy, RSIStrategyConfig


class TestRSIStrategyConfig:
    """Test RSI Strategy Configuration"""

    def test_default_config_values_are_valid(self):
        """Test that default configuration values are properly set"""
        # Arrange + Act
        config = RSIStrategyConfig()

        # Assert
        assert config.name == "rsi_mean_reversion"
        assert config.rsi_period == 14
        assert config.oversold_threshold == 30.0
        assert config.overbought_threshold == 70.0
        assert config.neutral_zone_lower == 40.0
        assert config.neutral_zone_upper == 60.0
        assert config.exit_on_neutral is True
        assert config.max_holding_period == 10
        assert config.volatility_adjustment is True

    def test_config_validation_with_valid_parameters(self):
        """Test configuration validation passes with valid parameters"""
        # Arrange
        config = RSIStrategyConfig(
            rsi_period=21,
            oversold_threshold=25.0,
            overbought_threshold=75.0,
            neutral_zone_lower=35.0,
            neutral_zone_upper=65.0,
        )

        # Act & Assert - Should not raise
        strategy = RSIMeanReversionStrategy(config)
        assert strategy.validate_parameters() is True

    def test_config_validation_fails_when_oversold_above_neutral_lower(self):
        """Test validation fails when oversold threshold >= neutral zone lower"""
        # Arrange
        config = RSIStrategyConfig(oversold_threshold=45.0, neutral_zone_lower=40.0)  # Above neutral_zone_lower (40.0)

        # Act & Assert
        with pytest.raises(ValueError, match="Oversold threshold should be below neutral zone lower"):
            RSIMeanReversionStrategy(config)

    def test_config_validation_fails_when_overbought_below_neutral_upper(self):
        """Test validation fails when overbought threshold <= neutral zone upper"""
        # Arrange
        config = RSIStrategyConfig(
            overbought_threshold=55.0, neutral_zone_upper=60.0  # Below neutral_zone_upper (60.0)
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Overbought threshold should be above neutral zone upper"):
            RSIMeanReversionStrategy(config)

    def test_config_validation_fails_when_neutral_zones_inverted(self):
        """Test validation fails when neutral zone upper <= lower"""
        # Arrange
        config = RSIStrategyConfig(neutral_zone_lower=65.0, neutral_zone_upper=60.0)  # Above neutral_zone_upper

        # Act & Assert
        with pytest.raises(ValueError, match="Neutral zone lower must be less than upper"):
            RSIMeanReversionStrategy(config)


class TestRSIMeanReversionStrategy:
    """Test RSI Mean Reversion Strategy Implementation"""

    @pytest.fixture
    def default_config(self):
        """Default RSI strategy configuration for tests"""
        return RSIStrategyConfig()

    @pytest.fixture
    def strategy(self, default_config):
        """RSI strategy instance with default config"""
        return RSIMeanReversionStrategy(default_config)

    @pytest.fixture
    def sample_data(self):
        """Sample market data with RSI values for testing"""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        # Create realistic price and RSI data
        close_prices = 100 + np.cumsum(np.random.randn(50) * 0.02)

        # Create RSI values that trigger various conditions (exactly 50 values)
        rsi_values = (
            [25, 20, 15, 35, 45]  # Oversold -> recovery (5)
            + [75, 80, 85, 65, 55]  # Overbought -> recovery (5)
            + list(np.random.uniform(40, 60, 20))  # Neutral zone (20)
            + [28, 32, 38, 42, 48]  # Oversold recovery (5)
            + list(np.random.uniform(45, 65, 15))  # More neutral (15)
        )

        atr_values = np.random.uniform(1.0, 3.0, 50)

        return pd.DataFrame(
            {"close": close_prices, "rsi_14": rsi_values, "rsi": rsi_values, "atr": atr_values},  # Fallback column
            index=dates,
        )

    def test_get_required_features_includes_rsi_and_base_features(self, strategy):
        """Test that required features include RSI and base OHLCV features"""
        # Act
        required_features = strategy.get_required_features()

        # Assert
        expected_base = ["open", "high", "low", "close", "volume"]
        expected_rsi = ["rsi_14", "rsi", "atr", "close"]

        for feature in expected_base + expected_rsi:
            assert feature in required_features

    def test_generate_signals_creates_correct_position_for_oversold(self, strategy, sample_data):
        """Test that long positions are generated when RSI is oversold"""
        # Arrange - Create data with clear oversold condition
        oversold_data = sample_data.copy()
        oversold_data.loc[oversold_data.index[5:8], "rsi_14"] = [25, 20, 35]  # Oversold then recovery

        # Act
        signals = strategy.generate_signals(oversold_data)

        # Assert
        oversold_entries = signals[signals["position"] > 0]
        assert len(oversold_entries) > 0, "Should generate long positions for oversold conditions"

        # Check that positions are only 1.0 or 0.0 (no fractional positions in basic implementation)
        assert all(pos in [0.0, 1.0] for pos in signals["position"].unique())

    def test_generate_signals_creates_correct_position_for_overbought(self, strategy, sample_data):
        """Test that short positions are generated when RSI is overbought"""


        #Create data with clear overbought condition, no oversold conditions
        overbought_data = sample_data.copy()
        overbought_data.loc[overbought_data.index[10:13], "rsi_14"] = [75, 80, 65]  # Overbought then recovery

        # no oversold conditions exist by setting all other RSI values to neutral zone
        mask = ~overbought_data.index.isin(overbought_data.index[10:13])
        overbought_data.loc[mask, "rsi_14"] = 50.0  # Neutral zone
        overbought_data.loc[mask, "rsi"] = 50.0

        # Act
        signals = strategy.generate_signals(overbought_data)

        # Assert
        overbought_entries = signals[signals["position"] < 0]
        assert len(overbought_entries) > 0, "Should generate short positions for overbought conditions"

        # Check that positions are only -1.0 or 0.0
        assert all(pos in [0.0, -1.0] for pos in signals["position"].unique())

    def test_generate_signals_sets_entry_prices_correctly(self, strategy, sample_data):
        """Test that entry prices are set to close prices when signals are generated"""
        # Act
        signals = strategy.generate_signals(sample_data)

        # Assert
        entry_signals = signals[signals["position"] != 0]
        entry_signals_with_price = entry_signals.dropna(subset=["entry_price"])

        if len(entry_signals_with_price) > 0:
            # Entry prices should match close prices where positions are taken
            for idx in entry_signals_with_price.index:
                expected_price = sample_data.loc[idx, "close"]
                actual_price = signals.loc[idx, "entry_price"]
                assert abs(actual_price - expected_price) < 1e-6, f"Entry price should match close price at {idx}"

    def test_generate_signals_sets_stop_loss_and_take_profit_when_atr_available(self, strategy, sample_data):
        """Test that stop loss and take profit are set based on ATR"""
        # Act
        signals = strategy.generate_signals(sample_data)

        # Assert
        entry_signals = signals[signals["position"] != 0]

        for idx in entry_signals.index:
            position = signals.loc[idx, "position"]
            close_price = sample_data.loc[idx, "close"]
            atr = sample_data.loc[idx, "atr"]

            if position > 0:  
                expected_stop_loss = close_price - 2 * atr
                expected_take_profit = close_price + 3 * atr
            else:  
                expected_stop_loss = close_price + 2 * atr
                expected_take_profit = close_price - 3 * atr

            if not pd.isna(signals.loc[idx, "stop_loss"]):
                assert abs(signals.loc[idx, "stop_loss"] - expected_stop_loss) < 1e-6
            if not pd.isna(signals.loc[idx, "take_profit"]):
                assert abs(signals.loc[idx, "take_profit"] - expected_take_profit) < 1e-6

    def test_generate_signals_handles_missing_rsi_column_gracefully(self, strategy):
        """Test strategy handles missing RSI columns by trying fallback"""
        # Arrange data Data without rsi_14 but withRSI 
        data_with_fallback = pd.DataFrame(
            {"close": [100, 101, 102], "rsi": [25, 30, 35], "atr": [1.5, 1.6, 1.4]},  # Only fallback column
            index=pd.date_range("2023-01-01", periods=3),
        )

        # Act  + Assert - Should not raise error
        signals = strategy.generate_signals(data_with_fallback)
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == 3

    def test_generate_signals_raises_error_when_no_rsi_columns_available(self, strategy):
        """Test strategy raises error when neither RSI column is available"""
        # Arrange - Data without any RSI columns
        data_without_rsi = pd.DataFrame(
            {"close": [100, 101, 102], "atr": [1.5, 1.6, 1.4]}, index=pd.date_range("2023-01-01", periods=3)
        )

        # Act + Assert
        with pytest.raises(ValueError, match="RSI columns .* not found in data"):
            strategy.generate_signals(data_without_rsi)

    def test_calculate_signal_strength_returns_higher_strength_for_extreme_rsi(self, strategy, sample_data):
        """Test signal strength increases as RSI becomes more extreme"""
        #  Create signals with different RSI extremity levels
        test_data = pd.DataFrame(
            {
                "close": [100, 100, 100, 100],
                "rsi_14": [15, 25, 75, 85],  # Very oversold, mildly oversold, mildly overbought, very overbought
                "atr": [1.0, 1.0, 1.0, 1.0],
            },
            index=pd.date_range("2023-01-01", periods=4),
        )

        signals = pd.DataFrame(
            {"position": [1.0, 1.0, -1.0, -1.0]}, index=test_data.index  # Long for oversold, short for overbought
        )

        # Act
        strength = strategy.calculate_signal_strength(test_data, signals)

        # Assert
        assert (
            strength.iloc[0] > strength.iloc[1]
        ), "Very oversold (15) should have higher strength than mildly oversold (25)"
        assert (
            strength.iloc[3] > strength.iloc[2]
        ), "Very overbought (85) should have higher strength than mildly overbought (75)"
        assert all(0 <= s <= 1 for s in strength), "All signal strengths should be between 0 and 1"

    def test_calculate_signal_strength_returns_zero_for_no_position(self, strategy, sample_data):
        """Test signal strength is zero when no position is taken"""
        # Arrange
        signals = pd.DataFrame({"position": [0.0, 0.0, 0.0]}, index=sample_data.index[:3])

        # Act
        strength = strategy.calculate_signal_strength(sample_data[:3], signals)

        # Assert
        assert all(s == 0.0 for s in strength), "Signal strength should be zero when no position is taken"

    def test_calculate_signal_strength_applies_volatility_adjustment_when_enabled(self, default_config, sample_data):
        """Test volatility adjustment reduces signal strength during high volatility periods"""
        # Arrange
        config_with_vol_adj = RSIStrategyConfig(volatility_adjustment=True)
        config_without_vol_adj = RSIStrategyConfig(volatility_adjustment=False)

        strategy_with_vol = RSIMeanReversionStrategy(config_with_vol_adj)
        strategy_without_vol = RSIMeanReversionStrategy(config_without_vol_adj)

        # Create data with high volatility period
        high_vol_data = sample_data.copy()
        high_vol_data.loc[high_vol_data.index[0], "atr"] = 10.0  # Very high ATR

        signals = pd.DataFrame({"position": [1.0]}, index=[high_vol_data.index[0]])

        # Act
        strength_with_vol = strategy_with_vol.calculate_signal_strength(high_vol_data[:1], signals)
        strength_without_vol = strategy_without_vol.calculate_signal_strength(high_vol_data[:1], signals)

        # Assert
        assert (
            strength_with_vol.iloc[0] <= strength_without_vol.iloc[0]
        ), "Volatility adjustment should reduce or maintain signal strength"

    def test_calculate_signal_strength_handles_missing_atr_gracefully(self, strategy):
        """Test signal strength calculation works without ATR column"""
        # Arrange
        data_without_atr = pd.DataFrame({"close": [100], "rsi_14": [25]}, index=pd.date_range("2023-01-01", periods=1))

        signals = pd.DataFrame({"position": [1.0]}, index=data_without_atr.index)

        # Act + Assert - should not raise error
        strength = strategy.calculate_signal_strength(data_without_atr, signals)
        assert len(strength) == 1
        assert 0 <= strength.iloc[0] <= 1


class TestRSIStrategyEdgeCases:
    """Test edge cases and error handling for RSI Strategy"""

    @pytest.fixture
    def strategy(self):
        return RSIMeanReversionStrategy(RSIStrategyConfig())

    def test_generate_signals_handles_empty_dataframe(self, strategy):
        """Test strategy handles empty DataFrame gracefully"""
        # Arrange
        empty_data = pd.DataFrame()

        # Act + Assert
        with pytest.raises((ValueError, KeyError)):
            strategy.generate_signals( empty_data)

    def test_generate_signals_handles_single_row_dataframe(self, strategy):
        """Test strategy handles single row DataFrame"""
        # Arrange
        single_row_data = pd.DataFrame(
            {"close": [100], "rsi_14": [25], "atr": [1.0]}, index=pd.date_range("2023-01-01", periods=1)
        )

        # Act
        signals = strategy.generate_signals(single_row_data)

        # Assert
        assert len(signals) == 1
        assert "position" in signals.columns

    def test_generate_signals_handles_all_nan_rsi_values(self, strategy):
        """Test strategy handles DataFrame with all NaN RSI values"""
        # Arrange
        nan_rsi_data = pd.DataFrame(
            {"close": [100, 101, 102], "rsi_14": [np.nan, np.nan, np.nan], "atr": [1.0, 1.1, 1.2]},
            index=pd.date_range("2023-01-01", periods=3),
        )

        # Act
        signals = strategy.generate_signals(nan_rsi_data)

        # Assert
        # Should not generate any position signals with NaN RSI
        assert all(signals["position"] == 0), "No positions should be taken with NaN RSI values"

    def test_calculate_signal_strength_handles_extreme_boundary_values(self, strategy):
        """Test signal strength calculation with RSI boundary values (0, 100)"""
        # Arrange
        boundary_data = pd.DataFrame(
            {"close": [100, 100], "rsi_14": [0, 100], "atr": [1.0, 1.0]},  # Extreme boundary values
            index=pd.date_range("2023-01-01", periods=2),
        )

        signals = pd.DataFrame(
            {"position": [1.0, -1.0]}, index=boundary_data.index  # Long for 0 RSI short for 100 RSI
        )

        # Act
        strength = strategy.calculate_signal_strength(boundary_data, signals)

        # Assert
        assert all(0 <= s <= 1 for s in strength), "Signal strengths should be bounded between 0 and 1"
        assert strength.iloc[0] > 0, "RSI of 0 should produce positive signal strength for long position"
        assert strength.iloc[1] > 0, "RSI of 100 should produce positive signal strength for short position"

    def test_strategy_description_contains_key_parameters(self, strategy):
        """Test strategy description includes key configuration parameters"""
        # Act
        description = strategy.get_strategy_description()

        # Assert
        assert "RSI Mean Reversion Strategy" in description
        assert str(strategy.config.oversold_threshold) in description
        assert str(strategy.config.overbought_threshold) in description
        assert str(strategy.config.neutral_zone_lower) in description
        assert str(strategy.config.neutral_zone_upper) in description


