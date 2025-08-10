#!/usr/bin/env python3

"""
Unit tests for MACD Momentum Strategy
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.implementations.macd_momentum_strategy import MACDMomentumStrategy, MACDStrategyConfig


class TestMACDStrategyConfig:
    """Test MACD Strategy Configuration"""

    def test_default_config_values_are_valid(self):
        """Test that default configuration values are properly set"""
        # Arrange + Act
        config = MACDStrategyConfig()

        # Assert
        assert config.name == "macd_momentum_strategy"
        assert config.fast_period == 12
        assert config.slow_period == 26
        assert config.signal_period == 9
        assert config.signal_threshold == 0.0
        assert config.histogram_threshold == 0.0
        assert config.momentum_confirmation is True
        assert config.divergence_detection is False
        assert config.exit_on_opposite_signal is True
        assert config.exit_on_signal_line_cross is True
        assert config.max_holding_period == 20
        assert config.volatility_adjustment is True

    def test_config_validation_with_valid_parameters(self):
        """Test configuration validation passes with valid parameters"""
        # Arrange
        config = MACDStrategyConfig(
            fast_period=8,
            slow_period=21,
            signal_period=6,
            signal_threshold=0.1,
            histogram_threshold=0.05,
        )

        # Act & Assert - Should not raise
        strategy = MACDMomentumStrategy(config)
        assert strategy.validate_parameters() is True

    def test_config_validation_fails_when_fast_period_greater_than_slow_period(self):
        """Test validation fails when fast period >= slow period"""
        # Arrange
        config = MACDStrategyConfig(fast_period=26, slow_period=12)  # Invalid: fast >= slow

        # Act & Assert
        with pytest.raises(ValueError, match="Fast period must be smaller than slow period"):
            MACDMomentumStrategy(config)

    def test_config_validation_fails_when_fast_period_equals_slow_period(self):
        """Test validation fails when fast period equals slow period"""
        # Arrange
        config = MACDStrategyConfig(fast_period=20, slow_period=20)  # Invalid: fast == slow

        # Act & Assert
        with pytest.raises(ValueError, match="Fast period must be smaller than slow period"):
            MACDMomentumStrategy(config)

    def test_config_validation_fails_when_signal_period_invalid(self):
        """Test validation fails when signal period <= 0"""
        # Arrange
        config = MACDStrategyConfig(signal_period=0)  # Invalid: signal period <= 0

        # Act & Assert
        with pytest.raises(ValueError, match="Signal Period must be positive"):
            MACDMomentumStrategy(config)


class TestMACDMomentumStrategy:
    """Test MACD Momentum Strategy Implementation"""

    @pytest.fixture
    def default_config(self):
        """Default MACD strategy configuration for tests"""
        return MACDStrategyConfig()

    @pytest.fixture
    def strategy(self, default_config):
        """MACD strategy instance with default config"""
        return MACDMomentumStrategy(default_config)

    @pytest.fixture
    def sample_data(self):
        """Sample market data with MACD values for testing"""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        # Create realistic price data
        close_prices = 100 + np.cumsum(np.random.randn(50) * 0.02)

        # Create MACD values that trigger various conditions
        macd_values = (
            [-0.5, -0.3, 0.1, 0.3, 0.2]  # Bullish crossover sequence (5)
            + [0.8, 1.0, 0.9, 0.7, 0.3]  # Strong bullish then weakening (5)
            + [0.1, -0.1, -0.4, -0.6, -0.3]  # Bearish crossover sequence (5)
            + list(np.random.uniform(-0.2, 0.2, 20))  # Neutral oscillation (20)
            + [0.2, 0.4, 0.6, 0.4, 0.1]  # Another bullish sequence (5)
            + list(np.random.uniform(-0.1, 0.1, 10))  # More neutral (10)
        )

        macd_signal_values = (
            [-0.4, -0.4, -0.2, 0.0, 0.1]  # Signal line lagging MACD (5)
            + [0.6, 0.8, 1.0, 0.9, 0.6]  # Signal following MACD up (5)
            + [0.3, 0.1, -0.2, -0.4, -0.5]  # Signal following MACD down (5)
            + list(np.random.uniform(-0.1, 0.1, 20))  # Neutral signal (20)
            + [0.0, 0.2, 0.4, 0.5, 0.4]  # Signal following up (5)
            + list(np.random.uniform(-0.05, 0.05, 10))  # More neutral signal (10)
        )

        # MACD Histogram = MACD - Signal
        macd_histogram = np.array(macd_values) - np.array(macd_signal_values)
        atr_values = np.random.uniform(1.0, 3.0, 50)

        return pd.DataFrame(
            {
                "close": close_prices,
                "macd": macd_values,
                "macd_signal": macd_signal_values,
                "macd_histogram": macd_histogram,
                "atr": atr_values,
            },
            index=dates,
        )

    def test_get_required_features_includes_macd_and_base_features(self, strategy):
        """Test that required features include MACD and base OHLCV features"""
        # Act
        required_features = strategy.get_required_features()

        # Assert
        expected_base = ["open", "high", "low", "close", "volume"]
        expected_macd = ["macd", "macd_signal", "macd_histogram", "close", "atr"]

        for feature in expected_base + expected_macd:
            assert feature in required_features

    def test_generate_signals_creates_correct_position_for_bullish_crossover(self, strategy, sample_data):
        """Test that long positions are generated when MACD crosses above signal"""
        # Arrange - Create data with clear bullish crossover
        bullish_data = sample_data.copy()
        # Set up bullish crossover: MACD crosses above signal
        bullish_data.loc[bullish_data.index[5:8], "macd"] = [-0.1, 0.1, 0.3]  # MACD rising
        bullish_data.loc[bullish_data.index[5:8], "macd_signal"] = [0.0, 0.0, 0.1]  # Signal stable/rising slower
        bullish_data.loc[bullish_data.index[5:8], "macd_histogram"] = [-0.1, 0.1, 0.2]  # Histogram turning positive

        # Act
        signals = strategy.generate_signals(bullish_data)

        # Assert
        bullish_entries = signals[signals["position"] > 0]
        assert len(bullish_entries) > 0, "Should generate long positions for bullish crossover"

        # Check that positions are only 1.0 or 0.0
        assert all(pos in [0.0, 1.0] for pos in signals["position"].unique())

    def test_generate_signals_creates_correct_position_for_bearish_crossover(self, strategy, sample_data):
        """Test that short positions are generated when MACD crosses below signal"""
        # Arrange - Create data with clear bearish crossover, no bullish conditions
        bearish_data = sample_data.copy()
        # Set up bearish crossover: MACD crosses below signal
        bearish_data.loc[bearish_data.index[10:13], "macd"] = [0.2, 0.0, -0.2]  # MACD falling
        bearish_data.loc[bearish_data.index[10:13], "macd_signal"] = [0.0, 0.0, -0.1]  # Signal stable/falling slower
        bearish_data.loc[bearish_data.index[10:13], "macd_histogram"] = [0.2, 0.0, -0.1]  # Histogram turning negative

        # Remove any potential bullish conditions by setting other values to neutral
        mask = ~bearish_data.index.isin(bearish_data.index[10:13])
        bearish_data.loc[mask, "macd"] = 0.0
        bearish_data.loc[mask, "macd_signal"] = 0.0
        bearish_data.loc[mask, "macd_histogram"] = 0.0

        # Act
        signals = strategy.generate_signals(bearish_data)

        # Assert
        bearish_entries = signals[signals["position"] < 0]
        assert len(bearish_entries) > 0, "Should generate short positions for bearish crossover"

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

            if position > 0:  # Long position
                expected_stop_loss = close_price - 2 * atr
                expected_take_profit = close_price + 3 * atr
            else:  # Short position
                expected_stop_loss = close_price + 2 * atr
                expected_take_profit = close_price - 3 * atr

            if not pd.isna(signals.loc[idx, "stop_loss"]):
                assert abs(signals.loc[idx, "stop_loss"] - expected_stop_loss) < 1e-6
            if not pd.isna(signals.loc[idx, "take_profit"]):
                assert abs(signals.loc[idx, "take_profit"] - expected_take_profit) < 1e-6

    def test_generate_signals_handles_missing_macd_columns_gracefully(self, strategy):
        """Test strategy raises error when MACD columns are missing"""
        # Arrange - Data without required MACD columns
        data_without_macd = pd.DataFrame(
            {"close": [100, 101, 102], "atr": [1.5, 1.6, 1.4]}, 
            index=pd.date_range("2023-01-01", periods=3)
        )

        # Act + Assert
        with pytest.raises(ValueError, match="Required MACD columns not found in data"):
            strategy.generate_signals(data_without_macd)

    def test_generate_signals_handles_partial_missing_macd_columns(self, strategy):
        """Test strategy raises error when some MACD columns are missing"""
        # Arrange - Data with only some MACD columns
        data_partial_macd = pd.DataFrame(
            {"close": [100, 101, 102], "macd": [0.1, 0.2, 0.1], "atr": [1.5, 1.6, 1.4]},
            index=pd.date_range("2023-01-01", periods=3)
        )

        # Act + Assert
        with pytest.raises(ValueError, match="Required MACD columns not found in data"):
            strategy.generate_signals(data_partial_macd)

    def test_calculate_signal_strength_returns_higher_strength_for_larger_histogram(self, strategy, sample_data):
        """Test signal strength increases as MACD histogram magnitude increases"""
        # Create signals with different histogram magnitudes
        test_data = pd.DataFrame(
            {
                "close": [100, 100, 100, 100],
                "macd": [0.2, 0.5, -0.2, -0.8],
                "macd_signal": [0.0, 0.0, 0.0, 0.0],
                "macd_histogram": [0.2, 0.5, -0.2, -0.8],  # Increasing magnitudes
                "atr": [1.0, 1.0, 1.0, 1.0],
            },
            index=pd.date_range("2023-01-01", periods=4),
        )

        signals = pd.DataFrame(
            {"position": [1.0, 1.0, -1.0, -1.0]}, index=test_data.index  # Long for positive, short for negative
        )

        # Act
        strength = strategy.calculate_signal_strength(test_data, signals)

        # Assert
        assert (
            strength.iloc[1] > strength.iloc[0]
        ), "Larger positive histogram (0.5) should have higher strength than smaller (0.2)"
        assert (
            strength.iloc[3] > strength.iloc[2]
        ), "Larger negative histogram (-0.8) should have higher strength than smaller (-0.2)"
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
        config_with_vol_adj = MACDStrategyConfig(volatility_adjustment=True)
        config_without_vol_adj = MACDStrategyConfig(volatility_adjustment=False)

        strategy_with_vol = MACDMomentumStrategy(config_with_vol_adj)
        strategy_without_vol = MACDMomentumStrategy(config_without_vol_adj)

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

    def test_calculate_signal_strength_handles_missing_macd_histogram_gracefully(self, strategy):
        """Test signal strength calculation works without MACD histogram column"""
        # Arrange
        data_without_histogram = pd.DataFrame(
            {"close": [100], "macd": [0.2], "macd_signal": [0.1]}, 
            index=pd.date_range("2023-01-01", periods=1)
        )

        signals = pd.DataFrame({"position": [1.0]}, index=data_without_histogram.index)

        # Act + Assert - should not raise error and return zero strength
        strength = strategy.calculate_signal_strength(data_without_histogram, signals)
        assert len(strength) == 1
        assert strength.iloc[0] == 0.0

    def test_detect_bullish_crossover_identifies_crossovers_correctly(self, strategy):
        """Test bullish crossover detection logic"""
        # Arrange
        macd = pd.Series([-0.2, -0.1, 0.1, 0.2])  # MACD crossing above signal
        macd_signal = pd.Series([0.0, 0.0, 0.0, 0.1])  # Signal line stable then rising

        # Act
        bullish_crosses = strategy._detect_bullish_crossover(macd, macd_signal)

        # Assert
        # Should detect crossover at index 2 (MACD goes from below to above signal)
        assert bullish_crosses.iloc[2] is True, "Should detect bullish crossover"
        assert bullish_crosses.iloc[0] is False, "Should not detect crossover at start"
        assert bullish_crosses.iloc[1] is False, "Should not detect crossover before actual cross"

    def test_detect_bearish_crossover_identifies_crossovers_correctly(self, strategy):
        """Test bearish crossover detection logic"""
        # Arrange
        macd = pd.Series([0.2, 0.1, -0.1, -0.2])  # MACD crossing below signal
        macd_signal = pd.Series([0.0, 0.0, 0.0, -0.1])  # Signal line stable then falling

        # Act
        bearish_crosses = strategy._detect_bearish_crossover(macd, macd_signal)

        # Assert
        # Should detect crossover at index 2 (MACD goes from above to below signal)
        assert bearish_crosses.iloc[2] is True, "Should detect bearish crossover"
        assert bearish_crosses.iloc[0] is False, "Should not detect crossover at start"
        assert bearish_crosses.iloc[1] is False, "Should not detect crossover before actual cross"

    def test_momentum_confirmation_filters_signals_when_enabled(self, sample_data):
        """Test momentum confirmation filters signals based on histogram"""
        # Arrange
        config_with_confirmation = MACDStrategyConfig(
            momentum_confirmation=True, 
            histogram_threshold=0.1
        )
        config_without_confirmation = MACDStrategyConfig(momentum_confirmation=False)

        strategy_with_confirmation = MACDMomentumStrategy(config_with_confirmation)
        strategy_without_confirmation = MACDMomentumStrategy(config_without_confirmation)

        # Create data with crossover but weak histogram
        weak_histogram_data = sample_data.copy()
        weak_histogram_data.loc[weak_histogram_data.index[5:8], "macd"] = [0.0, 0.2, 0.3]
        weak_histogram_data.loc[weak_histogram_data.index[5:8], "macd_signal"] = [0.1, 0.1, 0.2]
        weak_histogram_data.loc[weak_histogram_data.index[5:8], "macd_histogram"] = [0.05, 0.05, 0.05]  # Below threshold

        # Act
        signals_with_confirmation = strategy_with_confirmation.generate_signals(weak_histogram_data)
        signals_without_confirmation = strategy_without_confirmation.generate_signals(weak_histogram_data)

        # Assert
        entries_with_confirmation = len(signals_with_confirmation[signals_with_confirmation["position"] != 0])
        entries_without_confirmation = len(signals_without_confirmation[signals_without_confirmation["position"] != 0])

        assert entries_with_confirmation <= entries_without_confirmation, \
            "Momentum confirmation should reduce or maintain signal count"


class TestMACDStrategyEdgeCases:
    """Test edge cases and error handling for MACD Strategy"""

    @pytest.fixture
    def strategy(self):
        return MACDMomentumStrategy(MACDStrategyConfig())

    def test_generate_signals_handles_empty_dataframe(self, strategy):
        """Test strategy handles empty DataFrame gracefully"""
        # Arrange
        empty_data = pd.DataFrame()

        # Act + Assert
        with pytest.raises((ValueError, KeyError)):
            strategy.generate_signals(empty_data)

    def test_generate_signals_handles_single_row_dataframe(self, strategy):
        """Test strategy handles single row DataFrame"""
        # Arrange
        single_row_data = pd.DataFrame(
            {
                "close": [100],
                "macd": [0.1],
                "macd_signal": [0.0],
                "macd_histogram": [0.1],
                "atr": [1.0]
            },
            index=pd.date_range("2023-01-01", periods=1)
        )

        # Act
        signals = strategy.generate_signals(single_row_data)

        # Assert
        assert len(signals) == 1
        assert "position" in signals.columns

    def test_generate_signals_handles_all_nan_macd_values(self, strategy):
        """Test strategy handles DataFrame with all NaN MACD values"""
        # Arrange
        nan_macd_data = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "macd": [np.nan, np.nan, np.nan],
                "macd_signal": [np.nan, np.nan, np.nan],
                "macd_histogram": [np.nan, np.nan, np.nan],
                "atr": [1.0, 1.1, 1.2],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        # Act
        signals = strategy.generate_signals(nan_macd_data)

        # Assert
        # Should not generate any position signals with NaN MACD
        assert all(signals["position"] == 0), "No positions should be taken with NaN MACD values"

    def test_calculate_signal_strength_handles_extreme_histogram_values(self, strategy):
        """Test signal strength calculation with extreme MACD histogram values"""
        # Arrange
        extreme_data = pd.DataFrame(
            {
                "close": [100, 100],
                "macd": [10.0, -10.0],
                "macd_signal": [0.0, 0.0],
                "macd_histogram": [10.0, -10.0],  # Extreme histogram values
                "atr": [1.0, 1.0],
            },
            index=pd.date_range("2023-01-01", periods=2),
        )

        signals = pd.DataFrame(
            {"position": [1.0, -1.0]}, index=extreme_data.index  # Long for positive, short for negative
        )

        # Act
        strength = strategy.calculate_signal_strength(extreme_data, signals)

        # Assert
        assert all(0 <= s <= 1 for s in strength), "Signal strengths should be bounded between 0 and 1"
        assert strength.iloc[0] > 0, "Extreme positive histogram should produce positive signal strength"
        assert strength.iloc[1] > 0, "Extreme negative histogram should produce positive signal strength for short"

    def test_strategy_description_contains_key_parameters(self, strategy):
        """Test strategy description includes key configuration parameters"""
        # Act
        description = strategy.get_strategy_description()

        # Assert
        assert "MACD Momentum Strategy" in description
        assert str(strategy.config.fast_period) in description
        assert str(strategy.config.slow_period) in description
        assert str(strategy.config.signal_period) in description
        assert "bullish crossover" in description
        assert "bearish crossover" in description

    def test_exit_on_opposite_signal_behavior(self, strategy):
        """Test exit on opposite signal configuration"""
        # Arrange - Create alternating crossover signals
        alternating_data = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "macd": [0.1, 0.2, -0.1, -0.2, 0.1],  # Alternating crosses
                "macd_signal": [0.0, 0.1, 0.0, -0.1, 0.0],
                "macd_histogram": [0.1, 0.1, -0.1, -0.1, 0.1],
                "atr": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=pd.date_range("2023-01-01", periods=5),
        )

        # Act
        signals = strategy.generate_signals(alternating_data)

        # Assert
        # With exit_on_opposite_signal=True, positions should change with crossovers
        position_changes = (signals["position"].diff() != 0).sum()
        assert position_changes > 0, "Should have position changes with alternating crossovers"