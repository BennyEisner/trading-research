#!/usr/bin/env python3

"""
Pattern-Focused Feature Engineering for LSTM
Implements 12 key features optimized for temporal pattern detection
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import talib


class PatternFeatureEngine:
    """
    Generates 12 pattern-focused features for LSTM input
    Optimized for temporal pattern recognition and ensemble integration
    """

    def __init__(self):
        self.feature_names = [
            # Non-linear price pattenrs
            "price acceleration",
            "volume_price_divergence",
            "volatility_regime_change",
            "return_skewness_7d",
            # Temporal dependencies for LSTM optimization
            "momentum_persistence_7d",
            "volatility_clustering",
            "trend_exhaustion",
            "garch_volatility_forecast",
            # Market microstructure
            "intraday_range_expansion",  # Range breakouts
            "overnight_gap_behavior",
            "end_of_day_momentum",
            # Cross asset relationships
            "sector_relative_strength",
            "market_beta_instability",
            "vix_term_structure",
            # Core features for basic context
            "returns_1d",
            "returns_3d",
            "returns_7d",
            "volume_normalized",
            "close",
        ]

    def generate_pattern_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate 10-20 pattern-focused features from OHLCV data that complement signal strategies

        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            numpy array with shape (n_samples, 19) of pattern features
        """

        features = {}

        # Non-linear price patterns 
        features.update(self._generate_non_linear_price_patterns(data))

        # Trend Exhaustion and Health patterns
        features.update(self._generate_trend_exhaustion_patterns(data))

        # Volume and Market Microstructure Patterns 
        features.update(self._generate_volume_and_market_microstructure_patterns(data))

        # Technical patterns
        features.update(self._generate_volatility_regime_patterns(data))

        features.update(self._generate_cross_asset_relations(data))

        # Combine into array
        feature_matrix = np.column_stack([features[name] for name in self.feature_names])

        # Handle NaN values
        feature_matrix = self._handle_nan_values(feature_matrix)

        return feature_matrix

    def _generate_non_linear_price_patterns(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate non-linear price pattern features"""

        close = data["close"].values
        volume = data["volume"].values
        
        returns = data["close"].pct_change()
        momentum_direction = np.sign(returns.rolling(signal_window).mean())
        
        # Momentum and Trend Dynamics
        price_acceleration = returns.diff()
        momentum_persistence =  


        # Price momentum features
        price_momentum_5d = self._calculate_momentum(close, 5)
        price_momentum_20d = self._calculate_momentum(close, 20)

        # Volume momentum
        volume_sma_20 = talib.SMA(volume, timeperiod=20)
        volume_momentum = (volume / volume_sma_20 - 1.0) * 100

        return {
            "price_momentum_5d": price_momentum_5d,
            "price_momentum_20d": price_momentum_20d,
            "volume_momentum": volume_momentum,
        }

    def _generate_volatility_patterns(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility-based pattern features"""

        close = data["close"].values
        high = data["high"].values
        low = data["low"].values

        # Returns for volatility calculations
        returns = np.diff(np.log(close))
        returns = np.concatenate([[0], returns])  # Pad first value

        # Volatility regime (rolling volatility z-score)
        volatility = pd.Series(returns).rolling(window=20).std()
        volatility_long_term = volatility.rolling(window=60).mean()
        volatility_regime = (volatility - volatility_long_term) / volatility.rolling(window=60).std()

        # Volatility clustering (GARCH-like effect)
        abs_returns = np.abs(returns)
        volatility_clustering = (
            pd.Series(abs_returns).rolling(window=5).mean() / pd.Series(abs_returns).rolling(window=20).mean()
        )

        # GARCH signal (persistence in volatility)
        garch_signal = self._calculate_garch_signal(returns)

        return {
            "volatility_regime": volatility_regime.values,
            "volatility_clustering": volatility_clustering.values,
            "garch_signal": garch_signal,
        }

    def _generate_price_patterns(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate price action pattern features"""

        open_prices = data["open"].values
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        # Gap analysis
        gap_analysis = self._calculate_gap_analysis(open_prices, close)

        # Support/resistance levels
        support_resistance = self._calculate_support_resistance(high, low, close)

        # Trend strength
        trend_strength = self._calculate_trend_strength(close)

        return {
            "gap_analysis": gap_analysis,
            "support_resistance": support_resistance,
            "trend_strength": trend_strength,
        }

    def _generate_technical_patterns(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate technical indicator pattern features"""

        close = data["close"].values
        high = data["high"].values
        low = data["low"].values

        # RSI divergence
        rsi_divergence = self._calculate_rsi_divergence(close, high, low)

        # MACD momentum
        macd_momentum = self._calculate_macd_momentum(close)

        # Bollinger position
        bollinger_position = self._calculate_bollinger_position(close)

        return {
            "rsi_divergence": rsi_divergence,
            "macd_momentum": macd_momentum,
            "bollinger_position": bollinger_position,
        }

    def _calculate_momentum(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate price momentum over specified period"""
        momentum = np.zeros_like(prices)
        for i in range(period, len(prices)):
            momentum[i] = (prices[i] / prices[i - period] - 1.0) * 100
        return momentum

    def _calculate_garch_signal(self, returns: np.ndarray, alpha: float = 0.1, beta: float = 0.85) -> np.ndarray:
        """Calculate GARCH-like volatility persistence signal"""

        n = len(returns)
        garch_signal = np.zeros(n)
        variance = np.var(returns[:20])  # Initial variance estimate

        for i in range(20, n):
            # GARCH(1,1) update
            variance = alpha * returns[i - 1] ** 2 + beta * variance + (1 - alpha - beta) * np.var(returns[:i])
            garch_signal[i] = variance

        # Normalize
        garch_signal = (garch_signal - np.mean(garch_signal)) / (np.std(garch_signal) + 1e-8)

        return garch_signal

    def _calculate_gap_analysis(self, open_prices: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Calculate gap analysis (overnight gaps)"""

        gaps = np.zeros_like(open_prices)
        for i in range(1, len(open_prices)):
            gaps[i] = (open_prices[i] / close_prices[i - 1] - 1.0) * 100

        return gaps

    def _calculate_support_resistance(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate support/resistance level proximity"""

        support_resistance = np.zeros_like(close)
        window = 20

        for i in range(window, len(close)):
            recent_high = np.max(high[i - window : i])
            recent_low = np.min(low[i - window : i])

            # Position within recent range
            if recent_high != recent_low:
                support_resistance[i] = (close[i] - recent_low) / (recent_high - recent_low) * 100
            else:
                support_resistance[i] = 50.0  # Neutral position

        return support_resistance

    def _calculate_trend_strength(self, close: np.ndarray) -> np.ndarray:
        """Calculate trend strength using linear regression slope"""

        trend_strength = np.zeros_like(close)
        window = 20

        for i in range(window, len(close)):
            x = np.arange(window)
            y = close[i - window : i]

            # Linear regression slope
            slope = np.polyfit(x, y, 1)[0]
            trend_strength[i] = slope / close[i] * 100  # Normalize by current price

        return trend_strength

    def _calculate_rsi_divergence(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Calculate RSI divergence signal"""

        rsi = talib.RSI(close, timeperiod=14)
        rsi_divergence = np.zeros_like(close)
        window = 10

        for i in range(window, len(close)):
            # Price trend
            price_trend = (close[i] - close[i - window]) / close[i - window]

            # RSI trend
            rsi_trend = (rsi[i] - rsi[i - window]) / 100

            # Divergence occurs when trends oppose
            rsi_divergence[i] = price_trend * -rsi_trend * 100  # Multiply by -1 for divergence

        return rsi_divergence

    def _calculate_macd_momentum(self, close: np.ndarray) -> np.ndarray:
        """Calculate MACD momentum signal"""

        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        # MACD momentum is the rate of change of MACD histogram
        macd_momentum = np.zeros_like(close)
        for i in range(5, len(macd_hist)):
            if not (np.isnan(macd_hist[i]) or np.isnan(macd_hist[i - 5])):
                macd_momentum[i] = macd_hist[i] - macd_hist[i - 5]

        return macd_momentum

    def _calculate_bollinger_position(self, close: np.ndarray) -> np.ndarray:
        """Calculate position within Bollinger Bands"""

        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

        bollinger_position = np.zeros_like(close)
        for i in range(len(close)):
            if not (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
                if bb_upper[i] != bb_lower[i]:
                    bollinger_position[i] = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) * 100
                else:
                    bollinger_position[i] = 50.0

        return bollinger_position

    def _handle_nan_values(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Handle NaN values in feature matrix"""

        # Forward fill then backward fill
        df = pd.DataFrame(feature_matrix, columns=self.feature_names)
        df = df.fillna(method="ffill").fillna(method="bfill")

        # If still NaN, fill with median
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        return df.values

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of each feature"""

        descriptions = {
            "price_momentum_5d": "Short-term price momentum (5-day)",
            "price_momentum_20d": "Medium-term price momentum (20-day)",
            "volume_momentum": "Volume relative to 20-day average",
            "volatility_regime": "Current volatility vs long-term regime",
            "volatility_clustering": "Short vs long-term volatility clustering",
            "garch_signal": "GARCH-like volatility persistence",
            "gap_analysis": "Overnight price gaps",
            "support_resistance": "Position within recent trading range",
            "trend_strength": "Linear regression slope strength",
            "rsi_divergence": "RSI vs price divergence signal",
            "macd_momentum": "MACD histogram momentum",
            "bollinger_position": "Position within Bollinger Bands",
        }

        return descriptions


def create_sequence_features(features: np.ndarray, sequence_length: int = 30) -> np.ndarray:
    """
    Convert features to sequences for LSTM input

    Args:
        features: Feature matrix (n_samples, n_features)
        sequence_length: Length of sequences to create

    Returns:
        Sequence features (n_sequences, sequence_length, n_features)
    """

    n_samples, n_features = features.shape
    n_sequences = n_samples - sequence_length + 1

    sequences = np.zeros((n_sequences, sequence_length, n_features))

    for i in range(n_sequences):
        sequences[i] = features[i : i + sequence_length]

    return sequences


if __name__ == "__main__":
    # Test with synthetic data
    n_samples = 1000

    # Create synthetic OHLCV data
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(n_samples) * 0.02)

    synthetic_data = pd.DataFrame(
        {
            "open": prices + np.random.randn(n_samples) * 0.01,
            "high": prices + np.abs(np.random.randn(n_samples) * 0.015),
            "low": prices - np.abs(np.random.randn(n_samples) * 0.015),
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n_samples),
        }
    )

    # Test feature generation
    engine = PatternFeatureEngine()
    features = engine.generate_pattern_features(synthetic_data)

    print(f"Generated features shape: {features.shape}")
    print(f"Feature names: {engine.get_feature_names()}")

    # Test sequence creation
    sequences = create_sequence_features(features, sequence_length=30)
    print(f"Sequence features shape: {sequences.shape}")

    # Check for NaN values
    print(f"NaN values in features: {np.isnan(features).sum()}")
    print(f"NaN values in sequences: {np.isnan(sequences).sum()}")

    print("Pattern feature generation test completed successfully!")

