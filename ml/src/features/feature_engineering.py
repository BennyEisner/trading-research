#!/usr/bin/env python3

"""
Feature engineering for financial time series data
"""


import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Handles all feature engineering for financial data"""

    def __init__(self, symbol="APPLE", market_data=None, mag7_data=None):
        self.symbol = symbol
        self.market_data = market_data
        self.mag7_data = mag7_data
        self.scalars = {}
        self.feature_importance = {}
        self.selected_features = {
            "lagged_returns": ["daily_return_lag1", "daily_return_lag2", "daily_return_lag3", "daily_return_lag5", "daily_return_lag10"],
            "moving_averages": ["price_to_sma5", "price_to_sma20", "price_to_sma50", "sma5_to_sma20", "sma20_to_sma50", "ema12_to_ema26"],
            "technical_indicators": ["rsi_lag1", "macd_histogram_lag1", "bb_position", "stoch_k", "williams_r", "cci"],
            "volume_features": [
                "volume_ratio_lag1",
                "relative_volume",
                "vwap_deviation",
                "volume_price_trend",
                "money_flow_index",
                "obv_normalized",
            ],
            "volatility_features": [
                "volatility_ratio",
                "return_volatility_5d",
                "garch_volatility",
                "volatility_clustering",
                "volatility_skew",
            ],
            "momentum_features": ["price_momentum_7d", "momentum_2d", "momentum_5d", "momentum_strength", "momentum_persistence"],
            "market_structure": ["price_position", "buying_pressure", "market_impact", "liquidity_ratio", "spread_normalized"],
            "market_context": ["spy_correlation", "vix_level", "sector_performance", "market_regime", "yield_curve_slope"],
            "cross_asset": ["mag7_correlation", "relative_strength", "sector_rotation", "beta_stability", "idiosyncratic_vol"],
            "risk_factors": ["momentum_factor", "value_factor", "quality_factor", "size_factor", "profitability_factor"],
            "regime_detection": ["volatility_regime", "trend_regime", "volume_regime", "correlation_regime", "market_stress"],
            "seasonality": ["day_of_week", "month_of_year", "earnings_proximity", "options_expiry", "rebalancing_effect"],
            "alternative_data": ["sentiment_score", "news_flow", "social_sentiment", "search_volume", "analyst_revisions"],
        }
        # Selected features for LSTM
        self.selected_features = [
            # Core lagged returns
            "daily_return_lag1",
            "daily_return_lag2",
            "daily_return_lag5",
            # Moving averages
            "price_to_sma5",
            "price_to_sma20",
            "price_to_sma50",
            "sma5_to_sma20",
            "ema12_to_ema26",
            # Technical indicators
            "rsi_lag1",
            "macd_histogram_lag1",
            "bb_position",
            "stoch_k",
            "williams_r",
            # Volume and liquidity
            "volume_ratio_lag1",
            "relative_volume",
            "vwap_deviation",
            "money_flow_index",
            "obv_normalized",
            # Volatility measures
            "volatility_ratio",
            "return_volatility_5d",
            "garch_volatility",
            "volatility_clustering",
            # Momentum indicators
            "price_momentum_7d",
            "momentum_2d",
            "momentum_5d",
            "momentum_strength",
            # Market structure
            "price_position",
            "buying_pressure",
            "liquidity_ratio",
            # Market context for mag7
            "spy_correlation",
            "vix_level",
            "market_regime",
            # Cross-asset features
            "mag7_correlation",
            "relative_strength",
            "beta_stability",
            # Risk factors
            "momentum_factor",
            "quality_factor",
            # Regime detection
            "volatility_regime",
            "trend_regime",
            # Seasonality
            "day_of_week",
            "earnings_proximity",
            # Base features
            "volume",
            "close",
        ]

    def calculate_all_features(self, data):
        """
        Calculate feature set for LSTM model
        """
        print(f"Calculating features for {self.symbol}")
        data = data.copy()

        # Data quality check
        data = self._validate_and_clean_data(data)

        # Core feature selection
        data = self._calculate_price_features(data)
        data = self._calculate_moving_averages(data)
        data = self._calculate_technical_indicators(data)
        data = self._calculate_volume_features(data)
        data = self._calculate_momentum_features(data)
        data = self._calculate_volatility_features(data)
        data = self._calculate_market_structure_features(data)

        # Advanced feature calculations
        data = self._calculate_market_context_features(data)
        data = self._calculate_cross_asset_features(data)
        data = self._calculate_risk_factors(data)
        data = self._calculate_regime_features(data)
        data = self._calculate_seasonality_features(data)
        data = self._calculate_alternative_data_features(data)

        # LSTM-specific optimizations
        data = self._create_lstm_optimized_features(data)

        # Feature scaling and selection
        data = self._apply_feature_scaling(data)

        print(f"Feature calculation complete. Total features: {len(data.columns)}")

        return data

    def _validate_and_clean_data(self, data):
        # Check for required colums
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError("Missing required columns: {missing_cols}")

        # Handle missing values
        data = data.fillna(method="ffil").fillna(method="bfill")

        # Remove extreme outliers (i.e. data errors or irrelevant information)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ["open", "high", "low", "close", "volume"]:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                data.loc[z_scores > 5, col] = np.nan

        data = data.fillna(method="ffill")

        # Check for chronological order
        if "data" in data.columns:
            data = data.sort_values("date")

        return data

    def _calculate_price_features(self, data):
        # Basic relationships
        data["high_low_pct"] = (data["high"] - data["low"]) / data["close"] * 100
        data["open_close_pct"] = (data["close"] - data["open"]) / data["open"] * 100
        data["daily_return"] = data["close"].pct_change()

        # Volatility TR
        data["true_range"] = np.maximum(
            data["high"] - data["close"], np.maximum(abs(data["high"] - data["low"].shift(1)), abs(data["low"] - data["high"].shift(1)))
        )

        # Price position
        range_diff = data["high"] - data["low"]
        data["price_position"] = np.where(range_diff != 0, (data["close"] - data["low"]) / range_diff, 0.5)

        # Intraday features
        data["h1_range_normalized"] = (data["high"] - data["low"]) / data["close"]
        data["gap_normalized"] = (data["open"] - data["close"].shift(1)) / data["close"].shift(1)
        data["body_ratio"] = abs(data["close"] - data["open"]) / (data["high"] - data["low"] + 1e-10)
        data["upper_shadow"] = (data["high"] - np.maximum(data["close"], data["open"])) / data["close"]
        data["lower_shadow"] = (np.minimum(data["close"], data["open"]) - data["low"]) / data["close"]

        return data

    def _calculate_moving_averages(self, data):
        """Moving averages with multiple timeframes"""

        # SMAs
        for period in [5, 10, 20, 50, 100]:
            data[f"sma_{period}"] = data["close"].rolling(window=period).mean()

        # EMAs
        for span in [12, 26, 50, 100]:
            data[f"ema_{span}"] = data["close"].ewm(span=span).mean()

        # Hull Moving Average
        data["hma_20"] = self._calculate_hull_ma(data["close"], 20)

        # Normalized Price Features
        data["price_to_sma5"] = data["close"] / data["sma_5"] - 1
        data["price_to_sma20"] = data["close"] / data["sma_20"] - 1
        data["price_to_sma50"] = data["close"] / data["sma_50"] - 1
        data["sma5_to_sma20"] = data["sma_5"] / data["sma_20"] - 1
        data["sma20_to_sma50"] = data["sma_20"] / data["sma_50"] - 1
        data["ema12_to_ema26"] = data["ema_12"] / data["ema_26"] - 1

        # Moving average convergence/divergence
        data["ma_convergence"] = abs(data["sma_5"] - data["sma_20"]) / data["close"]
        data["ma_slope_5"] = data["sma_5"].diff(5) / data["sma_5"].shift(5)
        data["ma_slope_20"] = data["sma_20"].diff(5) / data["sma_20"].shift(5)

        return data

    def _calculate_hull_ma(self, series, period):
        """Calculate Hull Moving Average"""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))

        wma_half = series.rolling(window=half_period).apply(lambda x: np.average(x, weights=np.arange(1, half_period + 1)))
        wma_full = series.rolling(window=period).apply(lambda x: np.average(x, weights=np.arange(1, half_period + 1)))

        raw_hma = 2 * wma_half - wma_full

        hull_ma = raw_hma.rolling(window=sqrt_period).apply(lambda x: np.average(x, weights=np.arange(1, half_period + 1)))

        return hull_ma

    def _calculate_technical_indicators(self, data):
        # Multi-period RSI
        for period in [14, 21]:
            delta = data["close"].diff()
            gain = (delta.where(delta > 0.0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0.0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            data[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        data["rsi"] = data["rsi_14"]

        # MACD
        macd_line = data["ema_12"] - data["ema_26"]
        signal_line = macd_line.ewm(span=9).mean()
        data["macd_line"] = macd_line
        data["macd_signal"] = signal_line
        data["macd_histogram"] = macd_line - signal_line

        # Stochastic Oscillator
        lowest_low = data["low"].rolling(window=14).min()
        highest_high = data["high"].rolling(window=14).max()
        data["stoch_k"] = 100 * (data["close"] - lowest_low) / (highest_high - lowest_low + 1e-10)
        data["stoch_k"] = data["stoch_k"].rolling(window=3).mean()

        # Commodity Channel Index
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        data["cci"] = (typical_price - sma_tp) / (0.015 * mad)

        # Bollinger Bands
        bb_middle = data["close"].rolling(window=20).mean()
        bb_std = data["close"].rolling(window=20).std()
        data["bb_upper"] = bb_middle + (bb_std * 2)
        data["bb_lower"] = bb_middle - (bb_std * 2)

        bb_range = data["bb_upper"] - data["bb_lower"]
        data["bb_position"] = np.where(bb_range != 0, (data["close"] - data["bb_lower"]) / bb_range, 0.5)
        data["bb_width"] = bb_range / data["close"]
        data["bb_squeeze"] = (data["bb_width"] < data["bb_width"].rolling(20).quantile(0.1)).astype(int)

        # Average TR
        data["atr"] = data["true_range"].rolling(window=14).mean()
        data["atr_ratio"] = data["atr"] / data["close"]

        return data

    def _calculate_volume_features(self, data):
        data["volume_sma_20"] = data["volume"].rolling(window=20).mean()
        data["volume_ratio"] = data["volume"] / (data["volume_sma_20"] + 1e-10)
        data["relative_volume"] = data["volume"] / data["volume"].rolling(20).median()

        # VWAP and VWAP variations
        data["vwap_5"] = (data["close"] * data["volume"]).rolling(5).sum() / data["volume"].rolling(5).sum()
        data["vwap_20"] = (data["close"] * data["volume"]).rolling(20).sum() / data["volume"].rolling(20).sum()
        data["vwap_deviation"] = (data["close"] - data["vamp_20"]) / data["vwap_20"]

        # Volume weighted indicators
        data["volume_price_trend"] = ((data["close"] - data["close"].shift(1)) * data["volume"]).rolling(5).sum()

        # On-balance Volume
        obv = []
        obv_val = 0
        for i in range(len(data)):
            if i == 0:
                obv_val = data["volume"].iloc[i]
            else:
                if data["close"].iloc[i] > data["close"].iloc[i - 1]:
                    obv_val += data["volume"].iloc[i]
                elif data["close"].iloc[i] < data["close"].iloc[i - 1]:
                    obv_val -= data["volume"].iloc[i]
            obv.append(obv_val)

            data["obv"] = obv
            data["obv_normalized"] = data["obv"] / data["obv"].rolling(20).mean()

        # Money Flow Index
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        money_flow = typical_price * data["volume"]

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()

        money_flow_ratio = positive_flow / (negative_flow + 1e-10)
        data["money_flow_index"] = 100 - (100 / (1 + money_flow_ratio))
        
        return data

    def _calculate_volatility_features(self, data):
        """Calculate comprehensive volatility features including GARCH"""
        # Calculate log returns for volatility analysis
        returns = np.log(data["close"] / data["close"].shift(1)).dropna()

        # GARCH conditional volatility
        data["garch_volatility"] = self._calculate_GARCH_volatility(returns, data.index)

        # Additional volatility metrics
        data = self._calculate_supporting_volatility_features(data, returns)

        # Williams %R momentum oscillator
        data["williams_r"] = self._calculate_williams_r(data)

        return data

    def _calculate_supporting_volatility_features(self, data, returns):
        """Calculate additional volatility metrics complementing GARCH"""
        
        # Volatility ratio (short-term vs long-term)
        vol_5d = returns.rolling(5).std() * np.sqrt(252)  # Annualized
        vol_20d = returns.rolling(20).std() * np.sqrt(252)
        volatility_ratio = vol_5d / (vol_20d + 1e-8)
        data["volatility_ratio"] = volatility_ratio.reindex(data.index)

        # Simple 5-day rolling volatility
        data["return_volatility_5d"] = (returns.rolling(5).std() * np.sqrt(252)).reindex(data.index)

        # Volatility clustering indicator
        squared_returns = returns**2
        clustering_measure = squared_returns.rolling(10).corr(squared_returns.shift(1))
        data["volatility_clustering"] = clustering_measure.reindex(data.index)

        # Volatility distribution skewness
        vol_rolling = returns.rolling(20).std()
        vol_skew = vol_rolling.rolling(60).skew()
        data["volatility_skew"] = vol_skew.reindex(data.index)

        return data

    def _calculate_williams_r(self, data, period=14):
        """Calculate Williams %R momentum oscillator"""
        highest_high = data["high"].rolling(period).max()
        lowest_low = data["low"].rolling(period).min()
        
        # Williams %R formula: -100 * (HH - Close) / (HH - LL)
        williams_r = -100 * (highest_high - data["close"]) / (highest_high - lowest_low + 1e-10)
        
        return williams_r

    def _calculate_GARCH_volatility(self, returns, full_index, p=1, q=1, window=100, min_periods=30):
        """
        Calculate GARCH(p,q) conditional volatility using rolling window estimation
        
        GARCH(1,1) Model: σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
        
        Args:
            returns: pandas Series of log returns
            full_index: full DataFrame index for alignment
            p: autoregressive lags for variance (default 1)
            q: moving average lags for variance (default 1) 
            window: rolling window for parameter estimation (default 100)
            min_periods: minimum required observations (default 30)
            
        Returns:
            pandas Series of conditional volatilities aligned with full_index
        """

        volatility = np.full(len(full_index), np.nan)

        returns_array = returns.values
        n_obs = len(returns_array)

        if n_obs < min_periods:
            return pd.Series(volatility, index=full_index)

        def estimate_garch_params(ret_window):
            """Estimate GARCH parameters using Maximum Likelihood Estimation"""

            def garch_likelihood(params):
                omega, alpha, beta = params

                # Parameter constraints for model stationarity
                if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                    return 1e6  # Large penalty for invalid parameters
                n = len(ret_window)

                # Initialize conditional variance with sample variance
                sigma2 = np.full(n, np.var(ret_window))
                sigma2[0] = np.var(ret_window[:min(10, n)])

                # GARCH recursion: σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
                for t in range(1, n):
                    sigma2[t] = omega + alpha * ret_window[t-1]**2 + beta * sigma2[t-1]
                    sigma2[t] = max(sigma2[t], 1e-8)  # Numerical stability

                # Calculate negative log-likelihood
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + ret_window**2 / sigma2)

                return -log_likelihood

            initial_guess = [np.var(ret_window) * 0.1, 0.1, 0.8]

            bounds = [(1e-8, None), (0, 1), (0, 1)]

            try:
                # Optimize params
                result = minimize(garch_likelihood, initial_guess, bounds=bounds, method="L-BFGS-B")

                if result.success and result.x[1] + result.x[2] < 0.999:
                    return result.x
                else:
                    # Fallback to simple estimates
                    return [np.var(ret_window) * 0.05, 0.05, 0.90]
            except:
                # Emergency fallback
                return [np.var(ret_window) * 0.05, 0.05, 0.90]
        
        # Rolling Window GARCH Calculation
        start_idx = min_periods
        
        for i in range(start_idx, n_obs):
            # Window boundaries
            window_start = max(0, i - window + 1)
            window_end = i + 1
            
            ret_window = returns_array[window_start:window_end]
            
            if i == start_idx or i % max(1, window // 4) == 0:
                omega, alpha, beta = estimate_garch_params(ret_window)
            
            # GARCH Volatility Calculation
            window_length = len(ret_window)
            sigma2_window = np.zeros(window_length)
            
            # Initial variance estimate
            sigma2_window[0] = np.var(ret_window[:min(10, window_length)])
            
            # Apply GARCH recursion for entire window
            for t in range(1, window_length):
                sigma2_window[t] = omega + alpha * ret_window[t-1]**2 + beta * sigma2_window[t-1]
                
                # Numerical stability
                sigma2_window[t] = max(sigma2_window[t], 1e-8)
            
            current_vol = np.sqrt(sigma2_window[-1])
            
            # Map back to original index position
            original_idx = returns.index[i]
            full_idx_position = full_index.get_loc(original_idx)
            volatility[full_idx_position] = current_vol
        
        return pd.Series(volatility, index=full_index)
