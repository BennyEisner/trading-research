#!/usr/bin/env python3

"""
Feature engineering for financial time series data
"""


import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Handles all feature engineering for financial data"""

    def __init__(self, symbol="APPLE", market_data=None, mag7_data=None):
        self.symbol = symbol
        self.market_data = market_data
        self.mag7_data = mag7_data
        self.scalers = {}
        self.feature_importance = {}
        self.feature_selection_results = {}
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
        
        # Store feature groups for reference
        self.feature_groups = self.selected_features.copy()
        
        # Selected features for LSTM (will be updated by advanced selection methods)
        self.selected_features_list = [
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
        data = data.fillna(method="ffill").fillna(method="bfill")

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

        wma_half = series.rolling(window=half_period).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)))
        wma_full = series.rolling(window=period).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)))

        raw_hma = 2 * wma_half - wma_full

        hull_ma = raw_hma.rolling(window=sqrt_period).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)))

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
        data["vwap_deviation"] = (data["close"] - data["vwap_20"]) / data["vwap_20"]

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

    def _calculate_momentum_features(self, data):
        """Calculate momentum and trend features"""
        # Basic momentum features
        for period in [2, 5, 10, 20]:
            data[f"momentum_{period}d"] = data["close"].pct_change(period)
        
        # Price momentum (rate of change)
        data["price_momentum_7d"] = (data["close"] - data["close"].shift(7)) / data["close"].shift(7)
        
        # Momentum strength (normalized by volatility)
        returns_5d = data["daily_return"].rolling(5).std()
        data["momentum_strength"] = data["momentum_5d"] / (returns_5d + 1e-8)
        
        # Momentum persistence (how long momentum lasts)
        momentum_sign = np.sign(data["momentum_5d"])
        data["momentum_persistence"] = momentum_sign.rolling(10).sum() / 10
        
        # Trend strength features
        for period in [5, 10, 20]:
            # Linear regression slope as trend strength
            data[f"trend_strength_{period}d"] = data["close"].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
            )
        
        # Rate of Change (ROC) indicators
        data["roc_5"] = ((data["close"] - data["close"].shift(5)) / data["close"].shift(5)) * 100
        data["roc_10"] = ((data["close"] - data["close"].shift(10)) / data["close"].shift(10)) * 100
        
        # Momentum oscillator
        data["momentum_oscillator"] = (data["momentum_10d"] - data["momentum_20d"]) / (data["momentum_10d"] + data["momentum_20d"] + 1e-8)
        
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

        # Create comprehensive volatility features
        data["volatility_5d"] = (returns.rolling(5).std() * np.sqrt(252)).reindex(data.index)
        data["volatility_20d"] = (returns.rolling(20).std() * np.sqrt(252)).reindex(data.index)
        data["volatility_60d"] = (returns.rolling(60).std() * np.sqrt(252)).reindex(data.index)

        # Volatility ratio (short-term vs long-term)
        volatility_ratio = data["volatility_5d"] / (data["volatility_20d"] + 1e-8)
        data["volatility_ratio"] = volatility_ratio

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
                sigma2[0] = np.var(ret_window[: min(10, n)])

                # GARCH recursion: σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
                for t in range(1, n):
                    sigma2[t] = omega + alpha * ret_window[t - 1] ** 2 + beta * sigma2[t - 1]
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
            sigma2_window[0] = np.var(ret_window[: min(10, window_length)])

            # Apply GARCH recursion for entire window
            for t in range(1, window_length):
                sigma2_window[t] = omega + alpha * ret_window[t - 1] ** 2 + beta * sigma2_window[t - 1]

                # Numerical stability
                sigma2_window[t] = max(sigma2_window[t], 1e-8)

            current_vol = np.sqrt(sigma2_window[-1])

            # Map back to original index position
            original_idx = returns.index[i]
            full_idx_position = full_index.get_loc(original_idx)
            volatility[full_idx_position] = current_vol

        return pd.Series(volatility, index=full_index)

    def _calculate_market_structure_features(self, data):
        """Market microstructure and liquidity features"""
        # Liquidity proxy features
        data["liquidity_ratio"] = data["volume"] / (data["high"] - data["low"] + 1e-10)

        # Market impact proxy
        data["market_impact"] = abs(data["daily_return"]) / (data["volume"] + 1e-10)

        # Price efficiency measures
        data["price_efficiency"] = abs(data["close"] - data["vwap_20"]) / data["close"]

        # Support and resistance
        data["resistance_20"] = data["high"].rolling(window=20).max()
        data["support_20"] = data["low"].rolling(window=20).min()
        data["price_vs_resistance"] = data["close"] / data["resistance_20"]
        data["price_vs_support"] = data["close"] / data["support_20"]
        
        return data

    def _calculate_risk_factors(self, data):
        """Academic risk factor exposures"""
        # Momentum factor (simplified)
        data["momentum_factor"] = data["momentum_20d"] / data["volatility_20d"]

        # Value factor proxy - FIXED: Remove artificial negative bias
        data["value_factor"] = data["price_to_sma200"] if "sma_200" in data.columns else data["price_to_sma50"]

        # Quality factor (using return stability)
        data["quality_factor"] = data["daily_return"].rolling(60).mean() / data["volatility_60d"]

        # Size factor - FIXED: Remove hard-coded negative bias
        data["size_factor"] = 0.1  # Positive for large cap

        # Profitability factor (using return consistency)
        data["profitability_factor"] = (data["daily_return"] > 0).rolling(20).mean()

        # Low volatility factor - FIXED: Remove artificial negative bias  
        data["low_vol_factor"] = data["volatility_20d"] / data["volatility_20d"].rolling(60).mean()

        return data

    def _calculate_market_context_features(self, data):
        """Market context features"""
        # VIX level proxy (normalized volatility)
        data["vix_level"] = data["volatility_20d"] / (data["volatility_20d"].rolling(252).mean() + 1e-8)
        
        # SPY correlation (simplified - using rolling correlation with own returns as proxy)
        data["spy_correlation"] = data["daily_return"].rolling(20).corr(data["daily_return"].shift(1)).fillna(0.5)
        
        # Market regime indicator (combines volatility and trend)
        # Use fillna with scalar values instead of .get() which returns scalars
        vol_regime = data.get("volatility_regime", pd.Series([1] * len(data), index=data.index))
        trend_regime = data.get("trend_regime", pd.Series([0] * len(data), index=data.index))
        data["market_regime"] = vol_regime + trend_regime
        
        # Yield curve slope proxy (placeholder - would use real data in production)
        data["yield_curve_slope"] = np.sin(np.arange(len(data)) * 2 * np.pi / 252) * 0.1  # Seasonal proxy
        
        # Sector performance (relative to own performance)
        data["sector_performance"] = data["daily_return"].rolling(20).mean() / (data["daily_return"].rolling(60).mean() + 1e-8)
        
        return data

    def _calculate_cross_asset_features(self, data):
        """Cross-asset and relative performance features"""
        # Mag 7 correlation features
        if self.mag7_data is not None:
            mag7_returns = []
            for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]:
                if symbol != self.symbol and symbol in self.mag7_data:
                    mag7_returns.append(self.mag7_data[symbol].get("returns", pd.Series()))

            if mag7_returns:
                avg_mag7_returns = pd.concat(mag7_returns, axis=1).mean(axis=1)
                data["mag7_correlation"] = data["daily_return"].rolling(20).corr(avg_mag7_returns)
                data["relative_strength"] = (data["daily_return"] - avg_mag7_returns).rolling(10).mean()
            else:
                data["mag7_correlation"] = 0.5
                data["relative_strength"] = 0.0
        else:
            data["mag7_correlation"] = 0.5
            data["relative_strength"] = 0.0

        # Beta calculation (relative to market)
        if self.market_data is not None and "SPY" in self.market_data:
            spy_returns = self.market_data["SPY"].get("returns", pd.Series())
            if not spy_returns.empty:
                # Rolling beta calculation
                covariance = data["daily_return"].rolling(60).cov(spy_returns)
                market_variance = spy_returns.rolling(60).var()
                data["beta_stability"] = covariance / (market_variance + 1e-10)
            else:
                data["beta_stability"] = 1.0
        else:
            data["beta_stability"] = 1.0

        # Idiosyncratic volatility
        market_component = data["beta_stability"] * data.get("spy_correlation", 0.5) * data["volatility_20d"]
        data["idiosyncratic_vol"] = np.sqrt(np.maximum(0, data["volatility_20d"] ** 2 - market_component**2))

        # Sector rotation proxy
        data["sector_rotation"] = (data["relative_strength"] - data["relative_strength"].shift(5)) / 5

        return data

    def _calculate_regime_features(self, data):
        """Advanced regime detection features"""
        # Volatility regime using quantiles
        vol_20d = data["volatility_20d"]
        vol_low = vol_20d.rolling(252).quantile(0.33)
        vol_high = vol_20d.rolling(252).quantile(0.67)
        
        data["volatility_regime"] = np.where(
            vol_20d > vol_high, 2,  # High vol
            np.where(vol_20d < vol_low, 0, 1)  # Low/Medium vol
        )

        # Trend regime
        trend_strength = data["trend_strength_20d"] if "trend_strength_20d" in data.columns else data["sma5_to_sma20"]
        data["trend_regime"] = np.where(trend_strength > 0.01, 1, np.where(trend_strength < -0.01, -1, 0))  # Uptrend  # Downtrend/Sideways

        # Volume regime
        vol_median = data["volume"].rolling(60).median()
        data["volume_regime"] = np.where(
            data["volume"] > vol_median * 1.5, 1, np.where(data["volume"] < vol_median * 0.5, -1, 0)  # High volume  # Low/Normal volume
        )

        # Correlation regime
        if "spy_correlation" in data.columns:
            data["correlation_regime"] = np.where(
                data["spy_correlation"] > 0.7,
                1,  # High correlation
                np.where(data["spy_correlation"] < 0.3, -1, 0),  # Low/Medium correlation
            )
        else:
            data["correlation_regime"] = 0

        # Market stress indicator
        stress_indicators = [
            data["volatility_20d"] > data["volatility_20d"].rolling(60).quantile(0.8),
            data.get("vix_level", 1) > 1.5,
            abs(data["daily_return"]) > data["volatility_20d"] * 2,
        ]
        data["market_stress"] = sum(stress_indicators)

        return data

    def _calculate_seasonality_features(self, data):
        """Calendar and seasonal effects"""
        # Ensure we have a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            # Try to create datetime index from 'date' column if it exists
            if "date" in data.columns:
                # Convert date column to datetime index and drop the original column
                data = data.set_index(pd.to_datetime(data["date"]))
                data = data.drop(columns=["date"], errors='ignore')
            else:
                # Create a simple date range if no date information available
                data.index = pd.date_range(start="2020-01-01", periods=len(data), freq="D")

        # Day of week effects
        data["day_of_week"] = data.index.dayofweek / 6.0  # Normalized 0-1
        data["monday_effect"] = (data.index.dayofweek == 0).astype(int)
        data["friday_effect"] = (data.index.dayofweek == 4).astype(int)

        # Month of year effects
        data["month_of_year"] = data.index.month / 12.0  # Normalized 0-1
        data["january_effect"] = (data.index.month == 1).astype(int)
        data["december_effect"] = (data.index.month == 12).astype(int)

        # Quarter effects
        data["quarter"] = data.index.quarter / 4.0
        data["quarter_end"] = data.index.is_quarter_end.astype(int)

        # Options expiry effect (3rd Friday approximation)
        data["options_expiry"] = ((data.index.day >= 15) & (data.index.day <= 21) & (data.index.dayofweek == 4)).astype(int)

        # Earnings proximity (quarterly approximation)
        days_in_quarter = (data.index - data.index.to_period("Q").start_time).days
        data["earnings_proximity"] = np.exp(-0.1 * np.minimum(days_in_quarter, 90 - days_in_quarter))

        # Month-end rebalancing effect
        data["month_end"] = data.index.is_month_end.astype(int)
        data["rebalancing_effect"] = ((data.index.day >= 25) | (data.index.day <= 5)).astype(int)

        return data

    def _calculate_alternative_data_features(self, data):
        """Alternative data features (simulated when real data unavailable)"""
        # Sentiment score (simulated based on returns and volatility)
        data["sentiment_score"] = np.tanh(data["daily_return"].rolling(5).mean() / data["volatility_5d"])

        # News flow intensity (simulated)
        data["news_flow"] = abs(data["daily_return"]) / data["volatility_20d"]
        data["news_flow"] = np.clip(data["news_flow"], 0, 3)  # Cap at reasonable levels

        # Social sentiment (simulated)
        data["social_sentiment"] = (data["momentum_5d"] + data["sentiment_score"]) / 2

        # Search volume proxy (simulated based on volatility and returns)
        data["search_volume"] = (abs(data["daily_return"]) + data["volatility_5d"]) / 2

        # Analyst revisions proxy (simulated)
        data["analyst_revisions"] = data["momentum_20d"].rolling(10).mean()

        return data

    def _create_lstm_optimized_features(self, data):
        """Create features specifically optimized for LSTM models"""
        # Sequence-aware features
        data["return_sequence_sum_5"] = data["daily_return"].rolling(5).sum()
        data["return_sequence_sum_10"] = data["daily_return"].rolling(10).sum()

        # Temporal dependencies
        data["volatility_persistence"] = data["volatility_5d"].rolling(10).std()
        data["trend_persistence"] = data["trend_strength_5d"].rolling(10).std()

        # Regime transition indicators
        data["regime_change"] = (data["volatility_regime"] != data["volatility_regime"].shift(1)).astype(int)
        data["trend_change"] = (data["trend_regime"] != data["trend_regime"].shift(1)).astype(int)

        # Memory features (important for LSTM)
        data["return_memory_5"] = data["daily_return"].rolling(5).apply(lambda x: np.corrcoef(x, range(len(x)))[0, 1] if len(x) > 1 else 0)
        data["volume_memory_5"] = data["volume_ratio"].rolling(5).apply(lambda x: np.corrcoef(x, range(len(x)))[0, 1] if len(x) > 1 else 0)

        # Lagged features to prevent look-ahead bias
        lag_features = ["daily_return", "rsi", "volume_ratio", "volatility_ratio", "macd_histogram", "bb_position", "momentum_5d"]

        for feature in lag_features:
            if feature in data.columns:
                for lag in [1, 2, 3, 5, 10]:
                    data[f"{feature}_lag{lag}"] = data[feature].shift(lag)

        return data

    def _apply_feature_scaling(self, data):
        """Apply appropriate scaling for LSTM models"""
        # Identify numeric columns (excluding target if present)
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ["close", "open", "high", "low", "volume"]  # Keep raw price data

        # Features to scale
        features_to_scale = [col for col in numeric_columns if col not in exclude_cols]

        # Apply robust scaling to handle outliers
        for col in features_to_scale:
            if col in data.columns and data[col].notna().sum() > 10:
                scaler = RobustScaler()
                data[col] = scaler.fit_transform(data[[col]]).flatten()
                self.scalers[col] = scaler

        return data

    def select_features_for_lstm(self, data, target_col="daily_return", max_features=50, method="ensemble"):
        """
        Advanced feature selection for LSTM model using multiple methods
        
        Args:
            data: DataFrame with features
            target_col: Target column name
            max_features: Maximum number of features to select
            method: Selection method ('ensemble', 'rf', 'lasso', 'rfe', 'statistical')
            
        Returns:
            List of selected feature names
        """
        print(f"Starting advanced feature selection using {method} method...")
        
        # Get available features from our selected list
        available_features = [col for col in self.selected_features_list if col in data.columns]

        if len(available_features) == 0:
            print("Warning: No selected features found in data")
            return []

        # Remove target from features if present
        if target_col in available_features:
            available_features.remove(target_col)

        # Prepare data for feature selection
        feature_data = data[available_features].fillna(0)
        target_data = data[target_col].fillna(0)

        # Remove rows where target is NaN
        valid_indices = target_data.notna()
        feature_data = feature_data[valid_indices]
        target_data = target_data[valid_indices]

        if len(feature_data) < 100:  # Not enough data for advanced selection
            print("Warning: Insufficient data for advanced selection, using basic method")
            return self._basic_feature_selection(feature_data, target_data, available_features, max_features)

        # Apply selected method
        if method == "ensemble":
            return self._ensemble_feature_selection(feature_data, target_data, available_features, max_features)
        elif method == "rf":
            return self._random_forest_selection(feature_data, target_data, available_features, max_features)
        elif method == "lasso":
            return self._lasso_feature_selection(feature_data, target_data, available_features, max_features)
        elif method == "rfe":
            return self._rfe_feature_selection(feature_data, target_data, available_features, max_features)
        elif method == "cv":
            return self._cross_validation_selection(feature_data, target_data, available_features, max_features)
        else:
            return self._basic_feature_selection(feature_data, target_data, available_features, max_features)

    def _ensemble_feature_selection(self, X, y, feature_names, max_features):
        """
        Ensemble feature selection combining multiple methods
        """
        print("Running ensemble feature selection...")
        
        selection_results = {}
        methods = ['rf', 'lasso', 'rfe', 'statistical']
        
        for method in methods:
            try:
                if method == 'rf':
                    selected = self._random_forest_selection(X, y, feature_names, max_features * 2)
                elif method == 'lasso':
                    selected = self._lasso_feature_selection(X, y, feature_names, max_features * 2)
                elif method == 'rfe':
                    selected = self._rfe_feature_selection(X, y, feature_names, max_features * 2)
                else:
                    selected = self._basic_feature_selection(X, y, feature_names, max_features * 2)
                
                selection_results[method] = selected
                print(f"{method.upper()}: {len(selected)} features selected")
                
            except Exception as e:
                print(f"Warning: {method} selection failed: {e}")
                continue
        
        # Combine results using voting
        feature_votes = {}
        for method, features in selection_results.items():
            for feature in features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Select features that appear in at least 2 methods
        min_votes = max(1, len(selection_results) // 2)
        selected_features = [f for f, votes in feature_votes.items() if votes >= min_votes]
        
        # If not enough features, add top-voted ones
        if len(selected_features) < max_features:
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            for feature, _ in sorted_features:
                if feature not in selected_features:
                    selected_features.append(feature)
                    if len(selected_features) >= max_features:
                        break
        
        # Limit to max_features
        selected_features = selected_features[:max_features]
        
        print(f"Ensemble selection: {len(selected_features)} features selected")
        self.feature_selection_results['ensemble'] = {
            'selected_features': selected_features,
            'method_results': selection_results,
            'feature_votes': feature_votes
        }
        
        return selected_features

    def _random_forest_selection(self, X, y, feature_names, max_features):
        """
        Random Forest based feature importance selection
        """
        try:
            rf = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            )
            rf.fit(X, y)
            
            # Get feature importance
            importance_scores = dict(zip(feature_names, rf.feature_importances_))
            
            # Select top features
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f for f, _ in sorted_features[:max_features]]
            
            self.feature_importance.update(importance_scores)
            return selected_features
            
        except Exception as e:
            print(f"Random Forest selection failed: {e}")
            return feature_names[:max_features]

    def _lasso_feature_selection(self, X, y, feature_names, max_features):
        """
        LASSO regularization based feature selection
        """
        try:
            # Use cross-validation to find optimal alpha
            lasso_cv = LassoCV(cv=5, random_state=42, max_iter=2000)
            lasso_cv.fit(X, y)
            
            # Get non-zero coefficients
            lasso_coef = np.abs(lasso_cv.coef_)
            feature_importance = dict(zip(feature_names, lasso_coef))
            
            # Select features with non-zero coefficients
            selected_features = [f for f, coef in feature_importance.items() if coef > 1e-6]
            
            # If too many features, select top ones by coefficient magnitude
            if len(selected_features) > max_features:
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f for f, _ in sorted_features[:max_features] if f in selected_features]
            
            return selected_features
            
        except Exception as e:
            print(f"LASSO selection failed: {e}")
            return feature_names[:max_features]

    def _rfe_feature_selection(self, X, y, feature_names, max_features):
        """
        Recursive Feature Elimination with Cross-Validation
        """
        try:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Use time series cross-validation for financial data
            tscv = TimeSeriesSplit(n_splits=3)
            
            rfe_cv = RFECV(
                estimator=estimator,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            rfe_cv.fit(X, y)
            
            # Get selected features
            selected_mask = rfe_cv.support_
            selected_features = [f for f, selected in zip(feature_names, selected_mask) if selected]
            
            # Limit to max_features if needed
            if len(selected_features) > max_features:
                # Use feature ranking to select top features
                feature_ranking = dict(zip(feature_names, rfe_cv.ranking_))
                sorted_features = sorted(feature_ranking.items(), key=lambda x: x[1])
                selected_features = [f for f, _ in sorted_features[:max_features]]
            
            return selected_features
            
        except Exception as e:
            print(f"RFE selection failed: {e}")
            return feature_names[:max_features]

    def _cross_validation_selection(self, X, y, feature_names, max_features):
        """
        Cross-validation based feature selection
        """
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            feature_scores = {}
            
            for feature_idx, feature in enumerate(feature_names):
                # Test each feature individually using cross-validation
                feature_data = X.iloc[:, [feature_idx]]
                
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                cv_scores = cross_val_score(rf, feature_data, y, cv=tscv, scoring='neg_mean_squared_error')
                feature_scores[feature] = np.mean(cv_scores)
            
            # Select top features
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f for f, _ in sorted_features[:max_features]]
            
            return selected_features
            
        except Exception as e:
            print(f"CV selection failed: {e}")
            return feature_names[:max_features]

    def _basic_feature_selection(self, X, y, feature_names, max_features):
        """
        Basic statistical feature selection (fallback method)
        """
        try:
            selector = SelectKBest(score_func=f_regression, k=min(max_features, len(feature_names)))
            selector.fit(X, y)

            selected_mask = selector.get_support()
            selected_features = [feat for feat, selected in zip(feature_names, selected_mask) if selected]

            feature_scores = dict(zip(feature_names, selector.scores_))
            self.feature_importance.update(feature_scores)

            return selected_features
            
        except Exception as e:
            print(f"Basic selection failed: {e}")
            return feature_names[:max_features]

    def get_feature_importance(self):
        """Return feature importance scores"""
        return self.feature_importance

    def get_feature_selection_results(self):
        """Return detailed feature selection results"""
        return self.feature_selection_results

    def analyze_feature_selection(self, data, target_col="daily_return"):
        """
        Comprehensive analysis of different feature selection methods
        """
        print("Performing comprehensive feature selection analysis...")
        
        methods = ['statistical', 'rf', 'lasso', 'rfe', 'cv', 'ensemble']
        analysis_results = {}
        
        for method in methods:
            print(f"\nTesting {method.upper()} method...")
            try:
                selected_features = self.select_features_for_lstm(
                    data, target_col=target_col, max_features=30, method=method
                )
                
                # Evaluate feature set performance
                if len(selected_features) > 0:
                    performance = self._evaluate_feature_set(
                        data, selected_features, target_col
                    )
                    
                    analysis_results[method] = {
                        'selected_features': selected_features,
                        'n_features': len(selected_features),
                        'performance': performance
                    }
                    
                    print(f"{method.upper()}: {len(selected_features)} features, "
                          f"CV Score: {performance['cv_score']:.4f}")
                else:
                    print(f"{method.upper()}: No features selected")
                    
            except Exception as e:
                print(f"{method.upper()} failed: {e}")
                continue
        
        # Find best method
        if analysis_results:
            best_method = max(analysis_results.keys(), 
                            key=lambda x: analysis_results[x]['performance']['cv_score'])
            
            print(f"\nBest method: {best_method.upper()}")
            print(f"Best CV Score: {analysis_results[best_method]['performance']['cv_score']:.4f}")
            print(f"Features selected: {analysis_results[best_method]['n_features']}")
            
            self.feature_selection_results['analysis'] = analysis_results
            self.feature_selection_results['best_method'] = best_method
            
            return analysis_results[best_method]['selected_features']
        
        return []

    def _evaluate_feature_set(self, data, selected_features, target_col):
        """
        Evaluate performance of selected feature set using cross-validation
        """
        try:
            # Prepare data
            available_features = [f for f in selected_features if f in data.columns]
            if len(available_features) == 0:
                return {'cv_score': -999, 'n_features_available': 0}
            
            X = data[available_features].fillna(0)
            y = data[target_col].fillna(0)
            
            # Remove rows where target is NaN
            valid_indices = y.notna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) < 50:
                return {'cv_score': -999, 'n_features_available': len(available_features)}
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Use Random Forest for evaluation
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            cv_scores = cross_val_score(
                rf, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            return {
                'cv_score': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'n_features_available': len(available_features)
            }
            
        except Exception as e:
            print(f"Feature evaluation failed: {e}")
            return {'cv_score': -999, 'n_features_available': 0}

    def get_feature_groups(self):
        """Return organized feature groups"""
        return self.feature_groups

    def prepare_lstm_sequences(self, data, feature_columns, target_column, sequence_length=60):
        """
        Prepare sequences for LSTM training
        """
        # Ensure we have enough data
        if len(data) < sequence_length:
            raise ValueError(f"Not enough data. Need at least {sequence_length} samples")

        # Get feature and target data
        features = data[feature_columns].fillna(0).values
        targets = data[target_column].fillna(0).values

        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(features[i - sequence_length : i])
            y.append(targets[i])

        return np.array(X), np.array(y)

    def validate_features(self, data):
        """Validate feature quality and provide diagnostics"""
        issues = []

        # Check for features with too many NaN values
        nan_threshold = 0.1  # 10% threshold
        for col in data.columns:
            nan_ratio = data[col].isna().sum() / len(data)
            if nan_ratio > nan_threshold:
                issues.append(f"Feature '{col}' has {nan_ratio:.2%} NaN values")

        # Check for features with zero variance
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].var() == 0:
                issues.append(f"Feature '{col}' has zero variance")

        # Check for highly correlated features
        corr_matrix = data[numeric_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

        if high_corr_pairs:
            issues.append(f"High correlation (>0.95) found between: {high_corr_pairs}")

        return issues

    def get_feature_summary(self, data):
        """Get comprehensive feature summary"""
        summary = {
            "total_features": len(data.columns),
            "numeric_features": len(data.select_dtypes(include=[np.number]).columns),
            "missing_data": data.isna().sum().sum(),
            "date_range": f"{data.index.min()} to {data.index.max()}" if isinstance(data.index, pd.DatetimeIndex) else "No date index",
            "sample_size": len(data),
        }

        # Feature group breakdown
        for group_name, features in self.feature_groups.items():
            available = [f for f in features if f in data.columns]
            summary[f"{group_name}_features"] = f"{len(available)}/{len(features)} available"

        return summary
