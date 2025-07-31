#!/usr/bin/env python3

"""
Feature calculations for swing trading LSTM
All features optimized for 2-10 day holding periods with multi-ticker support
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import RobustScaler
import talib

from .utils.data_validation import DataValidator, clean_financial_data

warnings.filterwarnings("ignore")


class FeatureCalculator:
    """Calculate all pattern features for LSTM ensemble with multi-ticker support"""

    def __init__(self, symbol: str = "AAPL", market_data: Optional[pd.DataFrame] = None, 
                 sector_data: Optional[pd.DataFrame] = None, vix_data: Optional[pd.DataFrame] = None):
        self.symbol = symbol
        self.market_data = market_data  # SPY for beta calculations
        self.sector_data = sector_data  # Sector ETF data
        self.vix_data = vix_data  # VIX term structure data
        self.data_validator = DataValidator()

    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate complete feature set using modular processors"""
        print(f"Calculating feature set for {self.symbol}")

        # Validate and clean input data
        data = self._validate_data(data)
        
        features = pd.DataFrame(index=data.index)

        # Non-linear price patterns (4 features)
        features = self._calculate_non_linear_price_patterns(features, data)

        # Temporal dependencies (4 features)
        features = self._calculate_temporal_dependencies(features, data)

        # Market microstructure (3 features)
        features = self._calculate_microstructure_features(features, data)

        # Cross-asset relationships (3 features)
        features = self._calculate_cross_asset_relationships(features, data)

        # Core context features (5 features)
        features = self._calculate_core_context_features(features, data)

        # Clean and normalize final features
        features = self._clean_features(features)

        print(f"Feature calculation complete for {self.symbol}. Shape: {features.shape}")
        return features

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data using existing infrastructure"""
        
        # Use existing data validator
        validation_results = self.data_validator.validate_ohlcv_data(data, self.symbol)
        
        if not validation_results.get('data_quality_score', 0) > 0.7:
            print(f"Warning: Low data quality score for {self.symbol}: {validation_results['data_quality_score']:.2f}")
        
        # Clean data using existing infrastructure
        cleaned_data, cleaning_stats = clean_financial_data(
            data, 
            symbol=self.symbol,
            remove_outliers=True,
            outlier_threshold=5.0,
            fill_method="ffill"
        )
        
        if cleaning_stats['data_retention'] < 0.9:
            print(f"Warning: Significant data loss during cleaning for {self.symbol}: {cleaning_stats['data_retention']:.2%}")
        
        return cleaned_data

    def _calculate_non_linear_price_patterns(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate non-linear price pattern features (4 features)"""
        
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()
        
        # 1. Price acceleration (second derivative of price)
        price_momentum = returns.rolling(5).mean()
        features['price_acceleration'] = price_momentum.diff()
        
        # 2. Volume-price divergence
        volume_sma = volume.rolling(20).mean()
        volume_ratio = volume / volume_sma
        price_momentum_20 = returns.rolling(20).mean()
        # Divergence when volume increases but price momentum decreases (or vice versa)
        features['volume_price_divergence'] = volume_ratio * -price_momentum_20
        
        # 3. Volatility regime change
        volatility = returns.rolling(20).std()
        volatility_ma = volatility.rolling(60).mean()
        features['volatility_regime_change'] = (volatility - volatility_ma) / (volatility_ma + 1e-8)
        
        # 4. Return skewness (7-day rolling)
        features['return_skewness_7d'] = returns.rolling(7).skew()
        
        return features

    def _calculate_temporal_dependencies(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate temporal dependency features optimized for LSTM (4 features)"""
        
        close = data['close']
        returns = close.pct_change()
        
        # 1. Momentum persistence (7-day)
        momentum_5d = returns.rolling(5).mean()
        momentum_direction = np.sign(momentum_5d)
        # Count consecutive periods with same momentum direction
        features['momentum_persistence_7d'] = momentum_direction.rolling(7).apply(
            lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0, raw=False
        )
        
        # 2. Volatility clustering (GARCH-like effect)
        abs_returns = returns.abs()
        short_vol = abs_returns.rolling(5).mean()
        long_vol = abs_returns.rolling(20).mean()
        features['volatility_clustering'] = short_vol / (long_vol + 1e-8)
        
        # 3. Trend exhaustion (momentum vs volatility)
        momentum_strength = returns.rolling(10).mean().abs()
        volatility = returns.rolling(10).std()
        features['trend_exhaustion'] = momentum_strength / (volatility + 1e-8)
        
        # 4. GARCH volatility forecast
        features['garch_volatility_forecast'] = self._calculate_garch_forecast(returns)
        
        return features

    def _calculate_microstructure_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features (3 features)"""
        
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 1. Intraday range expansion (range breakouts)
        daily_range = high - low
        avg_range_20 = daily_range.rolling(20).mean()
        features['intraday_range_expansion'] = daily_range / (avg_range_20 + 1e-8)
        
        # 2. Overnight gap behavior
        overnight_gap = (open_price - close.shift(1)) / close.shift(1)
        features['overnight_gap_behavior'] = overnight_gap
        
        # 3. End of day momentum
        # Price position within daily range
        intraday_position = (close - low) / (high - low + 1e-8)
        features['end_of_day_momentum'] = intraday_position
        
        return features

    def _calculate_cross_asset_relationships(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-asset relationship features (3 features)"""
        
        close = data['close']
        returns = close.pct_change()
        
        # 1. Sector relative strength
        if self.sector_data is not None:
            sector_returns = self.sector_data['close'].pct_change()
            correlation = returns.rolling(60).corr(sector_returns)
            relative_performance = returns.rolling(20).mean() - sector_returns.rolling(20).mean()
            features['sector_relative_strength'] = relative_performance
        else:
            # Default to zero if no sector data
            features['sector_relative_strength'] = 0.0
        
        # 2. Market beta instability
        if self.market_data is not None:
            market_returns = self.market_data['close'].pct_change()
            # Rolling beta calculation
            rolling_beta = returns.rolling(60).apply(
                lambda x: self._calculate_beta(x, market_returns.loc[x.index]) if len(x) > 20 else np.nan,
                raw=False
            )
            beta_volatility = rolling_beta.rolling(20).std()
            features['market_beta_instability'] = beta_volatility
        else:
            # Default to zero if no market data
            features['market_beta_instability'] = 0.0
        
        # 3. VIX term structure
        if self.vix_data is not None:
            # VIX term structure slope (VIX9D - VIX)
            if 'vix9d' in self.vix_data.columns and 'vix' in self.vix_data.columns:
                vix_slope = self.vix_data['vix9d'] - self.vix_data['vix']
                # Align with stock data dates
                features['vix_term_structure'] = vix_slope.reindex(features.index, method='ffill')
            else:
                features['vix_term_structure'] = 0.0
        else:
            # Default to zero if no VIX data
            features['vix_term_structure'] = 0.0
        
        return features

    def _calculate_core_context_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate core context features for basic market information (5 features)"""
        
        close = data['close']
        volume = data['volume']
        
        # 1-3. Returns at different horizons
        features['returns_1d'] = close.pct_change()
        features['returns_3d'] = close.pct_change(periods=3)
        features['returns_7d'] = close.pct_change(periods=7)
        
        # 4. Volume normalized by 20-day average
        volume_ma = volume.rolling(20).mean()
        features['volume_normalized'] = volume / (volume_ma + 1e-8)
        
        # 5. Close price (for context)
        features['close'] = close
        
        return features

    def _calculate_garch_forecast(self, returns: pd.Series, alpha: float = 0.1, beta: float = 0.85) -> pd.Series:
        """Calculate GARCH(1,1) volatility forecast"""
        
        garch_forecast = pd.Series(index=returns.index, dtype=float)
        variance = returns.var()  # Initial variance estimate
        
        for i in range(20, len(returns)):  # Start after warmup period
            if not pd.isna(returns.iloc[i-1]):
                # GARCH(1,1) update: sigma^2_t = alpha * r^2_{t-1} + beta * sigma^2_{t-1} + gamma
                variance = alpha * returns.iloc[i-1]**2 + beta * variance + (1 - alpha - beta) * returns.iloc[:i].var()
                garch_forecast.iloc[i] = np.sqrt(variance)
        
        return garch_forecast

    def _calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        try:
            # Align series and remove NaN
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if len(aligned_data) < 10:  # Need sufficient data
                return np.nan
            
            stock_clean = aligned_data.iloc[:, 0]
            market_clean = aligned_data.iloc[:, 1]
            
            covariance = np.cov(stock_clean, market_clean)[0, 1]
            market_variance = np.var(market_clean)
            
            if market_variance == 0:
                return np.nan
            
            return covariance / market_variance
        except:
            return np.nan

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize final features"""
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Replace any remaining NaN with median values
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val if not pd.isna(median_val) else 0.0)
        
        # Replace infinite values with 99th percentile values
        for col in features.columns:
            if np.isinf(features[col]).any():
                finite_values = features[col][np.isfinite(features[col])]
                if len(finite_values) > 0:
                    clip_value = np.percentile(finite_values, 99)
                    features[col] = features[col].replace([np.inf, -np.inf], [clip_value, -clip_value])
        
        return features

    def get_feature_names(self) -> list:
        """Get list of all feature names"""
        return [
            # Non-linear price patterns
            'price_acceleration',
            'volume_price_divergence', 
            'volatility_regime_change',
            'return_skewness_7d',
            # Temporal dependencies
            'momentum_persistence_7d',
            'volatility_clustering',
            'trend_exhaustion',
            'garch_volatility_forecast',
            # Market microstructure
            'intraday_range_expansion',
            'overnight_gap_behavior',
            'end_of_day_momentum',
            # Cross-asset relationships
            'sector_relative_strength',
            'market_beta_instability',
            'vix_term_structure',
            # Core context
            'returns_1d',
            'returns_3d', 
            'returns_7d',
            'volume_normalized',
            'close'
        ]

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of each feature"""
        return {
            'price_acceleration': 'Second derivative of price momentum (acceleration)',
            'volume_price_divergence': 'Divergence between volume and price momentum',
            'volatility_regime_change': 'Change in volatility relative to long-term average',
            'return_skewness_7d': '7-day rolling skewness of returns',
            'momentum_persistence_7d': 'Consecutive periods with same momentum direction',
            'volatility_clustering': 'Short-term vs long-term volatility ratio',
            'trend_exhaustion': 'Momentum strength relative to volatility',
            'garch_volatility_forecast': 'GARCH(1,1) volatility forecast',
            'intraday_range_expansion': 'Daily range relative to 20-day average',
            'overnight_gap_behavior': 'Overnight price gap percentage',
            'end_of_day_momentum': 'Closing position within daily range',
            'sector_relative_strength': 'Performance relative to sector ETF',
            'market_beta_instability': 'Volatility of rolling beta coefficient',
            'vix_term_structure': 'VIX term structure slope (VIX9D - VIX)',
            'returns_1d': '1-day return',
            'returns_3d': '3-day return',
            'returns_7d': '7-day return',
            'volume_normalized': 'Volume relative to 20-day average',
            'close': 'Closing price for context'
        }
