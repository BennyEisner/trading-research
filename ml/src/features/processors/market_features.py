#!/usr/bin/env python3

"""
Market structure and regime features processor
"""

import numpy as np
import pandas as pd
from ..base import BaseFeatureProcessor, safe_divide


class MarketFeaturesProcessor(BaseFeatureProcessor):
    """
    Processor for market structure, regime, and contextual features
    """
    
    def __init__(self, symbol="AAPL", market_data=None, mag7_data=None):
        super().__init__("market_features")
        self.symbol = symbol
        self.market_data = market_data  
        self.mag7_data = mag7_data
        
        self.feature_names = [
            # Market structure
            "liquidity_ratio", "market_impact", "buying_pressure", "spread_normalized",
            "support_20", "resistance_20", "price_vs_support", "price_vs_resistance",
            
            # Market context
            "vix_level", "spy_correlation", "market_regime", "sector_performance",
            
            # Cross-asset
            "mag7_correlation", "relative_strength", "beta_stability", "sector_rotation",
            
            # Risk factors
            "momentum_factor", "value_factor", "quality_factor", "size_factor", "profitability_factor",
            
            # Regime detection
            "volatility_regime", "trend_regime", "volume_regime", "market_stress",
            
            # Seasonality
            "day_of_week", "month_of_year", "earnings_proximity", "options_expiry"
        ]
        
        self.dependencies = [
            "open", "high", "low", "close", "volume", "daily_return",
            "volatility_20d", "volatility_60d", "momentum_20d"
        ]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market features"""
        result = data.copy()
        
        # Create missing dependencies if needed
        result = self._ensure_dependencies(result)
        
        # Market structure features
        result = self._calculate_market_structure(result)
        
        # Market context features
        result = self._calculate_market_context(result)
        
        # Cross-asset features
        result = self._calculate_cross_asset(result)
        
        # Risk factors
        result = self._calculate_risk_factors(result)
        
        # Regime detection
        result = self._calculate_regime_features(result)
        
        # Seasonality features
        result = self._calculate_seasonality(result)
        
        return result
    
    def _ensure_dependencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create missing dependencies"""
        result = data.copy()
        
        # Ensure volatility features exist
        if "volatility_20d" not in result.columns and "daily_return" in result.columns:
            returns = result["daily_return"]
            result["volatility_20d"] = returns.rolling(20).std() * np.sqrt(252)
            result["volatility_60d"] = returns.rolling(60).std() * np.sqrt(252)
        
        # Ensure momentum features exist  
        if "momentum_20d" not in result.columns:
            result["momentum_20d"] = result["close"].pct_change(20)
            
        return result
    
    def _calculate_market_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features"""
        # Liquidity proxy
        data["liquidity_ratio"] = safe_divide(
            data["volume"], data["high"] - data["low"], 0
        )
        
        # Market impact proxy - FIXED: Use lagged return to prevent data leakage
        data["market_impact"] = safe_divide(
            abs(data["daily_return"].shift(1)), data["volume"], 0
        )
        
        # Buying pressure
        data["buying_pressure"] = safe_divide(
            data["close"] - data["low"], 
            data["high"] - data["low"], 0.5
        )
        
        # Spread normalized
        data["spread_normalized"] = safe_divide(
            data["high"] - data["low"], data["close"], 0
        )
        
        # Support and resistance levels
        data["support_20"] = data["low"].rolling(20).min()
        data["resistance_20"] = data["high"].rolling(20).max()
        data["price_vs_support"] = safe_divide(data["close"], data["support_20"], 1)
        data["price_vs_resistance"] = safe_divide(data["close"], data["resistance_20"], 1)
        
        return data
    
    def _calculate_market_context(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market context features"""
        # VIX level proxy (normalized volatility)
        data["vix_level"] = safe_divide(
            data["volatility_20d"],
            data["volatility_20d"].rolling(252).mean(), 1
        )
        
        # SPY correlation (simplified) - FIXED: Use only lagged returns
        data["spy_correlation"] = data["daily_return"].shift(1).rolling(20).corr(
            data["daily_return"].shift(2)
        ).fillna(0.5)
        
        # Market regime (simplified)
        data["market_regime"] = np.where(data["vix_level"] > 1.2, 1, 0)  # High vol regime
        
        # Sector performance (relative) - FIXED: Use lagged returns
        data["sector_performance"] = safe_divide(
            data["daily_return"].shift(1).rolling(20).mean(),
            data["daily_return"].shift(1).rolling(60).mean(), 1
        )
        
        return data
    
    def _calculate_cross_asset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-asset features (simplified)"""
        # Default values when external data unavailable
        data["mag7_correlation"] = 0.5
        data["relative_strength"] = 0.0
        data["beta_stability"] = 1.0
        data["sector_rotation"] = 0.0
        
        # Simple correlation proxy - FIXED: Use lagged returns
        data["mag7_correlation"] = data["daily_return"].shift(1).rolling(20).corr(
            data["daily_return"].shift(1).rolling(5).mean()
        ).fillna(0.5)
        
        # Relative strength - FIXED: Use lagged returns
        market_return = data["daily_return"].shift(1).rolling(20).mean()
        data["relative_strength"] = (data["daily_return"].shift(1) - market_return).rolling(10).mean()
        
        # Sector rotation
        data["sector_rotation"] = (
            data["relative_strength"] - data["relative_strength"].shift(5)
        ) / 5
        
        return data
    
    def _calculate_risk_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate academic risk factor exposures"""
        # Momentum factor
        data["momentum_factor"] = safe_divide(
            data["momentum_20d"], data["volatility_20d"], 0
        )
        
        # Value factor proxy - FIXED: Remove artificial negative bias
        data["value_factor"] = safe_divide(data["close"], data.get("sma_50", data["close"]), 1) - 1
        
        # Quality factor - FIXED: Use lagged returns
        data["quality_factor"] = safe_divide(
            data["daily_return"].shift(1).rolling(60).mean(),
            data["volatility_60d"], 0
        )
        
        # Size factor - FIXED: Remove hard-coded negative bias
        data["size_factor"] = 0.1
        
        # Profitability factor - FIXED: Use lagged returns
        data["profitability_factor"] = (data["daily_return"].shift(1) > 0).rolling(20).mean()
        
        return data
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime detection features"""
        # Volatility regime
        vol_20d = data["volatility_20d"]
        vol_low = vol_20d.rolling(252).quantile(0.33)
        vol_high = vol_20d.rolling(252).quantile(0.67)
        
        data["volatility_regime"] = np.where(
            vol_20d > vol_high, 2,  # High vol
            np.where(vol_20d < vol_low, 0, 1)  # Low/Medium vol
        )
        
        # Trend regime
        trend_strength = data.get("trend_strength_20d", data.get("sma5_to_sma20", 0))
        data["trend_regime"] = np.where(
            trend_strength > 0.01, 1,  # Uptrend
            np.where(trend_strength < -0.01, -1, 0)  # Downtrend/Sideways
        )
        
        # Volume regime
        vol_median = data["volume"].rolling(60).median()
        data["volume_regime"] = np.where(
            data["volume"] > vol_median * 1.5, 1,  # High volume
            np.where(data["volume"] < vol_median * 0.5, -1, 0)  # Low/Normal volume
        )
        
        # Market stress indicator - FIXED: Use lagged returns
        stress_conditions = [
            vol_20d > vol_20d.rolling(60).quantile(0.8),
            data["vix_level"] > 1.5,
            abs(data["daily_return"].shift(1)) > data["volatility_20d"] * 2
        ]
        data["market_stress"] = sum(stress_conditions)
        
        return data
    
    def _calculate_seasonality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate seasonality features"""
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if "date" in data.columns:
                # Convert date column to datetime index and drop the original column
                data = data.set_index(pd.to_datetime(data["date"]))
                data = data.drop(columns=["date"], errors='ignore')
            else:
                data.index = pd.date_range(start="2020-01-01", periods=len(data), freq="D")
        
        # Day of week effects
        data["day_of_week"] = data.index.dayofweek / 6.0  # Normalized
        
        # Month of year effects
        data["month_of_year"] = data.index.month / 12.0  # Normalized
        
        # Options expiry (3rd Friday approximation)
        data["options_expiry"] = (
            (data.index.day >= 15) & 
            (data.index.day <= 21) & 
            (data.index.dayofweek == 4)
        ).astype(int)
        
        # Earnings proximity (quarterly approximation)
        days_in_quarter = (data.index - data.index.to_period("Q").start_time).days
        data["earnings_proximity"] = np.exp(
            -0.1 * np.minimum(days_in_quarter, 90 - days_in_quarter)
        )
        
        return data