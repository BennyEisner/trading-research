#!/usr/bin/env python3

"""
Momentum and trend features processor
"""

import numpy as np
import pandas as pd
from ..base import BaseFeatureProcessor, safe_divide


class MomentumFeaturesProcessor(BaseFeatureProcessor):
    """
    Processor for momentum and trend-based features
    """
    
    def __init__(self):
        super().__init__("momentum_features")
        self.feature_names = [
            "momentum_2d", "momentum_5d", "momentum_10d", "momentum_20d",
            "price_momentum_7d", "momentum_strength", "momentum_persistence",
            "trend_strength_5d", "trend_strength_10d", "trend_strength_20d",
            "roc_5", "roc_10", "momentum_oscillator"
        ]
        self.dependencies = ["close", "daily_return"]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features"""
        self.validate_dependencies(data)
        result = data.copy()
        
        # Basic momentum features (rate of change)
        for period in [2, 5, 10, 20]:
            result[f"momentum_{period}d"] = result["close"].pct_change(period)
        
        # Price momentum
        result["price_momentum_7d"] = safe_divide(
            result["close"] - result["close"].shift(7),
            result["close"].shift(7), 0
        )
        
        # Momentum strength (normalized by volatility)
        returns_5d = result["daily_return"].rolling(5).std()
        result["momentum_strength"] = safe_divide(
            result["momentum_5d"], returns_5d, 0
        )
        
        # Momentum persistence
        momentum_sign = np.sign(result["momentum_5d"])
        result["momentum_persistence"] = momentum_sign.rolling(10).mean()
        
        # Trend strength features (linear regression slope)
        for period in [5, 10, 20]:
            result[f"trend_strength_{period}d"] = result["close"].rolling(period).apply(
                self._calculate_trend_slope, raw=False
            )
        
        # Rate of Change indicators
        result["roc_5"] = ((result["close"] - result["close"].shift(5)) / 
                          result["close"].shift(5)) * 100
        result["roc_10"] = ((result["close"] - result["close"].shift(10)) / 
                           result["close"].shift(10)) * 100
        
        # Momentum oscillator
        result["momentum_oscillator"] = safe_divide(
            result["momentum_10d"] - result["momentum_20d"],
            result["momentum_10d"] + result["momentum_20d"], 0
        )
        
        return result
    
    def _calculate_trend_slope(self, price_series):
        """Calculate linear regression slope for trend strength"""
        try:
            if len(price_series) > 1:
                x = np.arange(len(price_series))
                slope = np.polyfit(x, price_series, 1)[0]
                return slope
            else:
                return 0
        except:
            return 0