#!/usr/bin/env python3

"""
Basic price features processor
"""

import numpy as np
import pandas as pd
from ..base import BaseFeatureProcessor, safe_divide, calculate_returns


class PriceFeaturesProcessor(BaseFeatureProcessor):
    """
    Processor for basic price-based features
    """
    
    def __init__(self):
        super().__init__("price_features")
        self.feature_names = [
            "daily_return", "high_low_pct", "open_close_pct", "true_range",
            "price_position", "gap_normalized", "body_ratio", "upper_shadow", "lower_shadow",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_100",
            "ema_12", "ema_26", "ema_50", "ema_100", "hma_20",
            "price_to_sma5", "price_to_sma20", "price_to_sma50",
            "sma5_to_sma20", "sma20_to_sma50", "ema12_to_ema26"
        ]
        self.dependencies = ["open", "high", "low", "close", "volume"]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price features"""
        self.validate_dependencies(data)
        result = data.copy()
        
        # Basic price relationships
        result["daily_return"] = calculate_returns(result["close"], "simple")
        result["high_low_pct"] = (result["high"] - result["low"]) / result["close"] * 100
        result["open_close_pct"] = (result["close"] - result["open"]) / result["open"] * 100
        
        # True Range
        result["true_range"] = np.maximum(
            result["high"] - result["low"],
            np.maximum(
                abs(result["high"] - result["close"].shift(1)),
                abs(result["low"] - result["close"].shift(1))
            )
        )
        
        # Price position within daily range
        range_diff = result["high"] - result["low"]
        result["price_position"] = safe_divide(
            result["close"] - result["low"], range_diff, 0.5
        )
        
        # Intraday features
        result["gap_normalized"] = safe_divide(
            result["open"] - result["close"].shift(1),
            result["close"].shift(1), 0
        )
        result["body_ratio"] = safe_divide(
            abs(result["close"] - result["open"]),
            result["high"] - result["low"], 0.5
        )
        result["upper_shadow"] = safe_divide(
            result["high"] - np.maximum(result["close"], result["open"]),
            result["close"], 0
        )
        result["lower_shadow"] = safe_divide(
            np.minimum(result["close"], result["open"]) - result["low"],
            result["close"], 0
        )
        
        # Moving averages
        result = self._calculate_moving_averages(result)
        
        return result
    
    def _calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100]:
            data[f"sma_{period}"] = data["close"].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for span in [12, 26, 50, 100]:
            data[f"ema_{span}"] = data["close"].ewm(span=span).mean()
        
        # Hull Moving Average (simplified)
        data["hma_20"] = data["close"].rolling(20).mean()  # Simplified version
        
        # Normalized price features
        data["price_to_sma5"] = safe_divide(data["close"], data["sma_5"], 1) - 1
        data["price_to_sma20"] = safe_divide(data["close"], data["sma_20"], 1) - 1  
        data["price_to_sma50"] = safe_divide(data["close"], data["sma_50"], 1) - 1
        data["sma5_to_sma20"] = safe_divide(data["sma_5"], data["sma_20"], 1) - 1
        data["sma20_to_sma50"] = safe_divide(data["sma_20"], data["sma_50"], 1) - 1
        data["ema12_to_ema26"] = safe_divide(data["ema_12"], data["ema_26"], 1) - 1
        
        return data