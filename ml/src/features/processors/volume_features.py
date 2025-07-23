#!/usr/bin/env python3

"""
Volume-based features processor
"""

import numpy as np
import pandas as pd
from ..base import BaseFeatureProcessor, safe_divide


class VolumeFeaturesProcessor(BaseFeatureProcessor):
    """
    Processor for volume-based features
    """
    
    def __init__(self):
        super().__init__("volume_features")
        self.feature_names = [
            "volume_sma_20", "volume_ratio", "relative_volume",
            "vwap_5", "vwap_20", "vwap_deviation", "volume_price_trend",
            "obv", "obv_normalized", "money_flow_index"
        ]
        self.dependencies = ["open", "high", "low", "close", "volume"]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume features"""
        self.validate_dependencies(data)
        result = data.copy()
        
        # Basic volume statistics
        result["volume_sma_20"] = result["volume"].rolling(window=20).mean()
        result["volume_ratio"] = safe_divide(result["volume"], result["volume_sma_20"], 1.0)
        result["relative_volume"] = safe_divide(
            result["volume"], result["volume"].rolling(20).median(), 1.0
        )
        
        # VWAP calculations
        result["vwap_5"] = safe_divide(
            (result["close"] * result["volume"]).rolling(5).sum(),
            result["volume"].rolling(5).sum(),
            result["close"]
        )
        result["vwap_20"] = safe_divide(
            (result["close"] * result["volume"]).rolling(20).sum(),
            result["volume"].rolling(20).sum(),
            result["close"]
        )
        result["vwap_deviation"] = safe_divide(
            result["close"] - result["vwap_20"],
            result["vwap_20"], 0
        )
        
        # Volume Price Trend
        price_change = result["close"] - result["close"].shift(1)
        result["volume_price_trend"] = (price_change * result["volume"]).rolling(5).sum()
        
        # On-Balance Volume (simplified vectorized version)
        result = self._calculate_obv(result)
        
        # Money Flow Index
        result = self._calculate_money_flow_index(result)
        
        return result
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume"""
        # Vectorized OBV calculation
        price_change = data["close"].diff()
        volume_direction = np.where(price_change > 0, data["volume"],
                                  np.where(price_change < 0, -data["volume"], 0))
        
        data["obv"] = volume_direction.cumsum()
        data["obv_normalized"] = safe_divide(
            data["obv"], data["obv"].rolling(20).mean(), 1.0
        )
        
        return data
    
    def _calculate_money_flow_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Money Flow Index"""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        money_flow = typical_price * data["volume"]
        
        # Positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = money_flow.where(price_change > 0, 0).rolling(14).sum()
        negative_flow = money_flow.where(price_change < 0, 0).rolling(14).sum()
        
        money_flow_ratio = safe_divide(positive_flow, negative_flow, 1.0)
        data["money_flow_index"] = 100 - (100 / (1 + money_flow_ratio))
        
        return data