#!/usr/bin/env python3

"""
Technical indicators processor
"""

import numpy as np
import pandas as pd
from ..base import BaseFeatureProcessor, safe_divide


class TechnicalIndicatorsProcessor(BaseFeatureProcessor):
    """
    Processor for technical analysis indicators
    """
    
    def __init__(self):
        super().__init__("technical_indicators")
        self.feature_names = [
            "rsi_14", "rsi_21", "rsi",
            "macd", "macd_signal", "macd_histogram", 
            "stoch_k", "cci", "bb_upper", "bb_lower", "bb_position", 
            "bb_width", "bb_squeeze", "atr", "atr_ratio", "williams_r"
        ]
        self.dependencies = ["open", "high", "low", "close", "volume", "ema_12", "ema_26"]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        self.validate_dependencies(data)
        result = data.copy()
        
        # RSI (Relative Strength Index)
        result = self._calculate_rsi(result)
        
        # MACD (Moving Average Convergence Divergence)
        result = self._calculate_macd(result)
        
        # Stochastic Oscillator
        result = self._calculate_stochastic(result)
        
        # Commodity Channel Index (CCI)
        result = self._calculate_cci(result)
        
        # Bollinger Bands
        result = self._calculate_bollinger_bands(result)
        
        # Average True Range (ATR)
        result = self._calculate_atr(result)
        
        # Williams %R
        result = self._calculate_williams_r(result)
        
        return result
    
    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI for multiple periods with proper NaN handling"""
        for period in [14, 21]:
            delta = data["close"].diff()
            
            # Proper RSI calculation with NaN handling
            gain = delta.where(delta > 0, 0)  # Set negative values to 0
            loss = delta.where(delta < 0, 0).abs()  # Set positive values to 0, take abs
            
            # Use EMA-based RSI calculation (more stable)
            alpha = 1.0 / period
            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
            
            rs = safe_divide(avg_gain, avg_loss, 1.0)
            rsi = 100 - (100 / (1 + rs))
            
            # Fill initial NaN values with neutral RSI (50)
            data[f"rsi_{period}"] = rsi.fillna(50.0)
        
        data["rsi"] = data["rsi_14"]  # Default RSI
        return data
    
    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        macd_line = data["ema_12"] - data["ema_26"]
        signal_line = macd_line.ewm(span=9).mean()
        
        data["macd"] = macd_line
        data["macd_signal"] = signal_line
        data["macd_histogram"] = macd_line - signal_line
        
        return data
    
    def _calculate_stochastic(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        lowest_low = data["low"].rolling(window=14).min()
        highest_high = data["high"].rolling(window=14).max()
        
        stoch_k_raw = 100 * safe_divide(
            data["close"] - lowest_low,
            highest_high - lowest_low,
            0.5
        )
        data["stoch_k"] = stoch_k_raw.rolling(window=3).mean()
        
        return data
    
    def _calculate_cci(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Commodity Channel Index"""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        data["cci"] = safe_divide(typical_price - sma_tp, 0.015 * mad, 0)
        
        return data
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        bb_middle = data["close"].rolling(window=20).mean()
        bb_std = data["close"].rolling(window=20).std()
        
        data["bb_upper"] = bb_middle + (bb_std * 2)
        data["bb_lower"] = bb_middle - (bb_std * 2)
        
        bb_range = data["bb_upper"] - data["bb_lower"]
        data["bb_position"] = safe_divide(
            data["close"] - data["bb_lower"],
            bb_range,
            0.5
        )
        data["bb_width"] = safe_divide(bb_range, data["close"], 0)
        data["bb_squeeze"] = (
            data["bb_width"] < data["bb_width"].rolling(20).quantile(0.1)
        ).astype(int)
        
        return data
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range"""
        if "true_range" in data.columns:
            data["atr"] = data["true_range"].rolling(window=14).mean()
        else:
            # Calculate true range if not present
            tr1 = data["high"] - data["low"]
            tr2 = abs(data["high"] - data["close"].shift(1))
            tr3 = abs(data["low"] - data["close"].shift(1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            data["atr"] = true_range.rolling(window=14).mean()
            
        data["atr_ratio"] = safe_divide(data["atr"], data["close"], 0)
        return data
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Williams %R"""
        highest_high = data["high"].rolling(period).max()
        lowest_low = data["low"].rolling(period).min()
        
        data["williams_r"] = -100 * safe_divide(
            highest_high - data["close"],
            highest_high - lowest_low,
            0.5
        )
        
        return data