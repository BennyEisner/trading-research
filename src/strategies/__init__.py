#!/usr/bin/env python3

"""
Trading Strategies Module
Ensemble-based technical indicator strategies for financial time series
"""

from .base import BaseStrategy, StrategyConfig
from .ensemble import EnsembleManager, EnsembleConfig
from .indicators import (
    RSIMeanReversionStrategy,
    MACDMomentumStrategy, 
    BollingerBreakoutStrategy,
    VolumePriceTrendStrategy,
    VolatilityBreakoutStrategy
)

__all__ = [
    'BaseStrategy',
    'StrategyConfig',
    'EnsembleManager',
    'EnsembleConfig',
    'RSIMeanReversionStrategy',
    'MACDMomentumStrategy',
    'BollingerBreakoutStrategy',
    'VolumePriceTrendStrategy',
    'VolatilityBreakoutStrategy'
]