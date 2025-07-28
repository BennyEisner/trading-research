#!/usr/bin/env python3

"""
Simplified Strategy Data Adapter focused on data formatting only
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..base import BaseStrategy


class StrategyDataAdapter:
    """
    Simplified adapter focused on data formatting for strategy input
    
    Single responsibility: Format and clean data for strategy consumption
    """
    
    def __init__(self):
        """Initialize simplified data adapter"""
        self.formatting_results = {}
        
    def format_strategy_data(self, data: pd.DataFrame, strategy: BaseStrategy) -> pd.DataFrame:
        """Format data for strategy consumption
        
        Args:
            data: DataFrame with OHLCV and features
            strategy: Strategy instance to format data for
            
        Returns:
            Formatted DataFrame ready for strategy
        """
        strategy_name = strategy.name
        formatted_data = data.copy()
        
        # Apply basic data cleaning and formatting
        formatted_data = self._clean_data(formatted_data)
        formatted_data = self._apply_feature_fallbacks(formatted_data, strategy)
        
        # Store formatting results
        self.formatting_results[strategy_name] = {
            "original_columns": len(data.columns),
            "formatted_columns": len(formatted_data.columns),
            "data_points": len(formatted_data)
        }
        
        return formatted_data
        
    def format_ensemble_data(self, data: pd.DataFrame, strategies: List[BaseStrategy]) -> pd.DataFrame:
        """Format data for ensemble execution
        
        Args:
            data: DataFrame with OHLCV and features
            strategies: List of strategies to format data for
            
        Returns:
            Formatted DataFrame ready for ensemble
        """
        formatted_data = data.copy()
        
        # Apply ensemble-wide formatting
        formatted_data = self._clean_data(formatted_data)
        
        # Apply fallbacks for all strategies
        for strategy in strategies:
            formatted_data = self._apply_feature_fallbacks(formatted_data, strategy)
            
        return formatted_data
        
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic data cleaning"""
        cleaned_data = data.copy()
        
        # Handle infinite values
        for col in cleaned_data.select_dtypes(include=[np.number]).columns:
            cleaned_data[col] = cleaned_data[col].replace([np.inf, -np.inf], np.nan)
            
        return cleaned_data
        
    def _apply_feature_fallbacks(self, data: pd.DataFrame, strategy: BaseStrategy) -> pd.DataFrame:
        """Apply feature fallbacks for missing indicators"""
        result_data = data.copy()
        required_features = strategy.get_required_features()
        
        fallback_mappings = {
            "rsi_14": ["rsi"],
            "rsi": ["rsi_14"],
            "close": [],
            "atr": [],
            "macd": [],
            "macd_signal": [],
            "macd_histogram": []
        }
        
        for feature in required_features:
            if feature not in result_data.columns:
                fallbacks = fallback_mappings.get(feature, [])
                for fallback in fallbacks:
                    if fallback in result_data.columns:
                        result_data[feature] = result_data[fallback]
                        break
                        
        return result_data
        
    def get_formatting_summary(self) -> Dict[str, Any]:
        """Get summary of formatting operations"""
        return {
            "strategies_formatted": len(self.formatting_results),
            "formatting_details": self.formatting_results
        }
        
    def reset_validation_results(self) -> None:
        """Reset formatting results (for compatibility)"""
        self.formatting_results = {}

    # Legacy methods for backward compatibility
    def prepare_strategy_data(self, data: pd.DataFrame, strategy: BaseStrategy) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Legacy method - delegates to new format_strategy_data"""
        formatted_data = self.format_strategy_data(data, strategy)
        validation_results = {
            "strategy_name": strategy.name,
            "required_features": strategy.get_required_features(),
            "validation_passed": True,
            "warnings": [],
            "errors": [],
            "data_points": len(formatted_data)
        }
        return formatted_data, validation_results
        
    def prepare_ensemble_data(self, data: pd.DataFrame, strategies: List[BaseStrategy]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Legacy method - delegates to coordinator"""
        formatted_data = self.format_ensemble_data(data, strategies)
        ensemble_results = {
            "total_strategies": len(strategies),
            "valid_strategies": [s.name for s in strategies],
            "invalid_strategies": [],
            "validation_summary": {},
            "ensemble_ready": True,
        }
        return formatted_data, ensemble_results