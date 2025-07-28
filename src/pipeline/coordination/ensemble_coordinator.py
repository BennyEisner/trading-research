#!/usr/bin/env python3

"""
Ensemble Coordinator for multi-strategy logic and graceful degradation
"""

import warnings
from typing import Dict, List, Tuple, Any
import pandas as pd
from ...strategies.base import BaseStrategy
from ...strategies.core.compatibility_checker import StrategyCompatibilityChecker, FeatureValidationError


class EnsembleCoordinator:
    """
    Coordinates ensemble strategy execution with graceful degradation
    
    Single responsibility: Manage multi-strategy ensemble logic
    """
    
    def __init__(self, compatibility_checker: StrategyCompatibilityChecker = None, min_strategies: int = 1):
        """
        Initialize ensemble coordinator
        
        Args:
            compatibility_checker: Strategy compatibility checker
            min_strategies: Minimum strategies required for ensemble
        """
        self.compatibility_checker = compatibility_checker or StrategyCompatibilityChecker()
        self.min_strategies = min_strategies
        
    def coordinate_ensemble_execution(self, data: pd.DataFrame, strategies: List[BaseStrategy]) -> Tuple[List[BaseStrategy], Dict[str, Any]]:
        """
        Coordinate ensemble execution with graceful degradation
        
        Args:
            data: Input dataframe
            strategies: List of strategies to coordinate
            
        Returns:
            Tuple of (valid_strategies, coordination_results)
        """
        coordination_results = {
            "total_strategies": len(strategies),
            "valid_strategies": [],
            "invalid_strategies": [],
            "validation_summary": {},
            "ensemble_ready": False,
        }
        
        valid_strategies = []
        strategy_validations = {}
        
        # Validate each strategy
        for strategy in strategies:
            is_compatible, issues = self.compatibility_checker.check_strategy_compatibility(data, strategy)
            
            if is_compatible:
                valid_strategies.append(strategy)
                coordination_results["valid_strategies"].append(strategy.name)
                strategy_validations[strategy.name] = {
                    "validation_passed": True,
                    "issues": issues
                }
            else:
                coordination_results["invalid_strategies"].append({
                    "strategy_name": strategy.name,
                    "error": "; ".join(issues)
                })
                strategy_validations[strategy.name] = {
                    "validation_passed": False,
                    "error": "; ".join(issues)
                }
                
        # Determine ensemble readiness
        coordination_results["ensemble_ready"] = len(valid_strategies) >= self.min_strategies
        coordination_results["validation_summary"] = strategy_validations
        
        if not coordination_results["ensemble_ready"]:
            raise FeatureValidationError(
                f"Insufficient valid strategies for ensemble: {len(valid_strategies)} < {self.min_strategies}"
            )
            
        # Warn about degraded strategies
        if len(coordination_results["invalid_strategies"]) > 0:
            warnings.warn(
                f"Ensemble proceeding with {len(valid_strategies)}/{len(strategies)} strategies. "
                f"Invalid: {[s['strategy_name'] for s in coordination_results['invalid_strategies']]}"
            )
            
        return valid_strategies, coordination_results
        
    def get_ensemble_status(self, strategies: List[BaseStrategy]) -> Dict[str, Any]:
        """Get current ensemble coordination status"""
        return {
            "registered_strategies": [s.name for s in strategies],
            "total_strategies": len(strategies),
            "min_strategies_required": self.min_strategies,
            "compatibility_checker_config": {
                "min_data_points": self.compatibility_checker.min_data_points,
                "max_nan_ratio": self.compatibility_checker.max_nan_ratio
            }
        }