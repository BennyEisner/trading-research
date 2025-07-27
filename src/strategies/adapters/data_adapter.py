#!/usr/bin/env python3

"""
Strategy Data Adapter focused on ensemble system integration with proper validation
"""

import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BaseStrategy


class FeatureValidationError(Exception):
    """Raised when feature validation fails for strategy execution"""

    pass


class StrategyDataAdapter:
    """
    Adapter for bridging feature engineering output to strategy framework input

    responsibilities:
    1. Validate required features are present for each strategy
    2. Handle missing indicators
    3. Apply feature quality checks before strategy execution
    4. Provide ensemble aware error handling
    """

    def __init__(self, min_data_points: int = 50, max_nan_ratio: float = 0.1, validation_enabled: bool = True):
        """
        Initialize strategy data adapter

        Args:
            min_data_points: minimum data points required for strategy execution
            max_nan_ratio: maximum allowed NaN ratio in features (0.0-1.0)
            validation_enabled: whether to perform feature validation
        """

        self.min_data_points = min_data_points
        self.max_nan_ratio = max_nan_ratio
        self.validation_enabled = validation_enabled
        self.validation_results = {}

    def prepare_strategy_data(self, data: pd.DataFrame, strategy: BaseStrategy) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Prepare data for individual strategy execution

        Args:
            data: DataFrame with engineered features
            strategy: Strategy instance

        Returns:
            Tuple of (prepared_data, validation_results)

        Raises:
            FeatureValidationError: If critical features are missing or invalid
        """

        strategy_name = strategy.name
        required_features = strategy.get_required_features()

        # Initialize validation results
        validation_results = {
            "strategy_name": strategy_name,
            "required_features": required_features,
            "validation_passed": True,
            "warnings": [],
            "errors": [],
        }

        if not self.validation_enabled:
            return data.copy(), validation_results

        # 1 Check basic data requirements
        if len(data) < self.min_data_points:
            error_msg = f"Insufficient data points for {strategy_name}: {len(data)} < {self.min_data_points}"
            validation_results["errors"].append(error_msg)
            validation_results["validation_passed"] = False
            raise FeatureValidationError(error_msg)

        # 2. Check for required features
        missing_features = []
        available_features = []

        for feature in required_features:
            if feature not in data.columns:
                missing_features.append(feature)
            else:
                available_features.append(feature)

        # Handle missing features
        if missing_features:
            # Critical features (without fallbacks)
            critical_missing = [f for f in missing_features if not self._has_fallback(f, data.columns)]

            if critical_missing:
                error_msg = f"Critical features missing for {strategy_name}: {critical_missing}"
                validation_results["errors"].append(error_msg)
                validation_results["validation_passed"] = False
                raise FeatureValidationError(error_msg)
            else:

                validation_results["warnings"].append(f"Missing features with fallbacks: {missing_features}")

        # 3. Feature quality validation
        prepared_data = data.copy()
        for feature in available_features:
            feature_validation = self._validate_feature_quality(prepared_data, feature, strategy_name)

            if feature_validation["errors"]:
                validation_results["errors"].extend(feature_validation["errors"])
                validation_results["validation_passed"] = False

            if feature_validation["warnings"]:
                validation_results["warnings"].extend(feature_validation["warnings"])

            # Apply feature corrections if needed
            if feature_validation["corrected_data"] is not None:
                prepared_data[feature] = feature_validation["corrected_data"]

        # 4. Strategy-specific validation
        try:
            strategy_validation = strategy.validate_parameters()
            if not strategy_validation:
                error_msg = f"Strategy parameter validation failed for {strategy_name}"
                validation_results["errors"].append(error_msg)
                validation_results["validation_passed"] = False
        except Exception as e:
            error_msg = f"Strategy validation error for {strategy_name}: {str(e)}"
            validation_results["errors"].append(error_msg)
            validation_results["validation_passed"] = False

        # Store validation results
        self.validation_results[strategy_name] = validation_results

        # Raise error if validation failed
        if not validation_results["validation_passed"]:
            raise FeatureValidationError(
                f"Feature validation failed for {strategy_name}: {validation_results['errors']}"
            )

        return prepared_data, validation_results

    def prepare_ensemble_data(
        self, data: pd.DataFrame, strategies: List[BaseStrategy]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Prepare data for ensemble execution with graceful strategy degradation"""

        ensemble_results = {
            "total_strategies": len(strategies),
            "valid_strategies": [],
            "invalid_strategies": [],
            "validation_summary": {},
            "ensemble_ready": False,
        }

        valid_strategies = []
        strategy_validations = {}

        # Validate each strategy individually
        for strategy in strategies:
            try:
                prepared_data, validation_results = self.prepare_strategy_data(data, strategy)
                valid_strategies.append(strategy)
                ensemble_results["valid_strategies"].append(strategy.name)
                strategy_validations[strategy.name] = validation_results

            except FeatureValidationError as e:
                ensemble_results["invalid_strategies"].append({"strategy_name": strategy.name, "error": str(e)})
                strategy_validations[strategy.name] = {"validation_passed": False, "error": str(e)}

        # Determine if ensemble strategy can proceed (okay to run with just 1 strategy)
        min_strategies_for_ensemble = 1
        ensemble_results["ensemble_ready"] = len(valid_strategies) >= min_strategies_for_ensemble
        ensemble_results["validation_summary"] = strategy_validations

        if not ensemble_results["ensemble_ready"]:
            raise FeatureValidationError(
                f"Insufficient valid strategies for ensemble: {len(valid_strategies)} < {min_strategies_for_ensemble}"
            )

        # Log ensemble readiness
        if len(ensemble_results["invalid_strategies"]) > 0:
            warnings.warn(
                f"Ensemble proceeding with {len(valid_strategies)}/{len(strategies)} strategies. "
                f"Invalid: {[s['strategy_name'] for s in ensemble_results['invalid_strategies']]}"
            )

        return data.copy(), ensemble_results

    def _has_fallback(self, feature: str, available_columns: List[str]) -> bool:
        """Check if a missing feature has a fallback available"""

        fallback_mappings = {
            "rsi_14": ["rsi"],  # fallback to general 14 day rsi
            "rsi": ["rsi_14"],  # general RSI can use specific period
            "close": [],
            "atr": [],
            "macd": [],
            "macd_signal": [],
            "macd_histogram": [],
        }

        fallbacks = fallback_mappings.get(feature, [])
        return any(fallback in available_columns for fallback in fallbacks)

    def _validate_feature_quality(self, data: pd.DataFrame, feature: str, strategy_name: str) -> Dict[str, Any]:
        """Validate individual feature quality"""

        validation_result = {"feature": feature, "errors": [], "warnings": [], "corrected_data": None}

        if feature not in data.columns:
            validation_result["errors"].append(f"Feature '{feature}' not found in data")
            return validation_result

        feature_data = data[feature]

        # 1. Check for excessive NaN values
        nan_ratio = feature_data.isna().sum() / len(feature_data)
        if nan_ratio > self.max_nan_ratio:
            validation_result["errors"].append(
                f"Feature '{feature}' has {nan_ratio:.2%} NaN values (> {self.max_nan_ratio:.2%})"
            )
        elif nan_ratio > 0:
            validation_result["warnings"].append(f"Feature '{feature}' has {nan_ratio:.2%} NaN values")

        # 2. Check for infinite values
        if feature_data.dtype in ["float64", "int64"]:
            inf_count = np.isinf(feature_data).sum()
            if inf_count > 0:
                validation_result["errors"].append(f"Feature '{feature}' has {inf_count} infinite values")

        # 3. Check for constant values (zero variance)
        if feature_data.dtype in ["float64", "int64"]:
            if len(feature_data.dropna()) > 1:
                variance = feature_data.var()
                if pd.isna(variance) or variance == 0:
                    validation_result["warnings"].append(f"Feature '{feature}' has zero variance")

        # 4. Strategyspecific feature validation
        if feature.startswith("rsi"):
            # RSI should be between 0 and 100
            valid_rsi = feature_data.dropna()
            if len(valid_rsi) > 0:
                if valid_rsi.min() < 0 or valid_rsi.max() > 100:
                    validation_result["warnings"].append(f"RSI feature '{feature}' has values outside 0-100 range")

        elif feature == "atr":
            valid_atr = feature_data.dropna()
            if len(valid_atr) > 0:
                if valid_atr.min() <= 0:
                    validation_result["warnings"].append(f"ATR feature has non-positive values")

        return validation_result

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary for processed strategies"""

        summary = {
            "total_strategies_validated": len(self.validation_results),
            "successful_validations": 0,
            "failed_validations": 0,
            "strategies_with_warnings": 0,
            "common_issues": {},
            "strategy_details": self.validation_results,
        }

        # Aggregate statistics
        all_errors = []
        all_warnings = []

        for strategy_name, results in self.validation_results.items():
            if results["validation_passed"]:
                summary["successful_validations"] += 1
            else:
                summary["failed_validations"] += 1

            if results["warnings"]:
                summary["strategies_with_warnings"] += 1

            all_errors.extend(results.get("errors", []))
            all_warnings.extend(results.get("warnings", []))

        summary["common_errors"] = dict(Counter(all_errors).most_common(5))
        summary["common_warnings"] = dict(Counter(all_warnings).most_common(5))

        return summary

    def reset_validation_results(self):
        """Reset stored validation results"""
        self.validation_results = {}

