#!/usr/bin/env python3

"""
Ensemble strategy validation and backtesting requirements
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...validation.gapped_time_series_cv import GappedTimeSeriesCV
from ...validation.pipeline_validator import PipelineValidator
from ..adapters.data_adapter import StrategyDataAdapter
from ..core.compatibility_checker import FeatureValidationError
from ..base import BaseStrategy


class StrategyValidator:
    """Specialized for ensemble strategy validation and performance assessment"""

    def __init__(
        self,
        data_adapter: Optional[StrategyDataAdapter] = None,
        pipeline_validator: Optional[PipelineValidator] = None,
        min_backtest_days: int = 252,
        max_drawdown_threshold: float = 0.15,
        min_sharpe_ratio: float = 0.5,
    ):
        """
        Initialize strategy validator

        Args:
            data_adapter: Data adapter for feature validation
            pipeline_validator: Existing pipeline validator for data quality
            min_backtest_days: Minimum days required for strategy backtesting
            max_drawdown_threshold: Maximum acceptable drawdown
            min_sharpe_ratio: Minimum acceptable Sharpe ratio
        """
        self.data_adapter = data_adapter or StrategyDataAdapter()
        self.pipeline_validator = pipeline_validator or PipelineValidator()
        self.min_backtest_days = min_backtest_days
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_sharpe_ratio = min_sharpe_ratio

        self.validation_results = {}

    def validate_strategy_data_compatibility(
        self, data: pd.DataFrame, strategy: BaseStrategy
    ) -> Tuple[bool, List[str]]:
        """
        Validate strategy compatibility with provided data
        Leverages existing data adapter validation
        """
        issues = []

        try:
            # Use existing data adapter validation
            prepared_data, validation_results = self.data_adapter.prepare_strategy_data(data, strategy)

            # Store detailed validation results
            self.validation_results[f"{strategy.name}_data_compatibility"] = {
                "validation_passed": validation_results["validation_passed"],
                "required_features": validation_results["required_features"],
                "warnings": validation_results["warnings"],
                "errors": validation_results["errors"],
                "data_points": len(prepared_data),
                "feature_coverage": (
                    len([f for f in validation_results["required_features"] if f in data.columns])
                    / len(validation_results["required_features"])
                    if validation_results["required_features"]
                    else 1.0
                ),
            }

            # Additional strategy-specific checks
            if len(prepared_data) < self.min_backtest_days:
                issues.append(
                    f"Insufficient data for reliable backtesting: {len(prepared_data)} < {self.min_backtest_days} days"
                )

            # Check data freshness for live trading (skip in test environment)
            if "date" in data.columns:
                last_date = pd.to_datetime(data["date"]).max()
                days_since_last = (datetime.now() - last_date).days
                # Only warn about stale data if it's more than 30 days old (more lenient for testing)
                if days_since_last > 30:
                    issues.append(f"Data may be stale: last update {days_since_last} days ago")

            return len(issues) == 0, issues

        except FeatureValidationError as e:
            issues.append(f"Strategy data validation failed: {str(e)}")
            return False, issues

    def validate_strategy_signals(self, data: pd.DataFrame, strategy: BaseStrategy) -> Tuple[bool, List[str]]:
        """
        Validate strategy signal generation quality
        """
        issues = []

        try:
            # Prepare data using adapter
            prepared_data, _ = self.data_adapter.prepare_strategy_data(data, strategy)

            # Generate signals
            signals = strategy.generate_signals(prepared_data)

            # Signal structure validation
            required_signal_columns = ["position", "entry_price"]
            missing_signal_cols = [col for col in required_signal_columns if col not in signals.columns]
            if missing_signal_cols:
                issues.append(f"Missing required signal columns: {missing_signal_cols}")
                return False, issues

            # Signal quality checks
            position_values = signals["position"].dropna()
            valid_positions = position_values.isin([-1.0, 0.0, 1.0])
            if not valid_positions.all():
                invalid_count = (~valid_positions).sum()
                issues.append(f"Invalid position values detected: {invalid_count} signals")

            # Check signal frequency
            non_zero_signals = (position_values != 0).sum()
            signal_frequency = non_zero_signals / len(position_values) if len(position_values) > 0 else 0

            if signal_frequency == 0:
                issues.append("Strategy generated no trading signals")
            elif signal_frequency > 0.5:
                issues.append(f"Very high signal frequency ({signal_frequency:.2%}) - may indicate overtrading")
            elif signal_frequency < 0.01:
                issues.append(f"Very low signal frequency ({signal_frequency:.2%}) - may be too conservative")

            # Check for NaN in critical signal fields
            nan_positions = signals["position"].isna().sum()
            nan_prices = signals["entry_price"].isna().sum()

            if nan_positions > len(signals) * 0.1:  # More than 10% NaN positions
                issues.append(f"High NaN count in positions: {nan_positions} out of {len(signals)}")

            # Store signal validation results
            self.validation_results[f"{strategy.name}_signal_quality"] = {
                "total_signals": len(signals),
                "non_zero_signals": non_zero_signals,
                "signal_frequency": signal_frequency,
                "nan_positions": nan_positions,
                "nan_entry_prices": nan_prices,
                "valid_position_values": valid_positions.all(),
                "issues": issues,
            }

            return len(issues) == 0, issues

        except Exception as e:
            issues.append(f"Signal generation failed: {str(e)}")
            return False, issues

    def validate_ensemble_strategies(
        self, data: pd.DataFrame, strategies: List[BaseStrategy]
    ) -> Tuple[bool, List[str]]:
        """
        Validate ensemble of strategies for compatibility and diversity
        """
        issues = []

        if len(strategies) < 2:
            issues.append("Ensemble requires at least 2 strategies for diversification")
            return False, issues

        try:
            # Use data adapter for ensemble validation
            prepared_data, ensemble_results = self.data_adapter.prepare_ensemble_data(data, strategies)

            valid_strategy_count = len(ensemble_results["valid_strategies"])
            total_strategy_count = ensemble_results["total_strategies"]

            # Check ensemble viability
            if not ensemble_results["ensemble_ready"]:
                issues.append("Ensemble not ready - insufficient valid strategies")
                return False, issues

            # Only warn about low success rate, don't fail validation
            if valid_strategy_count < total_strategy_count * 0.7:  # Less than 70% success rate
                issues.append(
                    f"Warning: Low strategy success rate: {valid_strategy_count}/{total_strategy_count} strategies valid"
                )

            # Generate signals for diversity analysis
            strategy_signals = {}
            for strategy in strategies:
                if strategy.name in ensemble_results["valid_strategies"]:
                    try:
                        signals = strategy.generate_signals(prepared_data)
                        strategy_signals[strategy.name] = signals["position"].fillna(0)
                    except Exception as e:
                        issues.append(f"Warning: Failed to generate signals for {strategy.name}: {str(e)}")

            # Analyze signal diversity
            if len(strategy_signals) >= 2:
                signal_correlations = {}
                strategy_names = list(strategy_signals.keys())

                for i in range(len(strategy_names)):
                    for j in range(i + 1, len(strategy_names)):
                        strategy_a = strategy_names[i]
                        strategy_b = strategy_names[j]

                        signals_a = strategy_signals[strategy_a]
                        signals_b = strategy_signals[strategy_b]

                        # Calculate signal correlation
                        if len(signals_a) > 1 and len(signals_b) > 1:
                            corr = np.corrcoef(signals_a, signals_b)[0, 1]
                            if not np.isnan(corr):
                                signal_correlations[f"{strategy_a}_vs_{strategy_b}"] = corr

                # Check for overly correlated strategies
                high_correlations = {pair: corr for pair, corr in signal_correlations.items() if abs(corr) > 0.8}

                if high_correlations:
                    issues.append(f"Warning: Highly correlated strategies detected: {list(high_correlations.keys())}")

            # Store ensemble validation results
            self.validation_results["ensemble_validation"] = {
                "total_strategies": total_strategy_count,
                "valid_strategies": valid_strategy_count,
                "success_rate": valid_strategy_count / total_strategy_count,
                "signal_correlations": signal_correlations if "signal_correlations" in locals() else {},
                "ensemble_ready": ensemble_results["ensemble_ready"],
                "issues": issues,
            }

            return len(issues) == 0, issues

        except Exception as e:
            issues.append(f"Ensemble validation failed: {str(e)}")
            return False, issues

    def validate_backtest_readiness(self, data: pd.DataFrame, strategies: List[BaseStrategy]) -> Tuple[bool, List[str]]:
        """
        Validate data and strategies are ready for backtesting
        Integrates with existing pipeline validator
        """
        issues = []

        # Basic data validation using existing pipeline validator
        if "date" in data.columns and "close" in data.columns:
            ticker = "BACKTEST"  # Generic ticker for validation
            data_valid, data_issues = self.pipeline_validator.validate_raw_data(data, ticker)
            if not data_valid:
                issues.extend([f"Data quality issue: {issue}" for issue in data_issues])

        # Feature validation
        all_required_features = set()
        for strategy in strategies:
            all_required_features.update(strategy.get_required_features())

        feature_valid, feature_issues = self.pipeline_validator.validate_feature_data(data, list(all_required_features))
        if not feature_valid:
            issues.extend([f"Feature issue: {issue}" for issue in feature_issues])

        # Time series continuity check for backtesting
        if "date" in data.columns:
            date_col = pd.to_datetime(data["date"])
            data_sorted = data.copy().sort_values("date")

            # Check for sufficient data length
            if len(data_sorted) < self.min_backtest_days:
                issues.append(f"Insufficient data for backtesting: {len(data_sorted)} < {self.min_backtest_days} days")

            # Check date range coverage
            date_range = (date_col.max() - date_col.min()).days
            if date_range < self.min_backtest_days:
                issues.append(f"Date range too short for backtesting: {date_range} < {self.min_backtest_days} days")

        # Strategy-specific backtest readiness
        for strategy in strategies:
            strategy_valid, strategy_issues = self.validate_strategy_data_compatibility(data, strategy)
            if not strategy_valid:
                issues.extend([f"{strategy.name}: {issue}" for issue in strategy_issues])

        # Store backtest readiness results
        self.validation_results["backtest_readiness"] = {
            "data_length": len(data),
            "date_range_days": date_range if "date_range" in locals() else 0,
            "required_features_available": len(all_required_features.intersection(set(data.columns))),
            "total_required_features": len(all_required_features),
            "strategies_ready": len(strategies),
            "issues": issues,
        }

        return len(issues) == 0, issues

    def create_time_series_splits(
        self, data: pd.DataFrame, n_splits: int = 5, test_size: float = 0.2, gap_size: int = 10
    ) -> GappedTimeSeriesCV:
        """
        Create time series cross-validation splits using existing framework
        """
        cv = GappedTimeSeriesCV(n_splits=n_splits, test_size=test_size, gap_size=gap_size, expanding_window=True)

        # Validate splits don't have lookahead bias
        if "date" in data.columns:
            dates = pd.to_datetime(data["date"])
            if not cv.validate_no_leakage(dates):
                raise ValueError("Time series splits contain lookahead bias")

        return cv

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive validation summary
        Integrates with existing pipeline validator summary
        """
        strategy_summary = {
            "timestamp": datetime.now().isoformat(),
            "strategy_validation_results": self.validation_results,
            "overall_strategy_status": "PASS",
        }

        # Check for any strategy validation failures (ignore warnings)
        for validation_data in self.validation_results.values():
            if isinstance(validation_data, dict):
                issues = validation_data.get("issues", [])
                # Only count non-warning issues as failures
                error_issues = [issue for issue in issues if not issue.startswith("Warning:")]
                if error_issues:
                    strategy_summary["overall_strategy_status"] = "FAIL"
                    break

        # Get pipeline validator summary if available
        pipeline_summary = self.pipeline_validator.get_validation_summary()

        # Combine summaries
        combined_summary = {
            "timestamp": strategy_summary["timestamp"],
            "strategy_validation": strategy_summary,
            "pipeline_validation": pipeline_summary,
            "overall_status": (
                "PASS"
                if (
                    strategy_summary["overall_strategy_status"] == "PASS"
                    and pipeline_summary["overall_status"] == "PASS"
                )
                else "FAIL"
            ),
        }

        return combined_summary

    def reset_validation_results(self):
        """Reset validation results for new validation run"""
        self.validation_results = {}
        self.pipeline_validator.validation_results = {}


def create_strategy_validator(
    data_adapter: Optional[StrategyDataAdapter] = None, pipeline_validator: Optional[PipelineValidator] = None, **kwargs
) -> StrategyValidator:
    """Convenience function to create strategy validator"""
    return StrategyValidator(data_adapter=data_adapter, pipeline_validator=pipeline_validator, **kwargs)

