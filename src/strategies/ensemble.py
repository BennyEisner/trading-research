#!/usr/bin/env python3

"""
Ensemble strategy management and signal combination
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .base import BaseStrategy, StrategyConfig


class EnsembleConfig(BaseModel):
    """Configuration for ensemble strategy management"""

    # Strategy selection and weighting
    strategy_weights: Dict[str, float] = Field(default_factory=dict)

    # Signal combination methods
    combination_method: str = Field(
        default="weighted_average", pattern="^(weighted_average|voting|confidence_weighted)$"
    )

    # Position sizing
    max_total_position: float = Field(default=1.0, ge=0.0, le=2.0)
    position_sizing_method: str = Field(
        default="signal_strength", pattern="^(signal_strength|equal_weight|volatility_adjusted)$"
    )

    # Signal filtering
    min_strategies_agreement: int = Field(default=1, ge=1)
    min_ensemble_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Risk management
    max_correlation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    rebalance_frequency: str = Field(default="daily", pattern="^(daily|weekly|monthly)$")


@dataclass
class EnsembleSignal:
    """Ensemble signal with attribution to individual strategies"""

    timestamp: pd.Timestamp
    position: float  # Combined position size
    confidence: float  # Combined confidence score
    strategy_contributions: Dict[str, float]  # Individual strategy contributions
    active_strategies: List[str]  # Strategies that contributed to signal
    risk_metrics: Dict[str, float]  # Risk-related metrics


class EnsembleManager:
    """Manages multiple trading strategies and combines their signals

    Key responsibilities:
        1. Strategy registration and management
        2. Signal combination and weighting
        3. Risk management and position sizing
        4. Performance attribution and monitoring
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None

        # Signal history for correlation analysis
        self.signal_history: Dict[str, pd.Series] = {}

    def register_strategy(self, strategy: BaseStrategy) -> None:
        """
        Register a new strategy with the ensemble

        Args:
            strategy: Strategy instance to register
        """
        if strategy.name in self.strategies:
            raise ValueError(f"Strategy '{strategy.name}' already registered")

        self.strategies[strategy.name] = strategy
        self.strategy_performance[strategy.name] = {}
        self.signal_history[strategy.name] = pd.Series(dtype=float)

        # Set default weight if not specified
        if strategy.name not in self.config.strategy_weights:
            self.config.strategy_weights[strategy.name] = 1.0 / len(self.strategies)

        print(f"Registered strategy: {strategy.name}")

    def remove_strategy(self, strategy_name: str) -> None:
        """Remove a strategy from the ensemble"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")

        del self.strategies[strategy_name]
        del self.strategy_performance[strategy_name]
        del self.signal_history[strategy_name]

        if strategy_name in self.config.strategy_weights:
            del self.config.strategy_weights[strategy_name]

    def generate_ensemble_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate combined ensemble signals from all active strategies

        Args:
            data: Market data with technical indicators

        Returns:
            DataFrame with ensemble signals and attribution
        """
        if not self.strategies:
            raise ValueError("No strategies registered")

        # Generate signals from each strategy
        strategy_signals = {}
        for name, strategy in self.strategies.items():
            if strategy.config.enabled:
                try:
                    signals = strategy.run_strategy(data)
                    strategy_signals[name] = signals

                    # Update signal history for correlation analysis
                    self._update_signal_history(name, signals)

                except Exception as e:
                    print(f"Warning: Strategy {name} failed: {e}")
                    continue

        if not strategy_signals:
            raise ValueError("No strategies generated valid signals")

        # Combine signals
        ensemble_signals = self._combine_signals(strategy_signals, data)

        # Apply ensemble-level filters and risk management
        ensemble_signals = self._apply_ensemble_filters(ensemble_signals, data)

        return ensemble_signals

    def _combine_signals(self, strategy_signals: Dict[str, pd.DataFrame], data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine individual strategy signals into ensemble signals

        Args:
            strategy_signals: Dictionary of strategy signals
            data: Market data

        Returns:
            Combined ensemble signals
        """
        # Initialize ensemble signals DataFrame
        index = data.index
        ensemble = pd.DataFrame(index=index)

        if self.config.combination_method == "weighted_average":
            ensemble = self._weighted_average_combination(strategy_signals, index)
        elif self.config.combination_method == "voting":
            ensemble = self._voting_combination(strategy_signals, index)
        elif self.config.combination_method == "confidence_weighted":
            ensemble = self._confidence_weighted_combination(strategy_signals, index)
        else:
            raise ValueError(f"Unknown combination method: {self.config.combination_method}")

        return ensemble

    def _weighted_average_combination(self, strategy_signals: Dict[str, pd.DataFrame], index: pd.Index) -> pd.DataFrame:
        """Combine signals using weighted average"""

        ensemble = pd.DataFrame(index=index)
        ensemble["position"] = 0.0
        ensemble["confidence"] = 0.0
        ensemble["active_strategies"] = [[] for _ in range(len(index))]

        # Strategy contributions tracking
        strategy_contributions = {}

        total_weight = 0
        for strategy_name, signals in strategy_signals.items():
            weight = self.config.strategy_weights.get(strategy_name, 0)
            if weight <= 0:
                continue

            # Align signals with ensemble index
            aligned_signals = signals.reindex(index, fill_value=0)

            # Weighted contribution
            weighted_position = aligned_signals["position"] * weight
            weighted_confidence = aligned_signals["signal_strength"] * weight

            ensemble["position"] += weighted_position
            ensemble["confidence"] += weighted_confidence
            total_weight += weight

            # Track contributions
            strategy_contributions[strategy_name] = weighted_position

            # Track active strategies
            active_mask = aligned_signals["position"] != 0
            for idx in index[active_mask]:
                ensemble.loc[idx, "active_strategies"].append(strategy_name)

        # Normalize by total weight
        if total_weight > 0:
            ensemble["position"] /= total_weight
            ensemble["confidence"] /= total_weight

        # Add strategy contributions as columns
        for strategy_name, contribution in strategy_contributions.items():
            ensemble[f"{strategy_name}_contribution"] = contribution

        return ensemble

    def _voting_combination(self, strategy_signals: Dict[str, pd.DataFrame], index: pd.Index) -> pd.DataFrame:
        """Combine signals using majority voting"""

        ensemble = pd.DataFrame(index=index)

        # Initialize vote counters
        long_votes = pd.Series(0, index=index)
        short_votes = pd.Series(0, index=index)
        total_confidence = pd.Series(0.0, index=index)

        for strategy_name, signals in strategy_signals.items():
            if not self.config.strategy_weights.get(strategy_name, 0):
                continue

            aligned_signals = signals.reindex(index, fill_value=0)

            # Count votes
            long_mask = aligned_signals["position"] > 0
            short_mask = aligned_signals["position"] < 0

            long_votes[long_mask] += 1
            short_votes[short_mask] += 1
            total_confidence += aligned_signals["signal_strength"]

        # Determine ensemble position based on votes
        ensemble["position"] = 0.0
        ensemble.loc[long_votes > short_votes, "position"] = 1.0
        ensemble.loc[short_votes > long_votes, "position"] = -1.0

        # Scale by vote strength
        total_votes = long_votes + short_votes
        vote_strength = np.maximum(long_votes, short_votes) / np.maximum(total_votes, 1)
        ensemble["position"] *= vote_strength

        # Average confidence
        ensemble["confidence"] = total_confidence / len(strategy_signals)

        return ensemble

    def _confidence_weighted_combination(
        self, strategy_signals: Dict[str, pd.DataFrame], index: pd.Index
    ) -> pd.DataFrame:
        """Combine signals weighted by confidence scores"""

        ensemble = pd.DataFrame(index=index)
        ensemble["position"] = 0.0
        ensemble["confidence"] = 0.0

        total_weight = 0
        for strategy_name, signals in strategy_signals.items():
            base_weight = self.config.strategy_weights.get(strategy_name, 0)
            if base_weight <= 0:
                continue

            aligned_signals = signals.reindex(index, fill_value=0)

            # Weight by both base weight and signal confidence
            dynamic_weight = base_weight * aligned_signals["signal_strength"]

            ensemble["position"] += aligned_signals["position"] * dynamic_weight
            ensemble["confidence"] += aligned_signals["signal_strength"] * dynamic_weight
            total_weight += dynamic_weight

        # Normalize
        non_zero_weight = total_weight > 0
        ensemble.loc[non_zero_weight, "position"] /= total_weight[non_zero_weight]
        ensemble.loc[non_zero_weight, "confidence"] /= total_weight[non_zero_weight]

        return ensemble

    def _apply_ensemble_filters(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Apply ensemble-level filters and risk management"""

        filtered = signals.copy()

        # Minimum confidence filter
        low_confidence = filtered["confidence"] < self.config.min_ensemble_confidence
        filtered.loc[low_confidence, "position"] = 0

        # Maximum position size constraint
        filtered["position"] = filtered["position"].clip(
            -self.config.max_total_position, self.config.max_total_position
        )

        # Minimum strategies agreement (for voting method)
        if hasattr(filtered, "active_strategies"):
            insufficient_agreement = filtered["active_strategies"].str.len() < self.config.min_strategies_agreement
            filtered.loc[insufficient_agreement, "position"] = 0

        return filtered

    def _update_signal_history(self, strategy_name: str, signals: pd.DataFrame) -> None:
        """Update signal history for correlation analysis"""
        new_signals = signals["position"]
        self.signal_history[strategy_name] = pd.concat([self.signal_history[strategy_name], new_signals]).tail(
            1000
        )  # Keep last 1000 signals

    def calculate_strategy_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix between strategies"""
        if len(self.signal_history) < 2:
            return pd.DataFrame()

        # Create correlation matrix
        signals_df = pd.DataFrame(self.signal_history)
        self.correlation_matrix = signals_df.corr()

        return self.correlation_matrix

    def get_performance_attribution(self) -> Dict[str, Any]:
        """Get detailed performance attribution for all strategies"""

        attribution = {"individual_strategies": {}, "ensemble_metrics": {}, "correlation_analysis": {}}

        # Individual strategy performance
        for name, strategy in self.strategies.items():
            attribution["individual_strategies"][name] = strategy.get_performance_summary()

        # Ensemble-level metrics
        attribution["ensemble_metrics"] = {
            "total_strategies": len(self.strategies),
            "active_strategies": sum(1 for s in self.strategies.values() if s.config.enabled),
            "average_weight": np.mean(list(self.config.strategy_weights.values())),
            "weight_distribution": dict(self.config.strategy_weights),
        }

        # Correlation analysis
        if self.correlation_matrix is not None:
            max_correlation = self.correlation_matrix.abs().unstack().max()
            avg_correlation = self.correlation_matrix.abs().mean().mean()

            attribution["correlation_analysis"] = {
                "max_correlation": max_correlation,
                "average_correlation": avg_correlation,
                "highly_correlated_pairs": self._find_highly_correlated_strategies(),
            }

        return attribution

    def _find_highly_correlated_strategies(self) -> List[Tuple[str, str, float]]:
        """Find pairs of strategies with high correlation"""
        if self.correlation_matrix is None:
            return []

        highly_correlated = []
        correlation_threshold = self.config.max_correlation_threshold

        for i, strategy1 in enumerate(self.correlation_matrix.columns):
            for j, strategy2 in enumerate(self.correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    correlation = abs(self.correlation_matrix.iloc[i, j])
                    if correlation > correlation_threshold:
                        highly_correlated.append((strategy1, strategy2, correlation))

        return sorted(highly_correlated, key=lambda x: x[2], reverse=True)

    def rebalance_weights(self, performance_lookback: int = 60) -> None:
        """
        Rebalance strategy weights based on recent performance

        Args:
            performance_lookback: Number of periods to look back for performance
        """
        if len(self.strategies) < 2:
            return

        # Calculate recent performance for each strategy
        recent_performance = {}
        for name, strategy in self.strategies.items():
            if hasattr(strategy, "performance_metrics"):
                sharpe = strategy.performance_metrics.get("sharpe_ratio", 0)
                recent_performance[name] = max(0, sharpe)  # Ensure non-negative

        if not recent_performance or all(v == 0 for v in recent_performance.values()):
            return  # No performance data available

        # Normalize to create new weights
        total_performance = sum(recent_performance.values())
        for name in recent_performance:
            self.config.strategy_weights[name] = recent_performance[name] / total_performance

        print(f"Rebalanced weights: {self.config.strategy_weights}")

    def __repr__(self) -> str:
        active_strategies = sum(1 for s in self.strategies.values() if s.config.enabled)
        return f"EnsembleManager({len(self.strategies)} strategies, {active_strategies} active)"
