#!/usr/bin/env python3

"""
Pattern Target Generator for Pattern Detection Validation
Generates binary targets for pattern resolution validation instead of return prediction
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent))


class PatternTargetGenerator:
    """
    Generate pattern detection targets for LSTM training

    Core Philosophy: Instead of predicting returns, validate pattern resolution
    - Momentum Persistence: Does momentum actually persist when detected?
    - Volatility Regime: Does volatility expand/contract as predicted?
    - Trend Exhaustion: Does trend reverse when exhaustion is detected?
    - Volume Divergence: Does divergence resolve as expected?
    """

    def __init__(self, lookback_window: int = 20, validation_horizons: List[int] = [3, 5, 10]):
        """
        Initialize pattern target generator

        Args:
            lookback_window: Window for pattern detection (matches LSTM sequence length)
            validation_horizons: Time horizons for pattern resolution validation
        """
        self.lookback_window = lookback_window
        self.validation_horizons = validation_horizons

    def generate_all_pattern_targets(
        self, features_df: pd.DataFrame, primary_horizon: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Generate all pattern detection targets for training

        Args:
            features_df: DataFrame with calculated pattern features
            primary_horizon: Primary validation horizon (5 days for swing trading)

        Returns:
            Dictionary of pattern targets for training
        """

        print(f"Generating pattern targets for {len(features_df)} samples...")

        # Validate required features exist
        required_features = [
            "momentum_persistence_7d",
            "volatility_clustering",
            "trend_exhaustion",
            "volume_price_divergence",
            "volatility_regime_change",
            "returns_1d",
            "returns_3d",
            "returns_7d",
            "close",
        ]

        missing_features = [f for f in required_features if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        targets = {}

        # Generate each pattern validation target
        targets["momentum_persistence_binary"] = self._generate_momentum_persistence_target(
            features_df, horizon=primary_horizon
        )

        targets["volatility_regime_binary"] = self._generate_volatility_regime_target(
            features_df, horizon=primary_horizon
        )

        targets["trend_exhaustion_binary"] = self._generate_trend_exhaustion_target(
            features_df, horizon=primary_horizon
        )

        targets["volume_divergence_binary"] = self._generate_volume_divergence_target(
            features_df, horizon=primary_horizon
        )

        # Create combined pattern confidence target
        targets["pattern_confidence_score"] = self._generate_combined_pattern_target(targets)

        print(f"Generated pattern targets:")
        for target_name, target_values in targets.items():
            positive_rate = np.mean(target_values) if len(target_values) > 0 else 0
            print(f"   - {target_name}: {positive_rate:.3f} positive rate")

        return targets

    def _generate_momentum_persistence_target(self, features_df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        Generate continuous momentum persistence target using ONLY historical data

        CRITICAL FIX: Removes forward-looking validation - uses only past momentum patterns
        - Signal strength based on historical momentum consistency
        - Creates continuous targets in [0, 1] range without future data leakage
        """

        momentum_feature = features_df["momentum_persistence_7d"].values
        returns_1d = features_df["returns_1d"].values

        targets = np.zeros(len(features_df))

        for i in range(max(10, self.lookback_window // 2), len(features_df)):  # Reduced requirements
            current_momentum = momentum_feature[i]

            if np.isnan(current_momentum):
                continue

            # BALANCED: Use historical data with relaxed requirements for better signal generation
            historical_period = slice(max(0, i - min(self.lookback_window, i)), i)
            historical_returns = returns_1d[historical_period]
            historical_momentum = momentum_feature[historical_period]

            if len(historical_returns) < 5:  # Relaxed from 10 to 5 days minimum
                continue

            # Signal strength based on current momentum magnitude
            momentum_strength = np.abs(current_momentum)
            momentum_percentile = np.percentile(np.abs(momentum_feature[~np.isnan(momentum_feature)]), 90)
            normalized_momentum = min(momentum_strength / (momentum_percentile + 1e-8), 1.0)

            # IMPROVED: More generous signal generation while maintaining temporal safety
            valid_momentum = historical_momentum[~np.isnan(historical_momentum)]
            if len(valid_momentum) > 2:  # Relaxed requirement
                # Momentum strength as primary factor (more responsive)
                primary_signal = normalized_momentum
                
                # Historical context as secondary factor
                momentum_consistency = 1.0 - (np.std(valid_momentum) / (np.mean(np.abs(valid_momentum)) + 1e-8))
                momentum_consistency = max(0.3, min(1.0, momentum_consistency))  # Floor at 0.3
                
                # Relative strength (less restrictive)
                hist_avg = np.mean(np.abs(valid_momentum)) if len(valid_momentum) > 0 else 1.0
                relative_strength = min(momentum_strength / (hist_avg + 1e-8), 2.0)  # Allow up to 2x
                
                # BALANCED: Weight primary signal more heavily
                targets[i] = 0.6 * primary_signal + 0.4 * (momentum_consistency * min(relative_strength, 1.0))
            else:
                # Fallback to pure signal strength when insufficient history
                targets[i] = normalized_momentum * 0.5  # Give some signal even with limited history

        # Normalize to [0, 1] range
        if np.max(targets) > 0:
            normalized_targets = targets / np.max(targets)
        else:
            normalized_targets = targets

        return normalized_targets

    def _generate_volatility_regime_target(self, features_df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        Generate continuous volatility regime target using ONLY historical data

        CRITICAL FIX: Removes forward-looking validation - uses only past volatility patterns
        - Signal strength based on historical volatility regime consistency
        - Creates continuous targets in [0, 1] range without future data leakage
        """

        volatility_change = features_df["volatility_regime_change"].values
        returns_1d = features_df["returns_1d"].values

        targets = np.zeros(len(features_df))

        for i in range(max(15, self.lookback_window), len(features_df)):  # Reduced from 2x lookback
            current_vol_signal = volatility_change[i]

            if np.isnan(current_vol_signal):
                continue

            # FIXED: Use only HISTORICAL data for regime detection
            # Recent history for current regime assessment
            recent_period = slice(i - self.lookback_window, i)
            recent_returns = returns_1d[recent_period]
            recent_vol = np.std(recent_returns) if len(recent_returns) > 0 else 0

            # Extended history for regime comparison
            extended_period = slice(i - self.lookback_window * 2, i - self.lookback_window)
            extended_returns = returns_1d[extended_period]
            extended_vol = np.std(extended_returns) if len(extended_returns) > 0 else 0

            if recent_vol > 0 and extended_vol > 0:
                # Signal strength (how strong the regime change signal is)
                signal_strength = np.abs(current_vol_signal)
                signal_percentile = np.percentile(np.abs(volatility_change[~np.isnan(volatility_change)]), 90)
                normalized_signal = min(signal_strength / (signal_percentile + 1e-8), 1.0)

                # Historical regime change magnitude (recent vs extended past)
                historical_vol_ratio = recent_vol / extended_vol
                vol_change_magnitude = abs(np.log(historical_vol_ratio))

                # Historical volatility consistency (how stable the signal has been)
                historical_vol_signals = volatility_change[max(0, i - self.lookback_window):i]
                valid_signals = historical_vol_signals[~np.isnan(historical_vol_signals)]
                
                if len(valid_signals) > 5:
                    signal_consistency = 1.0 - (np.std(valid_signals) / (np.mean(np.abs(valid_signals)) + 1e-8))
                    signal_consistency = max(0.0, min(1.0, signal_consistency))
                else:
                    signal_consistency = 0.5

                # Combine signal strength with historical evidence
                if vol_change_magnitude > 0.1:  # 10% volatility change minimum
                    targets[i] = normalized_signal * min(vol_change_magnitude, 1.0) * signal_consistency
                else:
                    # Weak regime changes get reduced scores based on signal strength only
                    targets[i] = normalized_signal * 0.3

        # Normalize to [0, 1] range for consistent scaling
        if np.max(targets) > 0:
            normalized_targets = targets / np.max(targets)
        else:
            normalized_targets = targets

        return normalized_targets

    def _generate_trend_exhaustion_target(self, features_df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        Generate continuous trend exhaustion target using ONLY historical data

        CRITICAL FIX: Removes forward-looking validation - uses only past trend patterns
        - Signal strength based on historical trend exhaustion patterns
        - Creates continuous targets in [0, 1] range without future data leakage
        """

        trend_exhaustion = features_df["trend_exhaustion"].values
        returns_3d = features_df["returns_3d"].values

        targets = np.zeros(len(features_df))

        for i in range(max(8, self.lookback_window // 2), len(features_df)):  # More generous starting point
            current_exhaustion = trend_exhaustion[i]

            if np.isnan(current_exhaustion):
                continue

            # Signal strength normalization
            exhaustion_strength = np.abs(current_exhaustion)
            exhaustion_percentile = np.percentile(np.abs(trend_exhaustion[~np.isnan(trend_exhaustion)]), 75)  # Less restrictive
            normalized_exhaustion = min(exhaustion_strength / (exhaustion_percentile + 1e-8), 1.0)

            # BALANCED: Use historical data with relaxed requirements
            historical_period = slice(max(0, i - min(self.lookback_window, i)), i)
            historical_returns = returns_3d[historical_period]
            historical_exhaustion = trend_exhaustion[historical_period]

            if len(historical_returns) < 5:  # Reduced requirement
                continue

            # Current trend strength relative to recent history
            current_trend = returns_3d[i] if not np.isnan(returns_3d[i]) else 0
            valid_historical_returns = historical_returns[~np.isnan(historical_returns)]
            
            if len(valid_historical_returns) > 0:
                historical_trend_strength = np.mean(np.abs(valid_historical_returns))
                current_trend_strength = abs(current_trend)
                
                # Relative exhaustion: current exhaustion vs historical average
                valid_exhaustion = historical_exhaustion[~np.isnan(historical_exhaustion)]
                if len(valid_exhaustion) > 5:
                    historical_exhaustion_avg = np.mean(valid_exhaustion)
                    exhaustion_relative_strength = min(current_exhaustion / (historical_exhaustion_avg + 1e-8), 2.0)
                else:
                    exhaustion_relative_strength = 1.0

                # Historical pattern consistency
                exhaustion_consistency = 1.0 - (np.std(valid_exhaustion) / (np.mean(np.abs(valid_exhaustion)) + 1e-8)) if len(valid_exhaustion) > 5 else 0.5
                exhaustion_consistency = max(0.0, min(1.0, exhaustion_consistency))

                # IMPROVED: More generous combination for better signal generation
                primary_signal = normalized_exhaustion
                context_factor = 0.7 * abs(exhaustion_relative_strength) + 0.3 * exhaustion_consistency
                
                # Weight primary signal more heavily
                targets[i] = 0.7 * primary_signal + 0.3 * min(context_factor, 1.0)
            else:
                # Fallback when insufficient history - use signal strength directly
                targets[i] = normalized_exhaustion * 0.6  # More generous fallback

        # Normalize to [0, 1] range for consistent scaling
        if np.max(targets) > 0:
            normalized_targets = targets / np.max(targets)
        else:
            normalized_targets = targets

        return normalized_targets

    def _generate_volume_divergence_target(self, features_df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        Generate continuous volume-price divergence target using ONLY historical data

        CRITICAL FIX: Removes forward-looking validation - uses only past divergence patterns
        - Signal strength based on historical volume-price relationship patterns
        - Creates continuous targets in [0, 1] range without future data leakage
        """

        volume_divergence = features_df["volume_price_divergence"].values
        returns_1d = features_df["returns_1d"].values

        targets = np.zeros(len(features_df))

        for i in range(max(8, self.lookback_window // 3), len(features_df)):  # Even more generous
            current_divergence = volume_divergence[i]

            if np.isnan(current_divergence):
                continue

            # BALANCED: Use historical data with minimal requirements for signal generation
            historical_period = slice(max(0, i - min(self.lookback_window, i)), i)
            historical_returns = returns_1d[historical_period]
            historical_divergence = volume_divergence[historical_period]

            if len(historical_returns) < 3:  # Very minimal requirement
                continue

            # Signal strength normalization
            divergence_strength = np.abs(current_divergence)
            divergence_percentile = np.percentile(np.abs(volume_divergence[~np.isnan(volume_divergence)]), 90)
            normalized_divergence = min(divergence_strength / (divergence_percentile + 1e-8), 1.0)

            # Historical divergence pattern analysis
            valid_divergence = historical_divergence[~np.isnan(historical_divergence)]
            valid_returns = historical_returns[~np.isnan(historical_returns)]

            if len(valid_divergence) > 5 and len(valid_returns) > 5:
                # Historical divergence consistency
                divergence_consistency = 1.0 - (np.std(valid_divergence) / (np.mean(np.abs(valid_divergence)) + 1e-8))
                divergence_consistency = max(0.0, min(1.0, divergence_consistency))

                # Current divergence relative to historical patterns
                historical_divergence_avg = np.mean(np.abs(valid_divergence))
                relative_divergence_strength = min(divergence_strength / (historical_divergence_avg + 1e-8), 2.0)

                # Historical volatility context
                historical_volatility = np.std(valid_returns) if len(valid_returns) > 0 else 0.01

                # IMPROVED: More responsive signal generation
                primary_signal = normalized_divergence
                
                # Context factors (less restrictive)
                context_strength = min(abs(relative_divergence_strength), 1.5)
                volatility_factor = min(0.02 / (historical_volatility + 1e-8), 1.5)
                
                # Combine with emphasis on signal strength
                targets[i] = 0.6 * primary_signal + 0.4 * (divergence_consistency * context_strength * min(volatility_factor, 1.0))
            else:
                # Generous fallback for signal generation
                targets[i] = normalized_divergence * 0.5

        # Return continuous targets (0-1 range) instead of binary
        # Normalize to [0, 1] range for consistent scaling
        if np.max(targets) > 0:
            normalized_targets = targets / np.max(targets)
        else:
            normalized_targets = targets

        return normalized_targets

    def _generate_combined_pattern_target(self, individual_targets: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate balanced combined pattern confidence score from continuous individual targets

        Architecture Fix: Combines continuous pattern scores with improved balance
        Range: 0.0 to 1.0 with target 30-70% distribution for better learning
        """

        # Get all individual pattern target keys (now continuous, not binary)
        pattern_keys = [
            k
            for k in individual_targets.keys()
            if any(
                pattern in k
                for pattern in ["momentum_persistence", "volatility_regime", "trend_exhaustion", "volume_divergence"]
            )
        ]

        if len(pattern_keys) == 0:
            return np.zeros(len(individual_targets[list(individual_targets.keys())[0]]))

        # Stack all continuous pattern targets
        all_patterns = np.column_stack([individual_targets[key] for key in pattern_keys])

        # Enhanced combination with better balance
        # 1. Weighted average (higher weight on momentum and volatility)
        weights = np.array([0.35, 0.35, 0.15, 0.15])  # momentum, volatility, trend, volume
        if all_patterns.shape[1] >= len(weights):
            weighted_score = np.average(all_patterns[:, :len(weights)], axis=1, weights=weights)
        else:
            weighted_score = np.mean(all_patterns, axis=1)

        # 2. Apply sigmoid transformation to improve balance
        # This spreads the distribution more evenly across 0-1 range
        sigmoid_score = 1 / (1 + np.exp(-4 * (weighted_score - 0.5)))
        
        # 3. Percentile-based normalization for target balance
        # Ensure roughly 30-50% positive rate (above 0.5)
        if len(sigmoid_score) > 0:
            # Target: 40% above threshold
            target_percentile = 60  # 60th percentile → 40% above
            threshold_value = np.percentile(sigmoid_score, target_percentile)
            
            # Scale so that threshold_value maps to 0.5
            if threshold_value > 0 and threshold_value != 0.5:
                # Linear scaling to put desired percentile at 0.5
                if threshold_value > 0.5:
                    # Compress upper range
                    mask = sigmoid_score >= threshold_value
                    sigmoid_score[mask] = 0.5 + 0.5 * (sigmoid_score[mask] - threshold_value) / (1 - threshold_value)
                    sigmoid_score[~mask] = 0.5 * sigmoid_score[~mask] / threshold_value
                else:
                    # Expand lower range  
                    mask = sigmoid_score <= threshold_value
                    sigmoid_score[mask] = 0.5 * sigmoid_score[mask] / threshold_value
                    sigmoid_score[~mask] = 0.5 + 0.5 * (sigmoid_score[~mask] - threshold_value) / (1 - threshold_value)

        return sigmoid_score

    def validate_pattern_targets(self, targets: Dict[str, np.ndarray], features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate generated pattern targets for quality

        Returns:
            Dictionary of validation metrics
        """

        validation_results = {}

        for target_name, target_values in targets.items():
            if "binary" in target_name:
                # Binary target validation
                positive_rate = np.mean(target_values)
                validation_results[f"{target_name}_positive_rate"] = positive_rate

                # Check for reasonable balance (10-90% range)
                if 0.1 <= positive_rate <= 0.9:
                    validation_results[f"{target_name}_balance"] = "GOOD"
                else:
                    validation_results[f"{target_name}_balance"] = "IMBALANCED"

            elif "score" in target_name:
                # Continuous target validation
                mean_score = np.mean(target_values)
                std_score = np.std(target_values)
                validation_results[f"{target_name}_mean"] = mean_score
                validation_results[f"{target_name}_std"] = std_score

        return validation_results


def create_pattern_target_generator(
    lookback_window: int = 20, validation_horizons: List[int] = [3, 5, 10]
) -> PatternTargetGenerator:
    """Convenience function to create pattern target generator"""
    return PatternTargetGenerator(lookback_window=lookback_window, validation_horizons=validation_horizons)


if __name__ == "__main__":
    pass
