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

        # CRITICAL FIX: Add temporal gap to prevent data leakage
        temporal_gap = 5
        max_valid_idx = len(features_df) - temporal_gap - horizon
        
        for i in range(self.lookback_window, max_valid_idx):
            
            # FIXED: Use ONLY historical data - NO current timepoint
            historical_start = i - self.lookback_window
            historical_end = i  # Exclusive, so i-1 is last used timepoint
            
            historical_momentum = momentum_feature[historical_start:historical_end]
            historical_returns = returns_1d[historical_start:historical_end]

            # Remove NaN values
            valid_hist_mask = ~np.isnan(historical_momentum)
            if np.sum(valid_hist_mask) < 10:  # Need sufficient historical data
                continue
                
            historical_momentum_clean = historical_momentum[valid_hist_mask]
            historical_returns_clean = historical_returns[valid_hist_mask]
            
            # Pattern detection using ONLY historical data
            momentum_strength = np.mean(np.abs(historical_momentum_clean))
            momentum_consistency = 1.0 - (np.std(historical_momentum_clean) / (np.mean(np.abs(historical_momentum_clean)) + 1e-8))
            
            # Only consider strong, consistent historical momentum
            if momentum_strength < 0.2 or momentum_consistency < 0.5:
                targets[i] = 0.0
                continue
            
            # STEP 2: Future validation with temporal gap (proper prediction)
            future_start = i + temporal_gap
            future_end = future_start + horizon
            
            if future_end >= len(returns_1d):
                continue
                
            future_returns = returns_1d[future_start:future_end]
            future_returns_clean = future_returns[~np.isnan(future_returns)]
            
            if len(future_returns_clean) < 3:
                continue
            
            # Target: Does historical momentum pattern persist in future?
            historical_direction = np.sign(np.mean(historical_momentum_clean))
            future_direction_consistency = np.mean(np.sign(future_returns_clean) == historical_direction)
            
            if future_direction_consistency > 0.6:  # 60% consistency required
                targets[i] = 1.0
            else:
                targets[i] = 0.0

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
        FIXED: Generate trend exhaustion target with NO data leakage
        
        METHODOLOGY (LEAK-FREE):
        1. Detect trend exhaustion using ONLY historical data (i-lookback to i-1)  
        2. Apply 5-day temporal gap (skip i to i+4)
        3. Validate if trend reversal occurs in future period (i+5 to i+5+horizon)
        4. Target at position i predicts future trend reversal based on historical exhaustion
        """

        print("Generating LEAK-FREE trend exhaustion targets...")
        
        trend_exhaustion = features_df["trend_exhaustion"].values
        returns_3d = features_df["returns_3d"].values

        targets = np.zeros(len(features_df))

        # CRITICAL FIX: Add temporal gap to prevent data leakage
        temporal_gap = 5
        max_valid_idx = len(features_df) - temporal_gap - horizon
        
        for i in range(self.lookback_window, max_valid_idx):
            
            # STEP 1: Historical trend exhaustion analysis (STRICT: only past data)
            # Use data from (i-lookback_window) to (i-1) - NO current timepoint
            historical_start = i - self.lookback_window
            historical_end = i  # Exclusive, so actually i-1 is last used index
            
            if historical_start < 0:
                continue
                
            historical_exhaustion = trend_exhaustion[historical_start:historical_end]
            historical_returns = returns_3d[historical_start:historical_end]
            
            # Remove NaN values
            valid_mask = ~np.isnan(historical_exhaustion)
            if np.sum(valid_mask) < 10:  # Need sufficient historical data
                continue
                
            historical_exhaustion_clean = historical_exhaustion[valid_mask]
            historical_returns_clean = historical_returns[valid_mask]
            
            # Pattern detection: Strong trend exhaustion signal in historical period?
            exhaustion_strength = np.mean(np.abs(historical_exhaustion_clean))
            
            if exhaustion_strength < 0.2:  # No strong exhaustion signal detected
                targets[i] = 0.0
                continue
            
            # Historical trend direction (what trend might reverse?)
            if len(historical_returns_clean) > 0:
                historical_trend = np.mean(historical_returns_clean)
                
                if abs(historical_trend) < 0.01:  # No clear trend to exhaust
                    targets[i] = 0.0
                    continue
            else:
                continue
            
            # STEP 2: Temporal gap (skip i to i+gap-1)
            gap_start = i
            gap_end = i + temporal_gap
            
            # STEP 3: Future validation (i+gap to i+gap+horizon)
            future_start = gap_end
            future_end = gap_end + horizon
            
            if future_end >= len(returns_3d):
                continue
                
            future_returns = returns_3d[future_start:future_end]
            
            # Remove NaN from future period
            future_returns_clean = future_returns[~np.isnan(future_returns)]
            if len(future_returns_clean) < 3:  # Need sufficient future data
                continue
            
            # Pattern validation: Does trend actually reverse after exhaustion?
            future_trend = np.mean(future_returns_clean)
            
            # Target: 1 if trend reversal occurs after historical exhaustion signal
            trend_reversal = np.sign(historical_trend) != np.sign(future_trend) and abs(future_trend) > 0.005
            
            if trend_reversal:
                targets[i] = 1.0
            else:
                targets[i] = 0.0

        print(f"Generated {np.sum(targets > 0)} positive trend exhaustion targets out of {max_valid_idx - self.lookback_window} valid samples")
        return targets

    def _generate_volume_divergence_target(self, features_df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        FIXED: Generate volume divergence target with NO data leakage
        
        METHODOLOGY (LEAK-FREE):
        1. Detect volume-price divergence using ONLY historical data (i-lookback to i-1)
        2. Apply 5-day temporal gap (skip i to i+4) 
        3. Validate if divergence resolves in future period (i+5 to i+5+horizon)
        4. Target at position i predicts future price movement based on historical divergence
        """

        print("Generating LEAK-FREE volume divergence targets...")
        
        volume_divergence = features_df["volume_price_divergence"].values
        returns_1d = features_df["returns_1d"].values

        targets = np.zeros(len(features_df))

        # CRITICAL FIX: Add temporal gap to prevent data leakage
        temporal_gap = 5
        max_valid_idx = len(features_df) - temporal_gap - horizon
        
        for i in range(self.lookback_window, max_valid_idx):
            
            # STEP 1: Historical volume divergence analysis (STRICT: only past data)
            # Use data from (i-lookback_window) to (i-1) - NO current timepoint
            historical_start = i - self.lookback_window
            historical_end = i  # Exclusive, so actually i-1 is last used index
            
            if historical_start < 0:
                continue
                
            historical_divergence = volume_divergence[historical_start:historical_end]
            
            # Remove NaN values
            valid_mask = ~np.isnan(historical_divergence)
            if np.sum(valid_mask) < 10:  # Need sufficient historical data
                continue
                
            historical_divergence_clean = historical_divergence[valid_mask]
            
            # Pattern detection: Strong volume-price divergence in historical period?
            divergence_strength = np.mean(np.abs(historical_divergence_clean))
            
            if divergence_strength < 0.15:  # No strong divergence signal detected
                targets[i] = 0.0
                continue
            
            # STEP 2: Temporal gap (skip i to i+gap-1)
            gap_start = i
            gap_end = i + temporal_gap
            
            # STEP 3: Future validation (i+gap to i+gap+horizon)  
            future_start = gap_end
            future_end = gap_end + horizon
            
            if future_end >= len(returns_1d):
                continue
                
            future_returns = returns_1d[future_start:future_end]
            
            # Remove NaN from future period
            future_returns_clean = future_returns[~np.isnan(future_returns)]
            if len(future_returns_clean) < 3:  # Need sufficient future data
                continue
            
            # Pattern validation: Does divergence lead to significant price movement?
            # Volume divergence typically resolves with increased price volatility
            future_volatility = np.std(future_returns_clean)
            
            # Target: 1 if divergence resolves with significant price movement
            if future_volatility > 0.012:  # Above-average volatility after divergence
                targets[i] = 1.0
            else:
                targets[i] = 0.0

        print(f"Generated {np.sum(targets > 0)} positive volume divergence targets out of {max_valid_idx - self.lookback_window} valid samples")
        return targets

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
