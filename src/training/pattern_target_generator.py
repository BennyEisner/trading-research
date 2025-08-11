#!/usr/bin/env python3

"""
Pattern Target Generator for Pattern Detection Validation
Generates binary targets for pattern resolution validation instead of return prediction
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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

    def _compute_dynamic_horizon(self, features_df: pd.DataFrame, base_horizon: int = 5, 
                               min_horizon: int = 3, max_horizon: int = 15) -> np.ndarray:
        """
        Compute volatility-adjusted horizons per sample
        
        Logic:
        - Low volatility: Longer horizons (patterns take time to resolve)
        - High volatility: Shorter horizons (patterns resolve quickly)
        
        Args:
            features_df: DataFrame with features including returns_1d
            base_horizon: Base horizon to scale from
            min_horizon: Minimum horizon allowed
            max_horizon: Maximum horizon allowed
            
        Returns:
            Array of dynamic horizons per sample
        """
        returns_1d = features_df["returns_1d"].values
        horizons = np.full(len(features_df), base_horizon)
        
        # Rolling volatility calculation (20-day window to match LSTM sequence length)
        rolling_vol = pd.Series(returns_1d).rolling(window=self.lookback_window, min_periods=10).std()
        
        # Remove NaN values for percentile calculation
        valid_vol = rolling_vol.dropna()
        if len(valid_vol) < 50:  # Need sufficient data for robust percentiles
            print("Warning: Insufficient volatility data for dynamic horizons, using base horizon")
            return horizons
        
        # Volatility percentiles for scaling
        vol_25th = np.percentile(valid_vol, 25)  # Low volatility threshold
        vol_75th = np.percentile(valid_vol, 75)  # High volatility threshold
        
        print(f"Volatility thresholds: 25th={vol_25th:.4f}, 75th={vol_75th:.4f}")
        
        for i in range(len(features_df)):
            if i < self.lookback_window or np.isnan(rolling_vol.iloc[i]):
                continue  # Use base horizon for early samples or NaN values
                
            current_vol = rolling_vol.iloc[i]
            
            # Scale horizon based on volatility regime
            if current_vol < vol_25th:  # Low volatility - extend horizon
                horizons[i] = min(max_horizon, base_horizon + 3)
            elif current_vol > vol_75th:  # High volatility - shorten horizon  
                horizons[i] = max(min_horizon, base_horizon - 2)
            # else: use base_horizon for medium volatility
        
        horizon_counts = np.bincount(horizons.astype(int))
        print(f"Dynamic horizon distribution: min={np.min(horizons)}, max={np.max(horizons)}, mean={np.mean(horizons):.1f}")
        
        return horizons

    def _compute_adaptive_thresholds(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Replace fixed thresholds with quantile-based thresholds for balanced positive rates
        
        Args:
            features_df: DataFrame with pattern features
            
        Returns:
            Dictionary of adaptive thresholds per pattern type
        """
        thresholds = {}
        
        # Momentum persistence threshold (replace fixed 0.2)
        momentum_values = np.abs(features_df["momentum_persistence_7d"].dropna())
        if len(momentum_values) > 10:
            thresholds["momentum_strength"] = np.percentile(momentum_values, 70)  # Target 30% positive rate
            print(f"Adaptive momentum threshold: {thresholds['momentum_strength']:.4f} (was fixed 0.2)")
        else:
            thresholds["momentum_strength"] = 0.2  # Fallback
        
        # Volatility regime threshold
        vol_change_values = np.abs(features_df["volatility_regime_change"].dropna())
        if len(vol_change_values) > 10:
            thresholds["volatility_strength"] = np.percentile(vol_change_values, 75)  # Target 25% positive rate
            print(f"Adaptive volatility threshold: {thresholds['volatility_strength']:.4f}")
        else:
            thresholds["volatility_strength"] = 0.15  # Fallback
        
        # Trend exhaustion threshold  
        trend_values = np.abs(features_df["trend_exhaustion"].dropna())
        if len(trend_values) > 10:
            thresholds["trend_strength"] = np.percentile(trend_values, 80)  # Target 20% positive rate
            print(f"Adaptive trend threshold: {thresholds['trend_strength']:.4f}")
        else:
            thresholds["trend_strength"] = 0.2  # Fallback
        
        # Volume divergence threshold
        volume_values = np.abs(features_df["volume_price_divergence"].dropna()) 
        if len(volume_values) > 10:
            thresholds["volume_strength"] = np.percentile(volume_values, 75)  # Target 25% positive rate
            print(f"Adaptive volume threshold: {thresholds['volume_strength']:.4f}")
        else:
            thresholds["volume_strength"] = 0.15  # Fallback
            
        return thresholds

    def _get_fixed_thresholds(self) -> Dict[str, float]:
        """Return fixed thresholds as fallback when adaptive thresholds are disabled"""
        return {
            "momentum_strength": 0.2,
            "volatility_strength": 0.15,
            "trend_strength": 0.2,
            "volume_strength": 0.15
        }

    def calibrate_targets(self, targets_dict: Dict[str, np.ndarray], 
                         target_positive_rates: Dict[str, Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Calibrate pattern targets to achieve desired positive rates for balanced learning
        
        Args:
            targets_dict: Generated pattern targets {pattern_name: np.ndarray}
            target_positive_rates: Dict of {pattern: (min_rate, max_rate)} ranges
                                  Default: 20-40% for most patterns
        
        Returns:
            Dictionary with calibrated targets and calibration parameters for live inference
        """
        if target_positive_rates is None:
            target_positive_rates = {
                "momentum_persistence": (0.20, 0.40),
                "volatility_regime": (0.20, 0.40), 
                "trend_exhaustion": (0.15, 0.35),  # Slightly lower as trend reversals are rarer
                "volume_divergence": (0.20, 0.40)
            }
        
        calibration_params = {}
        calibrated_targets = {}
        
        print("Calibrating targets for balanced positive rates...")
        
        for pattern_name, targets in targets_dict.items():
            if pattern_name in target_positive_rates:
                min_rate, max_rate = target_positive_rates[pattern_name]
                target_rate = (min_rate + max_rate) / 2.0  # Target middle of range
                
                # Current positive rate (using threshold 0.5 for binary classification)
                current_rate = np.mean(targets > 0.5)
                
                print(f"{pattern_name}: current rate={current_rate:.3f}, target range=({min_rate:.3f}, {max_rate:.3f})")
                
                if current_rate < min_rate or current_rate > max_rate:
                    # Compute threshold adjustment to achieve target rate
                    if current_rate > 0 and current_rate < 1:
                        # Find threshold that gives us target_rate above it
                        sorted_targets = np.sort(targets)
                        threshold_idx = int((1.0 - target_rate) * len(sorted_targets))
                        optimal_threshold = sorted_targets[max(0, min(threshold_idx, len(sorted_targets)-1))]
                        
                        # Apply logistic scaling to adjust distribution around optimal threshold
                        scaling_factor = 4.0  # Controls steepness of sigmoid
                        calibrated = 1.0 / (1.0 + np.exp(-scaling_factor * (targets - optimal_threshold)))
                        
                        # Verify calibration worked
                        new_rate = np.mean(calibrated > 0.5)
                        
                        calibration_params[pattern_name] = {
                            "original_positive_rate": current_rate,
                            "target_positive_rate": target_rate,
                            "achieved_positive_rate": new_rate,
                            "threshold": optimal_threshold,
                            "scaling_factor": scaling_factor,
                            "calibrated": True
                        }
                        calibrated_targets[pattern_name] = calibrated
                        
                        print(f"  -> Calibrated to {new_rate:.3f} using threshold={optimal_threshold:.4f}")
                    else:
                        # Edge case: all 0s or all 1s - cannot calibrate effectively
                        calibration_params[pattern_name] = {
                            "original_positive_rate": current_rate,
                            "calibrated": False,
                            "reason": "uniform_distribution"
                        }
                        calibrated_targets[pattern_name] = targets
                        print(f"  -> Cannot calibrate (uniform distribution)")
                else:
                    # No calibration needed - already in target range
                    calibration_params[pattern_name] = {
                        "original_positive_rate": current_rate,
                        "calibrated": False,
                        "reason": "within_target_range"
                    }
                    calibrated_targets[pattern_name] = targets
                    print(f"  -> No calibration needed (within range)")
            else:
                # Pattern not in target calibration list
                calibrated_targets[pattern_name] = targets
        
        return {
            "calibrated_targets": calibrated_targets,
            "calibration_params": calibration_params,
            "positive_rates": {name: np.mean(targets > 0.5) for name, targets in calibrated_targets.items()}
        }

    def generate_all_pattern_targets(
        self, features_df: pd.DataFrame, 
        base_horizon: int = 5,
        enable_dynamic_horizons: bool = True,
        enable_adaptive_thresholds: bool = True,
        enable_target_calibration: bool = True
    ) -> Dict[str, Any]:
        """
        Generate all pattern detection targets with dynamic horizons and adaptive thresholds

        Args:
            features_df: DataFrame with calculated pattern features
            base_horizon: Base validation horizon (default 5 days for swing trading)
            enable_dynamic_horizons: Whether to use volatility-adjusted horizons
            enable_adaptive_thresholds: Whether to use quantile-based thresholds
            enable_target_calibration: Whether to calibrate targets for balanced positive rates

        Returns:
            Dictionary containing pattern targets, calibration params, and metadata
        """

        print(f"Generating enhanced pattern targets for {len(features_df)} samples...")
        print(f"Configuration: dynamic_horizons={enable_dynamic_horizons}, adaptive_thresholds={enable_adaptive_thresholds}, target_calibration={enable_target_calibration}")

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

        # Phase 1: Compute dynamic horizons per sample
        if enable_dynamic_horizons:
            dynamic_horizons = self._compute_dynamic_horizon(features_df, base_horizon)
            print(f"Dynamic horizons enabled: min={np.min(dynamic_horizons)}, max={np.max(dynamic_horizons)}, mean={np.mean(dynamic_horizons):.1f}")
        else:
            dynamic_horizons = np.full(len(features_df), base_horizon)
            print(f"Using fixed horizon: {base_horizon}")

        # Phase 2: Compute adaptive thresholds
        if enable_adaptive_thresholds:
            adaptive_thresholds = self._compute_adaptive_thresholds(features_df)
            print(f"Adaptive thresholds enabled: {adaptive_thresholds}")
        else:
            adaptive_thresholds = self._get_fixed_thresholds()
            print(f"Using fixed thresholds: {adaptive_thresholds}")

        # Phase 3: Generate individual pattern targets with dynamic parameters
        targets = {}

        print("Generating individual pattern targets...")
        targets["momentum_persistence"] = self._generate_momentum_persistence_target(
            features_df, dynamic_horizons, adaptive_thresholds["momentum_strength"]
        )

        targets["volatility_regime"] = self._generate_volatility_regime_target(
            features_df, dynamic_horizons, adaptive_thresholds["volatility_strength"]
        )

        targets["trend_exhaustion"] = self._generate_trend_exhaustion_target(
            features_df, dynamic_horizons, adaptive_thresholds["trend_strength"]
        )

        targets["volume_divergence"] = self._generate_volume_divergence_target(
            features_df, dynamic_horizons, adaptive_thresholds["volume_strength"]
        )

        # Phase 4: Target calibration for balanced positive rates
        if enable_target_calibration:
            calibration_results = self.calibrate_targets(targets)
            calibrated_targets = calibration_results["calibrated_targets"]
            
            # Update targets with calibrated versions
            targets.update(calibrated_targets)
            
            print(f"Target calibration completed:")
            for pattern, rate in calibration_results["positive_rates"].items():
                print(f"   - {pattern}: {rate:.3f} positive rate (calibrated)")
            
            # Optional: Create combined pattern confidence target for backward compatibility
            targets["pattern_confidence_score"] = self._generate_combined_pattern_target(targets)
            
            return {
                **targets,
                "calibration_params": calibration_results["calibration_params"],
                "positive_rates": calibration_results["positive_rates"],
                "dynamic_horizons": dynamic_horizons,
                "adaptive_thresholds": adaptive_thresholds,
                "configuration": {
                    "enable_dynamic_horizons": enable_dynamic_horizons,
                    "enable_adaptive_thresholds": enable_adaptive_thresholds,
                    "enable_target_calibration": enable_target_calibration,
                    "base_horizon": base_horizon
                }
            }
        else:
            # No calibration - calculate basic positive rates
            positive_rates = {name: np.mean(targets > 0.5) for name, targets in targets.items()}
            
            print(f"Generated pattern targets (no calibration):")
            for target_name, rate in positive_rates.items():
                print(f"   - {target_name}: {rate:.3f} positive rate")
            
            # Optional: Create combined pattern confidence target for backward compatibility
            targets["pattern_confidence_score"] = self._generate_combined_pattern_target(targets)
            
            return {
                **targets,
                "positive_rates": positive_rates,
                "dynamic_horizons": dynamic_horizons,
                "adaptive_thresholds": adaptive_thresholds,
                "configuration": {
                    "enable_dynamic_horizons": enable_dynamic_horizons,
                    "enable_adaptive_thresholds": enable_adaptive_thresholds,
                    "enable_target_calibration": enable_target_calibration,
                    "base_horizon": base_horizon
                }
            }

    def _generate_momentum_persistence_target(self, features_df: pd.DataFrame, 
                                            dynamic_horizons: np.ndarray, 
                                            strength_threshold: float) -> np.ndarray:
        """
        Generate continuous momentum persistence target with dynamic horizons and adaptive thresholds

        Enhanced Features:
        - Dynamic horizons per sample based on volatility
        - Adaptive threshold based on momentum strength percentiles
        - Signal strength based on historical momentum consistency
        - Creates continuous targets in [0, 1] range without future data leakage
        
        Args:
            features_df: DataFrame with pattern features
            dynamic_horizons: Array of horizons per sample (volatility-adjusted)
            strength_threshold: Adaptive threshold for momentum strength filtering
        """

        momentum_feature = features_df["momentum_persistence_7d"].values
        returns_1d = features_df["returns_1d"].values

        targets = np.zeros(len(features_df))

        # CRITICAL FIX: Add temporal gap to prevent data leakage
        temporal_gap = 5
        # Use max dynamic horizon for conservative boundary calculation
        max_horizon = int(np.max(dynamic_horizons))
        max_valid_idx = len(features_df) - temporal_gap - max_horizon
        
        print(f"Momentum persistence: using adaptive threshold {strength_threshold:.4f} and dynamic horizons (max={max_horizon})")
        
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
            
            # Use adaptive threshold instead of fixed 0.2
            if momentum_strength < strength_threshold or momentum_consistency < 0.5:
                targets[i] = 0.0
                continue
            
            # STEP 2: Future validation with temporal gap using dynamic horizon for this sample
            current_horizon = int(dynamic_horizons[i])
            future_start = i + temporal_gap
            future_end = future_start + current_horizon
            
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

    def _generate_volatility_regime_target(self, features_df: pd.DataFrame, 
                                          dynamic_horizons: np.ndarray, 
                                          strength_threshold: float) -> np.ndarray:
        """
        Generate continuous volatility regime target with dynamic horizons and adaptive thresholds

        Enhanced Features:
        - Dynamic horizons per sample based on volatility
        - Adaptive threshold based on volatility regime change percentiles  
        - Signal strength based on historical volatility regime consistency
        - Creates continuous targets in [0, 1] range without future data leakage
        
        Args:
            features_df: DataFrame with pattern features
            dynamic_horizons: Array of horizons per sample (volatility-adjusted)
            strength_threshold: Adaptive threshold for volatility strength filtering
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
                # Signal strength using adaptive threshold instead of fixed percentile
                signal_strength = np.abs(current_vol_signal)
                # Use adaptive threshold instead of computing percentile each time
                normalized_signal = min(signal_strength / (strength_threshold + 1e-8), 1.0)

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

    def _generate_trend_exhaustion_target(self, features_df: pd.DataFrame, 
                                         dynamic_horizons: np.ndarray, 
                                         strength_threshold: float) -> np.ndarray:
        """
        Generate trend exhaustion target with dynamic horizons and adaptive thresholds
        
        Enhanced Features:
        - Dynamic horizons per sample based on volatility
        - Adaptive threshold based on trend exhaustion strength percentiles
        - Leak-free temporal structure maintained
        
        METHODOLOGY (LEAK-FREE):
        1. Detect trend exhaustion using ONLY historical data (i-lookback to i-1)  
        2. Apply 5-day temporal gap (skip i to i+4)
        3. Validate if trend reversal occurs in future period using dynamic horizon
        4. Target at position i predicts future trend reversal based on historical exhaustion
        
        Args:
            features_df: DataFrame with pattern features
            dynamic_horizons: Array of horizons per sample (volatility-adjusted)
            strength_threshold: Adaptive threshold for trend exhaustion strength filtering
        """

        print("Generating LEAK-FREE trend exhaustion targets...")
        
        trend_exhaustion = features_df["trend_exhaustion"].values
        returns_3d = features_df["returns_3d"].values

        targets = np.zeros(len(features_df))

        # CRITICAL FIX: Add temporal gap to prevent data leakage
        temporal_gap = 5
        # Use max dynamic horizon for conservative boundary calculation
        max_horizon = int(np.max(dynamic_horizons))
        max_valid_idx = len(features_df) - temporal_gap - max_horizon
        
        print(f"Trend exhaustion: using adaptive threshold {strength_threshold:.4f} and dynamic horizons (max={max_horizon})")
        
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
            
            # Use adaptive threshold instead of fixed 0.2
            if exhaustion_strength < strength_threshold:  # No strong exhaustion signal detected
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
            
            # STEP 3: Future validation using dynamic horizon for this sample
            current_horizon = int(dynamic_horizons[i])
            future_start = gap_end
            future_end = gap_end + current_horizon
            
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

    def _generate_volume_divergence_target(self, features_df: pd.DataFrame, 
                                          dynamic_horizons: np.ndarray, 
                                          strength_threshold: float) -> np.ndarray:
        """
        Generate volume divergence target with dynamic horizons and adaptive thresholds
        
        Enhanced Features:
        - Dynamic horizons per sample based on volatility
        - Adaptive threshold based on volume divergence strength percentiles
        - Leak-free temporal structure maintained
        
        METHODOLOGY (LEAK-FREE):
        1. Detect volume-price divergence using ONLY historical data (i-lookback to i-1)
        2. Apply 5-day temporal gap (skip i to i+4) 
        3. Validate if divergence resolves in future period using dynamic horizon
        4. Target at position i predicts future price movement based on historical divergence
        
        Args:
            features_df: DataFrame with pattern features
            dynamic_horizons: Array of horizons per sample (volatility-adjusted)
            strength_threshold: Adaptive threshold for volume divergence strength filtering
        """

        print("Generating LEAK-FREE volume divergence targets...")
        
        volume_divergence = features_df["volume_price_divergence"].values
        returns_1d = features_df["returns_1d"].values

        targets = np.zeros(len(features_df))

        # CRITICAL FIX: Add temporal gap to prevent data leakage
        temporal_gap = 5
        # Use max dynamic horizon for conservative boundary calculation
        max_horizon = int(np.max(dynamic_horizons))
        max_valid_idx = len(features_df) - temporal_gap - max_horizon
        
        print(f"Volume divergence: using adaptive threshold {strength_threshold:.4f} and dynamic horizons (max={max_horizon})")
        
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
            
            # Use adaptive threshold instead of fixed 0.15
            if divergence_strength < strength_threshold:  # No strong divergence signal detected
                targets[i] = 0.0
                continue
            
            # STEP 2: Temporal gap (skip i to i+gap-1)
            gap_start = i
            gap_end = i + temporal_gap
            
            # STEP 3: Future validation using dynamic horizon for this sample
            current_horizon = int(dynamic_horizons[i])
            future_start = gap_end
            future_end = gap_end + current_horizon
            
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
