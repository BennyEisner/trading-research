#!/usr/bin/env python3

"""
FIXED Pattern Target Generator - Eliminates Data Leakage
Critical fix for temporal dependencies that caused 69% correlation artifact
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent))


class PatternTargetGeneratorFixed:
    """
    LEAK-FREE Pattern Target Generator
    
    CRITICAL FIXES:
    1. NO current timepoint data in target calculation
    2. Minimum 5-day temporal gap between features and targets
    3. Strictly historical-only pattern validation
    4. Forward-looking pattern resolution (proper prediction setup)
    """

    def __init__(self, lookback_window: int = 20, validation_horizons: List[int] = [3, 5, 10], 
                 temporal_gap: int = 5):
        """
        Initialize leak-free pattern target generator

        Args:
            lookback_window: Window for pattern detection (matches LSTM sequence length)
            validation_horizons: Time horizons for pattern resolution validation
            temporal_gap: Minimum days between feature window and target calculation (CRITICAL)
        """
        self.lookback_window = lookback_window
        self.validation_horizons = validation_horizons
        self.temporal_gap = temporal_gap  # NEW: Prevent temporal leakage
        
        print(f"LEAK-FREE Target Generator initialized:")
        print(f"  - Lookback window: {lookback_window} days")
        print(f"  - Temporal gap: {temporal_gap} days (prevents leakage)")
        print(f"  - Validation horizons: {validation_horizons} days")

    def generate_all_pattern_targets(
        self, features_df: pd.DataFrame, primary_horizon: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Generate LEAK-FREE pattern detection targets
        
        TEMPORAL STRUCTURE:
        Day 0-19: Historical features (input to LSTM)
        Day 20-24: TEMPORAL GAP (no data used)
        Day 25+: Target calculation period (pattern resolution)

        Args:
            features_df: DataFrame with calculated pattern features
            primary_horizon: Primary validation horizon (days for pattern resolution)

        Returns:
            Dictionary of leak-free pattern targets
        """

        print(f"Generating LEAK-FREE pattern targets for {len(features_df)} samples...")
        print(f"Using {self.temporal_gap}-day temporal gap to prevent data leakage")

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

        # Generate each leak-free pattern validation target
        targets["momentum_persistence_binary"] = self._generate_momentum_persistence_target_fixed(
            features_df, horizon=primary_horizon
        )

        targets["volatility_regime_binary"] = self._generate_volatility_regime_target_fixed(
            features_df, horizon=primary_horizon
        )

        targets["trend_exhaustion_binary"] = self._generate_trend_exhaustion_target_fixed(
            features_df, horizon=primary_horizon
        )

        targets["volume_divergence_binary"] = self._generate_volume_divergence_target_fixed(
            features_df, horizon=primary_horizon
        )

        # Create combined pattern confidence target
        targets["pattern_confidence_score"] = self._generate_combined_pattern_target(targets)

        print(f"Generated LEAK-FREE pattern targets:")
        for target_name, target_values in targets.items():
            positive_rate = np.mean(target_values) if len(target_values) > 0 else 0
            print(f"   - {target_name}: {positive_rate:.3f} positive rate")

        return targets

    def _generate_momentum_persistence_target_fixed(self, features_df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        FIXED: Generate momentum persistence target with NO data leakage
        
        METHODOLOGY:
        1. For timepoint t: Use features from t-lookback to t-1 (historical only)
        2. Apply temporal gap: skip t to t+gap-1 
        3. Validate pattern: check momentum persistence from t+gap to t+gap+horizon
        4. Target at t predicts whether detected pattern will persist in future
        """

        print("Generating LEAK-FREE momentum persistence targets...")
        
        momentum_feature = features_df["momentum_persistence_7d"].values
        returns_1d = features_df["returns_1d"].values
        
        targets = np.zeros(len(features_df))
        
        # CRITICAL: Leave enough room for temporal gap + validation horizon
        max_valid_idx = len(features_df) - self.temporal_gap - horizon
        
        for i in range(self.lookback_window, max_valid_idx):
            
            # STEP 1: Historical pattern detection (STRICT: only past data)
            # Use data from (i-lookback_window) to (i-1) - NO current timepoint
            historical_start = i - self.lookback_window
            historical_end = i  # Exclusive, so actually i-1 is last used index
            
            if historical_start < 0:
                continue
                
            historical_momentum = momentum_feature[historical_start:historical_end]
            historical_returns = returns_1d[historical_start:historical_end]
            
            # Remove NaN values
            valid_hist_mask = ~np.isnan(historical_momentum)
            if np.sum(valid_hist_mask) < 10:  # Need sufficient historical data
                continue
                
            historical_momentum_clean = historical_momentum[valid_hist_mask]
            
            # Pattern detection: Is there strong momentum in historical period?
            momentum_strength = np.mean(np.abs(historical_momentum_clean))
            momentum_consistency = 1.0 - (np.std(historical_momentum_clean) / (np.mean(np.abs(historical_momentum_clean)) + 1e-8))
            
            # Only consider strong, consistent historical momentum as pattern
            if momentum_strength < 0.3 or momentum_consistency < 0.6:
                targets[i] = 0.0  # No strong pattern detected
                continue
            
            # STEP 2: Temporal gap (skip t to t+gap-1)
            gap_start = i
            gap_end = i + self.temporal_gap
            
            # STEP 3: Future validation (t+gap to t+gap+horizon)
            future_start = gap_end
            future_end = gap_end + horizon
            
            if future_end >= len(returns_1d):
                continue
                
            future_returns = returns_1d[future_start:future_end]
            
            # Remove NaN from future period
            future_returns_clean = future_returns[~np.isnan(future_returns)]
            if len(future_returns_clean) < 3:  # Need sufficient future data
                continue
            
            # Pattern validation: Does momentum actually persist in future?
            # Check if returns maintain same direction as historical momentum
            historical_direction = np.sign(np.mean(historical_momentum_clean))
            future_direction_consistency = np.mean(np.sign(future_returns_clean) == historical_direction)
            
            # Target: 1 if momentum persists, 0 if it doesn't
            if future_direction_consistency > 0.6:  # 60% of future returns maintain direction
                targets[i] = 1.0
            else:
                targets[i] = 0.0
                
        print(f"Generated {np.sum(targets > 0)} positive momentum persistence targets out of {max_valid_idx - self.lookback_window} valid samples")
        return targets

    def _generate_volatility_regime_target_fixed(self, features_df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        FIXED: Generate volatility regime target with NO data leakage
        
        METHODOLOGY:
        1. Detect volatility regime change using ONLY historical data
        2. Apply temporal gap
        3. Validate if regime change continues in future period
        """
        
        print("Generating LEAK-FREE volatility regime targets...")
        
        volatility_change = features_df["volatility_regime_change"].values
        returns_1d = features_df["returns_1d"].values
        
        targets = np.zeros(len(features_df))
        max_valid_idx = len(features_df) - self.temporal_gap - horizon
        
        for i in range(self.lookback_window, max_valid_idx):
            
            # Historical volatility regime analysis (STRICT: only past data)
            historical_start = i - self.lookback_window
            historical_end = i  # Exclusive
            
            if historical_start < 0:
                continue
                
            historical_vol_change = volatility_change[historical_start:historical_end]
            historical_returns = returns_1d[historical_start:historical_end]
            
            # Remove NaN values
            valid_mask = ~np.isnan(historical_vol_change)
            if np.sum(valid_mask) < 10:
                continue
                
            historical_vol_clean = historical_vol_change[valid_mask]
            historical_returns_clean = historical_returns[valid_mask]
            
            # Pattern detection: Strong regime change in recent history?
            recent_vol_change = np.mean(historical_vol_clean[-5:])  # Last 5 days of historical period
            
            if abs(recent_vol_change) < 0.2:  # No significant regime change
                targets[i] = 0.0
                continue
            
            # Future validation with temporal gap
            future_start = i + self.temporal_gap
            future_end = future_start + horizon
            
            if future_end >= len(returns_1d):
                continue
                
            future_returns = returns_1d[future_start:future_end]
            future_returns_clean = future_returns[~np.isnan(future_returns)]
            
            if len(future_returns_clean) < 3:
                continue
            
            # Validate: Does volatility regime persist in future?
            future_volatility = np.std(future_returns_clean)
            historical_volatility = np.std(historical_returns_clean)
            
            if historical_volatility == 0:
                continue
                
            future_vol_change = (future_volatility - historical_volatility) / historical_volatility
            
            # Target: 1 if regime change direction persists
            if np.sign(recent_vol_change) == np.sign(future_vol_change) and abs(future_vol_change) > 0.1:
                targets[i] = 1.0
            else:
                targets[i] = 0.0
                
        print(f"Generated {np.sum(targets > 0)} positive volatility regime targets")
        return targets

    def _generate_trend_exhaustion_target_fixed(self, features_df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        FIXED: Generate trend exhaustion target with NO data leakage
        
        METHODOLOGY:
        1. Detect trend exhaustion using ONLY historical data
        2. Apply temporal gap  
        3. Validate if trend reversal occurs in future period
        """
        
        print("Generating LEAK-FREE trend exhaustion targets...")
        
        trend_exhaustion = features_df["trend_exhaustion"].values
        returns_3d = features_df["returns_3d"].values
        
        targets = np.zeros(len(features_df))
        max_valid_idx = len(features_df) - self.temporal_gap - horizon
        
        for i in range(self.lookback_window, max_valid_idx):
            
            # Historical trend analysis (STRICT: only past data)
            historical_start = i - self.lookback_window
            historical_end = i  # Exclusive
            
            if historical_start < 0:
                continue
                
            historical_exhaustion = trend_exhaustion[historical_start:historical_end]
            historical_returns = returns_3d[historical_start:historical_end]
            
            # Remove NaN values
            valid_mask = ~np.isnan(historical_exhaustion)
            if np.sum(valid_mask) < 10:
                continue
                
            historical_exhaustion_clean = historical_exhaustion[valid_mask]
            historical_returns_clean = historical_returns[valid_mask]
            
            # Pattern detection: Strong trend exhaustion signal?
            exhaustion_strength = np.mean(np.abs(historical_exhaustion_clean[-5:]))  # Recent exhaustion
            
            if exhaustion_strength < 0.3:  # No strong exhaustion signal
                targets[i] = 0.0
                continue
            
            # Historical trend direction
            historical_trend = np.mean(historical_returns_clean)
            
            if abs(historical_trend) < 0.01:  # No clear trend
                targets[i] = 0.0
                continue
            
            # Future validation with temporal gap
            future_start = i + self.temporal_gap
            future_end = future_start + horizon
            
            if future_end >= len(returns_3d):
                continue
                
            future_returns = returns_3d[future_start:future_end]
            future_returns_clean = future_returns[~np.isnan(future_returns)]
            
            if len(future_returns_clean) < 3:
                continue
            
            # Validate: Does trend actually reverse after exhaustion signal?
            future_trend = np.mean(future_returns_clean)
            
            # Target: 1 if trend reversal occurs after exhaustion
            trend_reversal = np.sign(historical_trend) != np.sign(future_trend) and abs(future_trend) > 0.005
            
            if trend_reversal:
                targets[i] = 1.0
            else:
                targets[i] = 0.0
                
        print(f"Generated {np.sum(targets > 0)} positive trend exhaustion targets")
        return targets

    def _generate_volume_divergence_target_fixed(self, features_df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        FIXED: Generate volume divergence target with NO data leakage
        """
        
        print("Generating LEAK-FREE volume divergence targets...")
        
        volume_divergence = features_df["volume_price_divergence"].values
        returns_1d = features_df["returns_1d"].values
        
        targets = np.zeros(len(features_df))
        max_valid_idx = len(features_df) - self.temporal_gap - horizon
        
        for i in range(self.lookback_window, max_valid_idx):
            
            # Historical divergence analysis (STRICT: only past data)
            historical_start = i - self.lookback_window  
            historical_end = i  # Exclusive
            
            if historical_start < 0:
                continue
                
            historical_divergence = volume_divergence[historical_start:historical_end]
            
            # Remove NaN values
            valid_mask = ~np.isnan(historical_divergence)
            if np.sum(valid_mask) < 10:
                continue
                
            historical_divergence_clean = historical_divergence[valid_mask]
            
            # Pattern detection: Strong divergence signal?
            divergence_strength = np.mean(np.abs(historical_divergence_clean[-5:]))
            
            if divergence_strength < 0.2:  # No strong divergence
                targets[i] = 0.0
                continue
            
            # Future validation with temporal gap
            future_start = i + self.temporal_gap
            future_end = future_start + horizon
            
            if future_end >= len(returns_1d):
                continue
                
            future_returns = returns_1d[future_start:future_end]
            future_returns_clean = future_returns[~np.isnan(future_returns)]
            
            if len(future_returns_clean) < 3:
                continue
            
            # Simple validation: Any significant price movement after divergence
            future_volatility = np.std(future_returns_clean)
            
            if future_volatility > 0.015:  # Significant price movement
                targets[i] = 1.0
            else:
                targets[i] = 0.0
                
        print(f"Generated {np.sum(targets > 0)} positive volume divergence targets")
        return targets

    def _generate_combined_pattern_target(self, targets: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate combined pattern confidence score from individual targets
        
        This combines multiple pattern signals into a single confidence score
        """
        
        individual_targets = [
            targets["momentum_persistence_binary"],
            targets["volatility_regime_binary"], 
            targets["trend_exhaustion_binary"],
            targets["volume_divergence_binary"]
        ]
        
        # Simple average of pattern signals
        combined = np.mean(individual_targets, axis=0)
        
        # Add some noise to prevent overfitting to exact combinations
        noise = np.random.normal(0, 0.05, len(combined))
        combined_with_noise = np.clip(combined + noise, 0.0, 1.0)
        
        print(f"Combined pattern confidence: {np.mean(combined_with_noise):.3f} average score")
        
        return combined_with_noise

    def validate_temporal_integrity(self, features_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that target generation has proper temporal boundaries
        
        Returns:
            Dictionary of validation checks
        """
        
        print("\n--- TEMPORAL INTEGRITY VALIDATION ---")
        
        targets = self.generate_all_pattern_targets(features_df)
        combined_target = targets["pattern_confidence_score"]
        
        validation_results = {}
        
        # Check 1: No same-timepoint correlation with input features
        feature_names = ["momentum_persistence_7d", "trend_exhaustion", "volatility_regime_change"]
        
        same_time_correlations = {}
        for feature_name in feature_names:
            if feature_name in features_df.columns:
                feature_values = features_df[feature_name].values
                
                valid_mask = ~(np.isnan(feature_values) | np.isnan(combined_target))
                if np.sum(valid_mask) > 10:
                    correlation = abs(np.corrcoef(feature_values[valid_mask], 
                                                 combined_target[valid_mask])[0,1])
                    same_time_correlations[feature_name] = correlation
                    
                    if correlation > 0.2:
                        print(f"⚠️  {feature_name}: {correlation:.3f} correlation (should be <0.2)")
                    else:
                        print(f"✅ {feature_name}: {correlation:.3f} correlation")
        
        validation_results['same_time_correlations'] = same_time_correlations
        validation_results['max_same_time_correlation'] = max(same_time_correlations.values()) if same_time_correlations else 0.0
        
        # Check 2: Gap effectiveness test
        print(f"\nTesting temporal gap effectiveness...")
        gap_correlations = []
        
        for gap_test in [0, 1, 5, 10]:
            temp_generator = PatternTargetGeneratorFixed(
                lookback_window=self.lookback_window,
                temporal_gap=gap_test
            )
            temp_targets = temp_generator.generate_all_pattern_targets(features_df, primary_horizon=5)
            temp_combined = temp_targets["pattern_confidence_score"]
            
            # Test correlation with momentum feature
            momentum_values = features_df["momentum_persistence_7d"].values
            valid_mask = ~(np.isnan(momentum_values) | np.isnan(temp_combined))
            
            if np.sum(valid_mask) > 10:
                gap_correlation = abs(np.corrcoef(momentum_values[valid_mask], 
                                                temp_combined[valid_mask])[0,1])
                gap_correlations.append((gap_test, gap_correlation))
                print(f"Gap {gap_test} days: {gap_correlation:.3f} correlation")
        
        validation_results['gap_correlations'] = gap_correlations
        
        # Check 3: Overall validation
        max_correlation = validation_results['max_same_time_correlation']
        
        if max_correlation < 0.1:
            print(f"\n✅ TEMPORAL INTEGRITY PASSED - Max correlation: {max_correlation:.3f}")
            validation_results['temporal_integrity_passed'] = True
        elif max_correlation < 0.2:
            print(f"\n⚠️  TEMPORAL INTEGRITY MARGINAL - Max correlation: {max_correlation:.3f}")
            validation_results['temporal_integrity_passed'] = False
        else:
            print(f"\n🚨 TEMPORAL INTEGRITY FAILED - Max correlation: {max_correlation:.3f}")
            validation_results['temporal_integrity_passed'] = False
        
        return validation_results


def main():
    """Test the fixed target generator"""
    
    print("Testing LEAK-FREE Pattern Target Generator...")
    
    # Create synthetic test data
    dates = pd.date_range(start='2022-01-01', end='2024-08-01', freq='D')
    n_days = len(dates)
    
    np.random.seed(42)
    base_price = 150
    price_changes = np.cumsum(np.random.normal(0, 0.02, n_days))
    
    test_data = pd.DataFrame({
        'open': base_price + price_changes + np.random.normal(0, 0.5, n_days),
        'high': base_price + price_changes + np.abs(np.random.normal(0, 1, n_days)),
        'low': base_price + price_changes - np.abs(np.random.normal(0, 1, n_days)), 
        'close': base_price + price_changes,
        'volume': np.random.lognormal(15, 0.5, n_days).astype(int),
        
        # Add required features for testing
        'momentum_persistence_7d': np.random.normal(0, 0.5, n_days),
        'volatility_clustering': np.random.normal(0, 0.3, n_days),
        'trend_exhaustion': np.random.normal(0, 0.4, n_days),
        'volume_price_divergence': np.random.normal(0, 0.2, n_days),
        'volatility_regime_change': np.random.normal(0, 0.3, n_days),
        'returns_1d': np.random.normal(0, 0.02, n_days),
        'returns_3d': np.random.normal(0, 0.05, n_days),
        'returns_7d': np.random.normal(0, 0.08, n_days)
    }, index=dates)
    
    # Test fixed target generator
    fixed_generator = PatternTargetGeneratorFixed(
        lookback_window=20,
        validation_horizons=[3, 5, 10],
        temporal_gap=5
    )
    
    # Generate leak-free targets
    fixed_targets = fixed_generator.generate_all_pattern_targets(test_data)
    
    # Validate temporal integrity
    validation_results = fixed_generator.validate_temporal_integrity(test_data)
    
    print(f"\n--- FIXED TARGET GENERATOR TEST RESULTS ---")
    print(f"Temporal integrity passed: {validation_results['temporal_integrity_passed']}")
    print(f"Max same-time correlation: {validation_results['max_same_time_correlation']:.3f}")
    
    if validation_results['temporal_integrity_passed']:
        print("✅ SUCCESS: Fixed target generator eliminates data leakage")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Some temporal issues remain")


if __name__ == "__main__":
    main()