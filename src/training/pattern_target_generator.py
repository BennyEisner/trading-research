#!/usr/bin/env python3

"""
Pattern Target Generator for Pattern Detection Validation
Generates binary targets for pattern resolution validation instead of return prediction
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import sys

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
    
    def __init__(self, 
                 lookback_window: int = 20,
                 validation_horizons: List[int] = [3, 5, 10]):
        """
        Initialize pattern target generator
        
        Args:
            lookback_window: Window for pattern detection (matches LSTM sequence length)
            validation_horizons: Time horizons for pattern resolution validation
        """
        self.lookback_window = lookback_window
        self.validation_horizons = validation_horizons
        
    def generate_all_pattern_targets(self, 
                                   features_df: pd.DataFrame,
                                   primary_horizon: int = 5) -> Dict[str, np.ndarray]:
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
            'momentum_persistence_7d', 'volatility_clustering', 'trend_exhaustion',
            'volume_price_divergence', 'volatility_regime_change', 'returns_1d',
            'returns_3d', 'returns_7d', 'close'
        ]
        
        missing_features = [f for f in required_features if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        targets = {}
        
        # Generate each pattern validation target
        targets['momentum_persistence_binary'] = self._generate_momentum_persistence_target(
            features_df, horizon=primary_horizon
        )
        
        targets['volatility_regime_binary'] = self._generate_volatility_regime_target(
            features_df, horizon=primary_horizon
        )
        
        targets['trend_exhaustion_binary'] = self._generate_trend_exhaustion_target(
            features_df, horizon=primary_horizon
        )
        
        targets['volume_divergence_binary'] = self._generate_volume_divergence_target(
            features_df, horizon=primary_horizon
        )
        
        # Create combined pattern confidence target
        targets['pattern_confidence_score'] = self._generate_combined_pattern_target(targets)
        
        print(f"Generated pattern targets:")
        for target_name, target_values in targets.items():
            positive_rate = np.mean(target_values) if len(target_values) > 0 else 0
            print(f"   - {target_name}: {positive_rate:.3f} positive rate")
        
        return targets
    
    def _generate_momentum_persistence_target(self, 
                                            features_df: pd.DataFrame, 
                                            horizon: int = 5) -> np.ndarray:
        """
        Generate momentum persistence validation target
        
        Question: When momentum_persistence_7d is high, does momentum actually persist?
        Target: 1 if momentum persists over horizon, 0 otherwise
        """
        
        momentum_feature = features_df['momentum_persistence_7d'].values
        returns_1d = features_df['returns_1d'].values
        
        targets = np.zeros(len(features_df))
        
        for i in range(len(features_df) - horizon):
            # Current momentum signal strength
            current_momentum = momentum_feature[i]
            
            # Look ahead to validate momentum persistence
            future_period = slice(i + 1, i + horizon + 1)
            future_returns = returns_1d[future_period]
            
            if len(future_returns) == 0:
                continue
                
            # High momentum signal (top 30%)
            momentum_threshold = np.percentile(momentum_feature[~np.isnan(momentum_feature)], 70)
            
            if current_momentum > momentum_threshold:
                # Validate: Does momentum actually persist?
                # Persistence = consistent directional movement
                future_signs = np.sign(future_returns[future_returns != 0])
                
                if len(future_signs) > 0:
                    # Momentum persists if >60% of future returns maintain direction
                    persistence_rate = np.max([
                        np.mean(future_signs > 0),  # Upward persistence
                        np.mean(future_signs < 0)   # Downward persistence
                    ])
                    
                    targets[i] = 1 if persistence_rate > 0.6 else 0
        
        return targets
    
    def _generate_volatility_regime_target(self, 
                                         features_df: pd.DataFrame, 
                                         horizon: int = 5) -> np.ndarray:
        """
        Generate volatility regime validation target
        
        Question: When volatility_regime_change is detected, does volatility actually change?
        Target: 1 if volatility regime shifts as predicted, 0 otherwise
        """
        
        volatility_change = features_df['volatility_regime_change'].values
        returns_1d = features_df['returns_1d'].values
        
        targets = np.zeros(len(features_df))
        
        for i in range(self.lookback_window, len(features_df) - horizon):
            current_vol_signal = volatility_change[i]
            
            # Calculate historical volatility (before signal)
            historical_period = slice(i - self.lookback_window, i)
            historical_returns = returns_1d[historical_period]
            historical_vol = np.std(historical_returns) if len(historical_returns) > 0 else 0
            
            # Calculate future volatility (after signal)
            future_period = slice(i + 1, i + horizon + 1)
            future_returns = returns_1d[future_period]
            future_vol = np.std(future_returns) if len(future_returns) > 0 else 0
            
            if historical_vol > 0:
                vol_change_threshold = np.percentile(np.abs(volatility_change[~np.isnan(volatility_change)]), 70)
                
                if abs(current_vol_signal) > vol_change_threshold:
                    # Validate: Does volatility actually change significantly?
                    vol_ratio = future_vol / historical_vol
                    
                    # Significant regime change if volatility changes by >30%
                    targets[i] = 1 if (vol_ratio > 1.3 or vol_ratio < 0.7) else 0
        
        return targets
    
    def _generate_trend_exhaustion_target(self, 
                                        features_df: pd.DataFrame, 
                                        horizon: int = 5) -> np.ndarray:
        """
        Generate trend exhaustion validation target
        
        Question: When trend_exhaustion is detected, does trend actually reverse?
        Target: 1 if trend reverses as predicted, 0 otherwise
        """
        
        trend_exhaustion = features_df['trend_exhaustion'].values
        returns_3d = features_df['returns_3d'].values
        
        targets = np.zeros(len(features_df))
        
        for i in range(self.lookback_window, len(features_df) - horizon):
            current_exhaustion = trend_exhaustion[i]
            
            # Strong exhaustion signal (top 20%)
            exhaustion_threshold = np.percentile(np.abs(trend_exhaustion[~np.isnan(trend_exhaustion)]), 80)
            
            if abs(current_exhaustion) > exhaustion_threshold:
                # Recent trend direction
                recent_trend = returns_3d[i] if not np.isnan(returns_3d[i]) else 0
                
                # Future trend direction
                future_period = slice(i + 1, i + horizon + 1)
                future_returns = returns_3d[future_period]
                future_trend = np.mean(future_returns) if len(future_returns) > 0 else 0
                
                # Trend reversal: recent and future trends have opposite signs
                if abs(recent_trend) > 0.01 and abs(future_trend) > 0.01:  # Minimum significance
                    targets[i] = 1 if (recent_trend * future_trend < 0) else 0
        
        return targets
    
    def _generate_volume_divergence_target(self, 
                                         features_df: pd.DataFrame, 
                                         horizon: int = 5) -> np.ndarray:
        """
        Generate volume-price divergence validation target
        
        Question: When volume_price_divergence is detected, does it resolve as expected?
        Target: 1 if divergence resolves (price follows volume), 0 otherwise
        """
        
        volume_divergence = features_df['volume_price_divergence'].values
        returns_1d = features_df['returns_1d'].values
        
        targets = np.zeros(len(features_df))
        
        for i in range(len(features_df) - horizon):
            current_divergence = volume_divergence[i]
            
            # Strong divergence signal (top 25%)
            divergence_threshold = np.percentile(np.abs(volume_divergence[~np.isnan(volume_divergence)]), 75)
            
            if abs(current_divergence) > divergence_threshold:
                # Future price movement following volume signal
                future_period = slice(i + 1, i + horizon + 1)
                future_returns = returns_1d[future_period]
                
                if len(future_returns) > 0:
                    future_movement = np.mean(future_returns)
                    
                    # Divergence resolves if future movement is significant (>0.5% magnitude)
                    targets[i] = 1 if abs(future_movement) > 0.005 else 0
        
        return targets
    
    def _generate_combined_pattern_target(self, 
                                        individual_targets: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate combined pattern confidence score
        
        This represents overall pattern detection confidence
        Range: 0.0 to 1.0 based on how many patterns validate
        """
        
        pattern_keys = [k for k in individual_targets.keys() if 'binary' in k]
        
        if len(pattern_keys) == 0:
            return np.zeros(len(individual_targets[list(individual_targets.keys())[0]]))
        
        # Stack all binary targets
        all_patterns = np.column_stack([individual_targets[key] for key in pattern_keys])
        
        # Combined score: proportion of patterns that validate
        combined_score = np.mean(all_patterns, axis=1)
        
        return combined_score
    
    def validate_pattern_targets(self, 
                               targets: Dict[str, np.ndarray],
                               features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate generated pattern targets for quality
        
        Returns:
            Dictionary of validation metrics
        """
        
        validation_results = {}
        
        for target_name, target_values in targets.items():
            if 'binary' in target_name:
                # Binary target validation
                positive_rate = np.mean(target_values)
                validation_results[f"{target_name}_positive_rate"] = positive_rate
                
                # Check for reasonable balance (10-90% range)
                if 0.1 <= positive_rate <= 0.9:
                    validation_results[f"{target_name}_balance"] = "GOOD"
                else:
                    validation_results[f"{target_name}_balance"] = "IMBALANCED"
            
            elif 'score' in target_name:
                # Continuous target validation
                mean_score = np.mean(target_values)
                std_score = np.std(target_values)
                validation_results[f"{target_name}_mean"] = mean_score
                validation_results[f"{target_name}_std"] = std_score
        
        return validation_results


def create_pattern_target_generator(lookback_window: int = 20,
                                  validation_horizons: List[int] = [3, 5, 10]) -> PatternTargetGenerator:
    """Convenience function to create pattern target generator"""
    return PatternTargetGenerator(lookback_window=lookback_window, 
                                validation_horizons=validation_horizons)


# Example usage and testing
if __name__ == "__main__":
    
    # Create synthetic test data with pattern features
    dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="D")
    n_days = len(dates)
    
    # Synthetic pattern features
    np.random.seed(42)
    test_features = pd.DataFrame({
        'momentum_persistence_7d': np.random.randn(n_days) * 0.1,
        'volatility_clustering': np.random.randn(n_days) * 0.2,
        'trend_exhaustion': np.random.randn(n_days) * 0.15,
        'volume_price_divergence': np.random.randn(n_days) * 0.3,
        'volatility_regime_change': np.random.randn(n_days) * 0.25,
        'returns_1d': np.random.randn(n_days) * 0.02,
        'returns_3d': np.random.randn(n_days) * 0.035,
        'returns_7d': np.random.randn(n_days) * 0.05,
        'close': 100 + np.cumsum(np.random.randn(n_days) * 0.01)
    }, index=dates)
    
    # Create pattern target generator
    generator = create_pattern_target_generator(lookback_window=20, validation_horizons=[3, 5, 10])
    
    # Generate pattern targets
    pattern_targets = generator.generate_all_pattern_targets(test_features, primary_horizon=5)
    
    # Validate targets
    validation_results = generator.validate_pattern_targets(pattern_targets, test_features)
    
    print("\\nPattern Target Validation Results:")
    for metric, value in validation_results.items():
        print(f"   {metric}: {value}")
    
    print("\\nPattern target generation validated successfully!")
    print(f"Generated targets for {len(pattern_targets)} pattern types")
    print("Ready for pattern detection LSTM training")