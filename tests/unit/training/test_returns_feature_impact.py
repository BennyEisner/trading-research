#!/usr/bin/env python3

"""
Returns Feature Impact Tests
Controlled experiments testing the impact of returns features on model correlation
"""

import numpy as np
import pandas as pd
import pytest

from src.training.shared_backbone_trainer import SharedBackboneTrainer
from src.training.pattern_target_generator import PatternTargetGenerator


class TestReturnsFeatureImpact:
    """Test the impact of returns features through controlled experiments"""
    
    @pytest.fixture
    def experimental_data(self):
        """Create experimental data for testing returns feature impact"""
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        # Create realistic market data
        base_returns = np.random.normal(0, 0.02, 100)
        cumulative_price = 100 * np.cumprod(1 + base_returns)
        
        data = pd.DataFrame({
            'date': dates,
            'close': cumulative_price,
            'volume': np.random.lognormal(15, 0.5, 100),
            'momentum_persistence_7d': np.random.normal(0, 0.3, 100),
            'volatility_clustering': np.random.normal(0, 0.2, 100),
            'trend_exhaustion': np.random.normal(0, 0.4, 100),
            'volume_price_divergence': np.random.normal(0, 0.2, 100),
            'volatility_regime_change': np.random.normal(0, 0.3, 100),
            'returns_1d': base_returns,
            'returns_3d': np.convolve(base_returns, np.ones(3), mode='same'),
            'returns_7d': np.convolve(base_returns, np.ones(7), mode='same')
        })
        
        return data
    
    def test_returns_features_correlation_impact(self, experimental_data):
        """CONTROLLED EXPERIMENT: Test correlation WITH vs WITHOUT returns features"""
        
        # Define feature sets
        core_features = [
            'momentum_persistence_7d', 
            'trend_exhaustion',
            'volatility_clustering', 
            'volume_price_divergence',
            'volatility_regime_change'
        ]
        
        returns_features = ['returns_1d', 'returns_3d', 'returns_7d']
        
        # Generate pattern targets (same for both experiments)
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(experimental_data)
        pattern_confidence = targets_dict['pattern_confidence_score']
        
        # Experiment 1: WITHOUT returns features
        correlations_without_returns = {}
        
        for feature_name in core_features:
            if feature_name in experimental_data.columns:
                feature_values = experimental_data[feature_name].values
                
                # Calculate correlation
                valid_mask = ~(np.isnan(feature_values) | np.isnan(pattern_confidence))
                if np.sum(valid_mask) > 10:
                    correlation = abs(np.corrcoef(feature_values[valid_mask], 
                                                pattern_confidence[valid_mask])[0,1])
                    correlations_without_returns[feature_name] = correlation
        
        # Experiment 2: WITH returns features
        correlations_with_returns = {}
        
        all_features = core_features + returns_features
        for feature_name in all_features:
            if feature_name in experimental_data.columns:
                feature_values = experimental_data[feature_name].values
                
                # Calculate correlation
                valid_mask = ~(np.isnan(feature_values) | np.isnan(pattern_confidence))
                if np.sum(valid_mask) > 10:
                    correlation = abs(np.corrcoef(feature_values[valid_mask], 
                                                pattern_confidence[valid_mask])[0,1])
                    correlations_with_returns[feature_name] = correlation
        
        print("CONTROLLED EXPERIMENT RESULTS:")
        print(f"WITHOUT returns features:")
        for feature, corr in correlations_without_returns.items():
            print(f"  {feature}: {corr:.3f}")
        
        print(f"WITH returns features:")
        for feature, corr in correlations_with_returns.items():
            print(f"  {feature}: {corr:.3f}")
        
        # Test returns features impact
        returns_correlations = {k: v for k, v in correlations_with_returns.items() 
                               if k in returns_features}
        
        print(f"Returns features correlations:")
        for feature, corr in returns_correlations.items():
            print(f"  {feature}: {corr:.3f}")
            
            # Test that returns correlations are reasonable (not massive leakage)
            assert corr < 0.3, f"Returns feature {feature} correlation {corr:.3f} too high (possible leakage)"
        
        # Calculate impact metrics
        core_corr_mean = np.mean(list(correlations_without_returns.values()))
        returns_corr_mean = np.mean(list(returns_correlations.values()))
        
        print(f"\nIMPACT ANALYSIS:")
        print(f"Core features mean correlation: {core_corr_mean:.3f}")
        print(f"Returns features mean correlation: {returns_corr_mean:.3f}")
        
        if returns_corr_mean > core_corr_mean * 2:
            print(f"⚠️  Returns features show unusually high correlation - investigate further")
        else:
            print(f"✅ Returns features correlation within reasonable bounds")
    
    def test_sequence_generation_with_without_returns(self, experimental_data):
        """Test sequence generation performance WITH vs WITHOUT returns features"""
        
        trainer = SharedBackboneTrainer(tickers=["TEST"], use_expanded_universe=False)
        
        # Generate pattern targets
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(experimental_data)
        pattern_targets = targets_dict['pattern_confidence_score']
        
        # Test WITHOUT returns features
        core_features = [
            'momentum_persistence_7d', 
            'trend_exhaustion',
            'volatility_clustering', 
            'volume_price_divergence',
            'volatility_regime_change'
        ]
        
        X_without, y_without = trainer._prepare_pattern_detection_sequences(
            features_df=experimental_data,
            feature_columns=core_features,
            pattern_targets=pattern_targets,
            sequence_length=20,
            stride=1
        )
        
        # Test WITH returns features  
        all_features = core_features + ['returns_1d', 'returns_3d', 'returns_7d']
        
        X_with, y_with = trainer._prepare_pattern_detection_sequences(
            features_df=experimental_data,
            feature_columns=all_features,
            pattern_targets=pattern_targets,
            sequence_length=20,
            stride=1
        )
        
        print(f"SEQUENCE GENERATION EXPERIMENT:")
        print(f"Without returns: {len(X_without)} sequences, {X_without.shape[2]} features")
        print(f"With returns: {len(X_with)} sequences, {X_with.shape[2]} features")
        
        # Validate both experiments generated reasonable data
        assert len(X_without) > 0, "Failed to generate sequences without returns features"
        assert len(X_with) > 0, "Failed to generate sequences with returns features"
        assert len(X_without) == len(X_with), "Sequence count should be same for both experiments"
        
        # Feature count difference should be exactly 3 (the returns features)
        feature_diff = X_with.shape[2] - X_without.shape[2]
        assert feature_diff == 3, f"Expected 3 additional features, got {feature_diff}"
        
        print("✅ SEQUENCE GENERATION EXPERIMENT PASSED")
    
    def test_returns_features_predictive_power_isolation(self, experimental_data):
        """Test isolated predictive power of returns features"""
        
        # Generate targets
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(experimental_data)
        pattern_confidence = targets_dict['pattern_confidence_score']
        
        # Test each returns feature in isolation
        returns_features = ['returns_1d', 'returns_3d', 'returns_7d']
        
        isolated_correlations = {}
        
        for returns_feature in returns_features:
            feature_values = experimental_data[returns_feature].values
            
            # Calculate isolated correlation
            valid_mask = ~(np.isnan(feature_values) | np.isnan(pattern_confidence))
            if np.sum(valid_mask) > 10:
                correlation = abs(np.corrcoef(feature_values[valid_mask], 
                                            pattern_confidence[valid_mask])[0,1])
                isolated_correlations[returns_feature] = correlation
        
        print(f"ISOLATED RETURNS FEATURE ANALYSIS:")
        for feature, corr in isolated_correlations.items():
            print(f"{feature}: {corr:.3f} isolated correlation")
            
            # Test reasonable bounds
            assert corr < 0.4, f"{feature} isolated correlation {corr:.3f} too high"
        
        # Test that returns features aren't dramatically more predictive than core features
        max_returns_corr = max(isolated_correlations.values())
        
        # Compare to core feature (momentum_persistence_7d)
        momentum_values = experimental_data['momentum_persistence_7d'].values
        valid_mask = ~(np.isnan(momentum_values) | np.isnan(pattern_confidence))
        if np.sum(valid_mask) > 10:
            momentum_correlation = abs(np.corrcoef(momentum_values[valid_mask], 
                                                 pattern_confidence[valid_mask])[0,1])
            
            print(f"Core feature comparison - momentum_persistence_7d: {momentum_correlation:.3f}")
            
            # Returns shouldn't be dramatically more predictive than core features
            if max_returns_corr > momentum_correlation * 3:
                print(f"⚠️  Returns features show unusually high isolated predictive power")
            else:
                print(f"✅ Returns features isolated predictive power within reasonable bounds")
        
        print("✅ ISOLATED RETURNS FEATURE ANALYSIS PASSED")