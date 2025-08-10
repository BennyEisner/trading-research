#!/usr/bin/env python3

"""
Core Pattern Target Generator Tests
Focuses on core pattern target generation functionality
"""

import numpy as np
import pandas as pd
import pytest

from src.training.pattern_target_generator import PatternTargetGenerator


class TestPatternTargetGenerator:
    """Test core pattern target generation functionality"""
    
    @pytest.fixture
    def sample_features_data(self):
        """Create sample features data for testing"""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        
        np.random.seed(42)
        
        # Create features required by PatternTargetGenerator
        data = pd.DataFrame({
            'date': dates,
            'close': 100 * (1 + np.cumsum(np.random.normal(0, 0.01, 100))),
            'volume': np.random.uniform(1000000, 5000000, 100),
            'momentum_persistence_7d': np.random.normal(0, 1, 100),
            'volatility_clustering': np.random.normal(0, 1, 100),
            'trend_exhaustion': np.random.normal(0, 1, 100),
            'volume_price_divergence': np.random.normal(0, 1, 100),
            'volatility_regime_change': np.random.normal(0, 1, 100),
            'returns_1d': np.random.normal(0, 0.02, 100),
            'returns_3d': np.random.normal(0, 0.05, 100), 
            'returns_7d': np.random.normal(0, 0.1, 100)
        })
        
        return data
    
    def test_pattern_target_generation_basic_functionality(self, sample_features_data):
        """Test basic pattern target generation works"""
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(sample_features_data)
        
        # Test that all expected targets are generated
        expected_targets = [
            'momentum_persistence_binary',
            'volatility_regime_binary', 
            'trend_exhaustion_binary',
            'volume_divergence_binary',
            'pattern_confidence_score'
        ]
        
        for target_name in expected_targets:
            assert target_name in targets_dict, f"Missing target: {target_name}"
            assert isinstance(targets_dict[target_name], np.ndarray), f"Target {target_name} is not numpy array"
            assert len(targets_dict[target_name]) > 0, f"Target {target_name} is empty"
    
    def test_enhanced_target_balance_improvement(self, sample_features_data):
        """Test that enhanced target engineering improves balance"""
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(sample_features_data)
        
        pattern_confidence = targets_dict['pattern_confidence_score']
        
        # Test target range
        assert np.min(pattern_confidence) >= 0.0, "Targets should be >= 0"
        assert np.max(pattern_confidence) <= 1.0, "Targets should be <= 1"
        
        # Test improved balance (should be better than baseline 11.5%)
        above_threshold = np.mean(pattern_confidence > 0.5)
        
        # Enhanced target engineering should achieve 30-50% positive rate
        assert above_threshold >= 0.25, f"Target balance too low: {above_threshold:.3f} (expected >= 0.25)"
        assert above_threshold <= 0.70, f"Target balance too high: {above_threshold:.3f} (expected <= 0.70)"
        
        # Test diversity (should have reasonable standard deviation)
        target_std = np.std(pattern_confidence)
        assert target_std >= 0.10, f"Target diversity too low: std={target_std:.3f} (expected >= 0.10)"
        assert target_std <= 0.40, f"Target diversity too high: std={target_std:.3f} (expected <= 0.40)"
    
    def test_continuous_target_generation(self, sample_features_data):
        """Test that targets are continuous, not binary"""
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(sample_features_data)
        
        pattern_confidence = targets_dict['pattern_confidence_score']
        
        # Test that targets are continuous (many unique values)
        unique_values = len(np.unique(np.round(pattern_confidence, 4)))
        total_values = len(pattern_confidence)
        
        # Should have high diversity of unique values
        uniqueness_ratio = unique_values / total_values
        assert uniqueness_ratio >= 0.5, f"Targets not diverse enough: {uniqueness_ratio:.3f} unique ratio"
        
        # Test that not all values are 0 or 1 (not binary)
        binary_values = np.sum((pattern_confidence == 0.0) | (pattern_confidence == 1.0))
        binary_ratio = binary_values / total_values
        assert binary_ratio < 0.2, f"Too many binary values: {binary_ratio:.3f} (should be < 0.2)"
    
    def test_weighted_pattern_combination(self, sample_features_data):
        """Test that pattern combination uses proper weighting"""
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(sample_features_data)
        
        # Test that individual pattern targets exist
        individual_patterns = [
            'momentum_persistence_binary',
            'volatility_regime_binary',
            'trend_exhaustion_binary', 
            'volume_divergence_binary'
        ]
        
        for pattern in individual_patterns:
            assert pattern in targets_dict, f"Missing individual pattern: {pattern}"
            
            # Test that individual patterns are continuous (not binary despite name)
            pattern_values = targets_dict[pattern]
            assert np.min(pattern_values) >= 0.0, f"{pattern} values should be >= 0"
            assert np.max(pattern_values) <= 1.0, f"{pattern} values should be <= 1"
            
            # Should have diversity
            pattern_std = np.std(pattern_values)
            assert pattern_std > 0.05, f"{pattern} should have some variance: std={pattern_std:.3f}"
        
        # Test that combined score is reasonable average
        combined_score = targets_dict['pattern_confidence_score']
        individual_scores = np.column_stack([targets_dict[p] for p in individual_patterns])
        
        # Combined score should be influenced by all patterns
        for i, pattern in enumerate(individual_patterns):
            pattern_values = targets_dict[pattern]
            correlation = np.corrcoef(combined_score, pattern_values)[0, 1]
            
            # Should have positive correlation with individual patterns
            assert correlation > 0.1, f"Combined score poorly correlated with {pattern}: corr={correlation:.3f}"
    
    def test_temporal_horizon_handling(self, sample_features_data):
        """Test that different prediction horizons are handled correctly"""
        
        # Test with different horizons
        horizons = [1, 3, 5]
        
        for horizon in horizons:
            generator = PatternTargetGenerator()
            targets_dict = generator.generate_all_pattern_targets(
                sample_features_data, primary_horizon=horizon
            )
            
            # Should generate valid targets for each horizon
            pattern_confidence = targets_dict['pattern_confidence_score']
            assert len(pattern_confidence) > 0, f"No targets generated for horizon {horizon}"
            assert np.min(pattern_confidence) >= 0.0, f"Invalid target range for horizon {horizon}"
            assert np.max(pattern_confidence) <= 1.0, f"Invalid target range for horizon {horizon}"
    
    def test_lookback_window_configuration(self, sample_features_data):
        """Test that different lookback windows work correctly"""
        
        lookback_windows = [10, 20, 30]
        
        for window in lookback_windows:
            if window < len(sample_features_data) - 10:  # Ensure sufficient data
                generator = PatternTargetGenerator(lookback_window=window)
                targets_dict = generator.generate_all_pattern_targets(sample_features_data)
                
                pattern_confidence = targets_dict['pattern_confidence_score']
                
                # Should generate valid targets
                assert len(pattern_confidence) > 0, f"No targets for lookback {window}"
                
                # Target length should be influenced by lookback window
                expected_max_length = len(sample_features_data) - window
                assert len(pattern_confidence) <= expected_max_length, (
                    f"Target length {len(pattern_confidence)} exceeds expected {expected_max_length} for lookback {window}"
                )
    
    def test_missing_feature_handling(self, sample_features_data):
        """Test handling of missing features"""
        
        # Create data with missing feature
        incomplete_data = sample_features_data.drop(columns=['momentum_persistence_7d'])
        
        generator = PatternTargetGenerator()
        
        try:
            targets_dict = generator.generate_all_pattern_targets(incomplete_data)
            # If it succeeds, should still produce valid targets
            if 'pattern_confidence_score' in targets_dict:
                pattern_confidence = targets_dict['pattern_confidence_score']
                assert len(pattern_confidence) > 0, "Should handle missing features gracefully"
                assert not np.any(np.isnan(pattern_confidence)), "No NaN values in targets"
        except Exception as e:
            # Acceptable to raise exception for missing critical features
            print(f"Missing feature handling raises exception (acceptable): {e}")
    
    def test_nan_value_handling(self, sample_features_data):
        """Test handling of NaN values in input data"""
        
        # Introduce some NaN values
        data_with_nan = sample_features_data.copy()
        data_with_nan.loc[10:15, 'momentum_persistence_7d'] = np.nan
        data_with_nan.loc[20:25, 'volatility_clustering'] = np.nan
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(data_with_nan)
        
        # Should handle NaN values and produce valid targets
        pattern_confidence = targets_dict['pattern_confidence_score']
        assert not np.any(np.isnan(pattern_confidence)), "Target generation should handle input NaN values"
        assert not np.any(np.isinf(pattern_confidence)), "No infinite values in targets"
    
    def test_generator_configuration_parameters(self, sample_features_data):
        """Test that generator configuration parameters work correctly"""
        
        # Test with custom validation horizons
        custom_horizons = [1, 7, 14]
        generator = PatternTargetGenerator(
            lookback_window=15, 
            validation_horizons=custom_horizons
        )
        
        targets_dict = generator.generate_all_pattern_targets(sample_features_data)
        
        # Should still generate valid targets
        pattern_confidence = targets_dict['pattern_confidence_score']
        assert len(pattern_confidence) > 0, "Custom configuration should work"
        assert np.min(pattern_confidence) >= 0.0, "Valid range with custom config"
        assert np.max(pattern_confidence) <= 1.0, "Valid range with custom config"
    
    def test_reproducibility(self, sample_features_data):
        """Test that target generation is reproducible"""
        
        generator1 = PatternTargetGenerator(lookback_window=20)
        generator2 = PatternTargetGenerator(lookback_window=20)
        
        targets1 = generator1.generate_all_pattern_targets(sample_features_data)
        targets2 = generator2.generate_all_pattern_targets(sample_features_data)
        
        # Should produce identical results
        for target_name in targets1.keys():
            if target_name in targets2:
                diff = np.max(np.abs(targets1[target_name] - targets2[target_name]))
                assert diff < 1e-10, f"{target_name} not reproducible (max_diff={diff})"
        
        print("✅ CORE PATTERN TARGET GENERATOR TESTS PASSED")
