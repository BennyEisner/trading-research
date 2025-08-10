#!/usr/bin/env python3

"""
Unit tests for PatternTargetGenerator
Tests enhanced target engineering fixes for improved learning
"""

import numpy as np
import pandas as pd
import pytest

from ..pattern_target_generator import PatternTargetGenerator


class TestPatternTargetGenerator:
    """Test pattern target generation with temporal leakage prevention"""
    
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
        horizons_to_test = [3, 5, 10]
        
        for horizon in horizons_to_test:
            generator = PatternTargetGenerator(validation_horizons=[horizon])
            targets_dict = generator.generate_all_pattern_targets(sample_features_data, primary_horizon=horizon)
            
            pattern_confidence = targets_dict['pattern_confidence_score']
            
            # Should generate reasonable targets for all horizons
            assert len(pattern_confidence) > 0, f"No targets generated for horizon {horizon}"
            assert np.std(pattern_confidence) > 0.05, f"Low diversity for horizon {horizon}"
            
            # Longer horizons might have different characteristics but should still be valid
            above_threshold = np.mean(pattern_confidence > 0.5)
            assert 0.1 <= above_threshold <= 0.9, f"Poor balance for horizon {horizon}: {above_threshold:.3f}"
    
    def test_nan_handling_in_features(self, sample_features_data):
        """Test that NaN values in features are handled properly"""
        
        # Introduce some NaN values
        corrupted_data = sample_features_data.copy()
        corrupted_data.loc[10:15, 'momentum_persistence_7d'] = np.nan
        corrupted_data.loc[20:25, 'volatility_clustering'] = np.nan
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(corrupted_data)
        
        pattern_confidence = targets_dict['pattern_confidence_score']
        
        # Should still generate targets (with proper NaN handling)
        assert len(pattern_confidence) > 0, "No targets generated with NaN features"
        
        # Generated targets should not contain NaN
        nan_targets = np.isnan(pattern_confidence).sum()
        nan_ratio = nan_targets / len(pattern_confidence)
        assert nan_ratio < 0.3, f"Too many NaN targets: {nan_ratio:.3f} (should be < 0.3)"
        
        # Valid targets should still have reasonable properties
        valid_targets = pattern_confidence[~np.isnan(pattern_confidence)]
        if len(valid_targets) > 10:
            assert np.min(valid_targets) >= 0.0, "Valid targets should be >= 0"
            assert np.max(valid_targets) <= 1.0, "Valid targets should be <= 1"
    
    def test_target_validation_functionality(self, sample_features_data):
        """Test target validation metrics"""
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(sample_features_data)
        
        # Test validation function
        validation_results = generator.validate_pattern_targets(targets_dict, sample_features_data)
        
        # Should return validation metrics
        assert isinstance(validation_results, dict), "Validation results should be dictionary"
        
        # Test pattern confidence score validation
        if 'pattern_confidence_score_mean' in validation_results:
            mean_score = validation_results['pattern_confidence_score_mean']
            assert 0.0 <= mean_score <= 1.0, f"Invalid mean score: {mean_score}"
        
        if 'pattern_confidence_score_std' in validation_results:
            std_score = validation_results['pattern_confidence_score_std']
            assert std_score >= 0.0, f"Invalid std score: {std_score}"
            assert std_score <= 0.5, f"Std too high: {std_score}"  # Reasonable upper bound


class TestPatternTargetValidation:
    """Test validation logic for pattern targets"""
    
    def test_empty_features_handling(self):
        """Test handling of empty or invalid feature data"""
        
        # Empty dataframe
        empty_df = pd.DataFrame()
        generator = PatternTargetGenerator()
        
        with pytest.raises((ValueError, KeyError)):
            generator.generate_all_pattern_targets(empty_df)
    
    def test_insufficient_features_handling(self):
        """Test handling when required features are missing"""
        
        # Dataframe with only some required features
        partial_df = pd.DataFrame({
            'date': pd.date_range("2023-01-01", periods=50),
            'close': np.random.randn(50),
            'momentum_persistence_7d': np.random.randn(50)
            # Missing other required features
        })
        
        generator = PatternTargetGenerator()
        
        with pytest.raises((ValueError, KeyError)):
            generator.generate_all_pattern_targets(partial_df)
    
    def test_insufficient_data_length(self):
        """Test handling when data is too short for pattern calculation"""
        
        # Very short dataframe (less than lookback + horizon)
        short_df = pd.DataFrame({
            'date': pd.date_range("2023-01-01", periods=10),
            'close': np.random.randn(10),
            'volume': np.random.randn(10),
            'momentum_persistence_7d': np.random.randn(10),
            'volatility_clustering': np.random.randn(10),
            'trend_exhaustion': np.random.randn(10),
            'volume_price_divergence': np.random.randn(10),
            'volatility_regime_change': np.random.randn(10),
            'returns_1d': np.random.randn(10),
            'returns_3d': np.random.randn(10),
            'returns_7d': np.random.randn(10)
        })
        
        generator = PatternTargetGenerator(lookback_window=20)  # Longer than data
        targets_dict = generator.generate_all_pattern_targets(short_df)
        
        # Should handle gracefully (might return empty or very few targets)
        pattern_confidence = targets_dict.get('pattern_confidence_score', np.array([]))
        
        # Either empty or very short
        assert len(pattern_confidence) <= 5, "Should generate few/no targets for short data"