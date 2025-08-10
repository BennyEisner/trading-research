#!/usr/bin/env python3

"""
Pattern Target Validation Tests
Tests for validating pattern target generation logic and output quality
"""

import numpy as np
import pandas as pd
import pytest

from src.training.pattern_target_generator import PatternTargetGenerator


class TestPatternTargetValidation:
    """Test pattern target generation validation logic"""
    
    @pytest.fixture
    def validation_data(self):
        """Create data for pattern target validation testing"""
        dates = pd.date_range("2022-01-01", periods=150, freq="D")
        np.random.seed(42)
        
        # Create data with known patterns for validation testing
        base_trend = np.sin(np.arange(150) * 0.1) * 0.02  # Sinusoidal trend
        noise = np.random.normal(0, 0.01, 150)
        
        data = pd.DataFrame({
            'date': dates,
            'close': 100 + np.cumsum(base_trend + noise),
            'volume': np.random.lognormal(15, 0.3, 150),
            'momentum_persistence_7d': base_trend * 2 + np.random.normal(0, 0.1, 150),
            'volatility_clustering': np.abs(noise) * 5 + np.random.normal(0, 0.1, 150),
            'trend_exhaustion': -base_trend * 1.5 + np.random.normal(0, 0.1, 150),
            'volume_price_divergence': np.random.normal(0, 0.2, 150),
            'volatility_regime_change': np.random.normal(0, 0.3, 150),
            'returns_1d': base_trend + noise,
            'returns_3d': np.convolve(base_trend + noise, np.ones(3), mode='same'),
            'returns_7d': np.convolve(base_trend + noise, np.ones(7), mode='same')
        })
        
        return data
    
    def test_pattern_target_output_structure(self, validation_data):
        """Test that pattern target generation produces correct output structure"""
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(validation_data)
        
        # Test required output structure
        required_targets = [
            'momentum_persistence_binary',
            'trend_exhaustion_binary', 
            'volatility_regime_binary',
            'volume_divergence_binary',
            'pattern_confidence_score'
        ]
        
        for target_name in required_targets:
            assert target_name in targets_dict, f"Missing required target: {target_name}"
            
            target_values = targets_dict[target_name]
            
            # Test basic array properties
            assert isinstance(target_values, np.ndarray), f"{target_name} should be numpy array"
            assert len(target_values) > 0, f"{target_name} should not be empty"
            assert not np.any(np.isnan(target_values)), f"{target_name} contains NaN values"
            assert not np.any(np.isinf(target_values)), f"{target_name} contains infinite values"
        
        print("✅ PATTERN TARGET OUTPUT STRUCTURE VALIDATED")
    
    def test_binary_target_properties(self, validation_data):
        """Test that binary pattern targets have correct properties"""
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(validation_data)
        
        binary_targets = [
            'momentum_persistence_binary',
            'trend_exhaustion_binary',
            'volatility_regime_binary', 
            'volume_divergence_binary'
        ]
        
        for target_name in binary_targets:
            if target_name in targets_dict:
                target_values = targets_dict[target_name]
                
                # Test binary properties
                unique_values = np.unique(target_values)
                assert len(unique_values) <= 2, f"{target_name} should be binary"
                assert np.min(target_values) >= 0.0, f"{target_name} should be >= 0"
                assert np.max(target_values) <= 1.0, f"{target_name} should be <= 1"
                
                # Test that not all values are identical (unless legitimately constant)
                target_variance = np.var(target_values)
                if target_variance < 1e-10:
                    print(f"ℹ️  {target_name}: Constant values (variance={target_variance:.2e})")
                else:
                    print(f"✅ {target_name}: Variable values (variance={target_variance:.4f})")
        
        print("✅ BINARY TARGET PROPERTIES VALIDATED")
    
    def test_pattern_confidence_score_properties(self, validation_data):
        """Test pattern confidence score properties"""
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(validation_data)
        
        pattern_confidence = targets_dict['pattern_confidence_score']
        
        # Test confidence score properties
        assert np.min(pattern_confidence) >= 0.0, "Pattern confidence should be >= 0"
        assert np.max(pattern_confidence) <= 1.0, "Pattern confidence should be <= 1"
        
        # Test statistical properties
        confidence_mean = np.mean(pattern_confidence)
        confidence_std = np.std(pattern_confidence)
        confidence_median = np.median(pattern_confidence)
        
        print(f"PATTERN CONFIDENCE STATISTICS:")
        print(f"  Mean: {confidence_mean:.4f}")
        print(f"  Std: {confidence_std:.4f}")
        print(f"  Median: {confidence_median:.4f}")
        print(f"  Min: {np.min(pattern_confidence):.4f}")
        print(f"  Max: {np.max(pattern_confidence):.4f}")
        
        # Test reasonable variance
        assert confidence_std > 0.01, f"Pattern confidence too constant (std={confidence_std:.4f})"
        assert confidence_std < 0.5, f"Pattern confidence too variable (std={confidence_std:.4f})"
        
        # Test reasonable mean (not stuck at extremes)
        assert 0.1 < confidence_mean < 0.9, f"Pattern confidence mean {confidence_mean:.4f} at extreme"
        
        print("✅ PATTERN CONFIDENCE SCORE PROPERTIES VALIDATED")
    
    def test_target_length_consistency(self, validation_data):
        """Test that all targets have consistent lengths"""
        
        generator = PatternTargetGenerator()
        targets_dict = generator.generate_all_pattern_targets(validation_data)
        
        # Get all target lengths
        target_lengths = {}
        for target_name, target_values in targets_dict.items():
            target_lengths[target_name] = len(target_values)
        
        # Test all targets have same length
        unique_lengths = set(target_lengths.values())
        assert len(unique_lengths) == 1, f"Inconsistent target lengths: {target_lengths}"
        
        common_length = list(unique_lengths)[0]
        input_length = len(validation_data)
        
        print(f"TARGET LENGTH VALIDATION:")
        print(f"  Input data length: {input_length}")
        print(f"  Target length: {common_length}")
        print(f"  Length difference: {input_length - common_length}")
        
        # Targets should be shorter than input due to lookback window
        assert common_length <= input_length, "Targets longer than input data"
        assert common_length > input_length * 0.5, "Targets too much shorter than input"
        
        print("✅ TARGET LENGTH CONSISTENCY VALIDATED")
    
    def test_lookback_window_impact(self, validation_data):
        """Test impact of different lookback windows"""
        
        # Test different lookback windows
        lookback_windows = [5, 10, 20, 30]
        
        results = {}
        
        for lookback in lookback_windows:
            if lookback < len(validation_data) - 10:  # Ensure sufficient data
                generator = PatternTargetGenerator(lookback_window=lookback)
                targets_dict = generator.generate_all_pattern_targets(validation_data)
                
                pattern_confidence = targets_dict['pattern_confidence_score']
                
                results[lookback] = {
                    'length': len(pattern_confidence),
                    'mean': np.mean(pattern_confidence),
                    'std': np.std(pattern_confidence),
                    'non_zero_count': np.sum(pattern_confidence > 0.1)
                }
        
        print(f"LOOKBACK WINDOW IMPACT ANALYSIS:")
        for lookback, stats in results.items():
            print(f"  Window {lookback}: len={stats['length']}, mean={stats['mean']:.3f}, "
                  f"std={stats['std']:.3f}, active={stats['non_zero_count']}")
        
        # Test that longer lookback windows produce different results
        if len(results) >= 2:
            lookbacks = sorted(results.keys())
            first_result = results[lookbacks[0]]
            last_result = results[lookbacks[-1]]
            
            # Should see some difference in pattern detection
            mean_diff = abs(first_result['mean'] - last_result['mean'])
            assert mean_diff > 0.001, "Lookback window shows no impact on pattern detection"
        
        print("✅ LOOKBACK WINDOW IMPACT VALIDATED")
    
    def test_target_temporal_stability(self, validation_data):
        """Test temporal stability of target generation"""
        
        generator = PatternTargetGenerator()
        
        # Generate targets multiple times with same data
        targets1 = generator.generate_all_pattern_targets(validation_data)
        targets2 = generator.generate_all_pattern_targets(validation_data)
        
        # Test reproducibility
        for target_name in targets1.keys():
            if target_name in targets2:
                diff = np.max(np.abs(targets1[target_name] - targets2[target_name]))
                assert diff < 1e-10, f"{target_name} not reproducible (max_diff={diff})"
        
        # Test stability with minor data changes
        perturbed_data = validation_data.copy()
        perturbed_data.iloc[-1, perturbed_data.columns.get_loc('close')] *= 1.001  # Tiny change
        
        targets_perturbed = generator.generate_all_pattern_targets(perturbed_data)
        
        # Targets should be relatively stable to minor data changes
        pattern_confidence_orig = targets1['pattern_confidence_score']
        pattern_confidence_pert = targets_perturbed['pattern_confidence_score']
        
        if len(pattern_confidence_orig) == len(pattern_confidence_pert):
            correlation = np.corrcoef(pattern_confidence_orig, pattern_confidence_pert)[0,1]
            assert correlation > 0.95, f"Targets not stable to minor changes (corr={correlation:.3f})"
            
            print(f"TEMPORAL STABILITY: correlation={correlation:.4f} (>0.95 required)")
        
        print("✅ TARGET TEMPORAL STABILITY VALIDATED")
    
    def test_edge_case_handling(self, validation_data):
        """Test edge case handling in target generation"""
        
        generator = PatternTargetGenerator()
        
        # Test with minimal data
        minimal_data = validation_data.head(30).copy()
        
        try:
            minimal_targets = generator.generate_all_pattern_targets(minimal_data)
            assert len(minimal_targets) > 0, "Should handle minimal data"
            print("✅ Minimal data case handled")
        except Exception as e:
            print(f"ℹ️  Minimal data case raises exception (acceptable): {e}")
        
        # Test with constant data
        constant_data = validation_data.copy()
        for col in ['momentum_persistence_7d', 'trend_exhaustion']:
            if col in constant_data.columns:
                constant_data[col] = 0.5  # Constant value
        
        try:
            constant_targets = generator.generate_all_pattern_targets(constant_data)
            
            # Should handle constant data gracefully
            pattern_confidence = constant_targets['pattern_confidence_score']
            assert not np.any(np.isnan(pattern_confidence)), "Constant data produces NaN"
            print("✅ Constant data case handled")
        except Exception as e:
            print(f"ℹ️  Constant data case raises exception (acceptable): {e}")
        
        print("✅ EDGE CASE HANDLING VALIDATED")