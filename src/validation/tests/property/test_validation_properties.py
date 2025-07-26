#!/usr/bin/env python3

"""
Property-based tests for validation framework
Comprehensive testing with Hypothesis - assumes modern installation
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from hypothesis import assume, given, note, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.extra.pandas import columns, data_frames

from validation.enhanced_robustness_tests import create_robustness_tester
from validation.gapped_time_series_cv import create_gapped_cv
from validation.pipeline_validator import create_pipeline_validator
from validation.robust_time_series_validator import create_robust_validator


@st.composite
def financial_returns(draw, min_size=30, max_size=500):
    """Generate realistic financial return series"""
    size = draw(st.integers(min_size, max_size))

    returns = draw(
        arrays(
            np.float64,
            shape=size,
            elements=st.floats(
                min_value=-0.15,  # Max 15% daily loss
                max_value=0.15,  # Max 15% daily gain
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    assume(np.std(returns) > 1e-6)
    return returns


@st.composite
def correlated_predictions(draw, actual_returns, min_correlation=0.1, max_correlation=0.95):
    """Generate predictions with specified correlation to actual returns"""
    correlation = draw(st.floats(min_correlation, max_correlation))
    noise_level = draw(st.floats(0.001, 0.05))

    noise = draw(
        arrays(
            np.float64,
            shape=len(actual_returns),
            elements=st.floats(-0.05, 0.05, allow_nan=False, allow_infinity=False),
        )
    )

    predictions = actual_returns * correlation + noise * noise_level
    return predictions, correlation


@st.composite
def ohlcv_data(draw, min_periods=50, max_periods=300):
    """Generate realistic OHLCV data"""
    n_periods = draw(st.integers(min_periods, max_periods))

    # Start with base price
    base_price = draw(st.floats(50, 500))

    # Generate price changes
    price_changes = draw(
        arrays(np.float64, shape=n_periods, elements=st.floats(-0.1, 0.1, allow_nan=False, allow_infinity=False))
    )

    # Create price series
    prices = [base_price]
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Ensure positive prices

    prices = np.array(prices)

    # Generate OHLC with proper relationships
    opens = prices * draw(arrays(np.float64, n_periods, elements=st.floats(0.995, 1.005)))

    highs = np.maximum(opens, prices) * draw(arrays(np.float64, n_periods, elements=st.floats(1.0, 1.02)))

    lows = np.minimum(opens, prices) * draw(arrays(np.float64, n_periods, elements=st.floats(0.98, 1.0)))

    volumes = draw(arrays(np.int64, n_periods, elements=st.integers(100000, 50000000)))

    dates = pd.date_range(start="2020-01-01", periods=n_periods, freq="D")

    return pd.DataFrame({"date": dates, "open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes})


class TestValidationInvariants(unittest.TestCase):
    """Test mathematical invariants that must always hold"""

    @given(returns=financial_returns(), correlated_data=st.data())
    @settings(max_examples=50, deadline=3000)
    def test_directional_accuracy_bounds(self, returns, correlated_data):
        """Property: Directional accuracy must always be between 0 and 1"""
        predictions, correlation = correlated_data.draw(correlated_predictions(returns))

        tester = create_robustness_tester()
        accuracy = tester._directional_accuracy(returns, predictions)

        note(f"Testing with {len(returns)} samples, correlation={correlation:.3f}")

        # Core invariant: accuracy must be valid probability
        assert 0.0 <= accuracy <= 1.0
        assert not np.isnan(accuracy)
        assert isinstance(accuracy, (float, np.floating))

    @given(financial_returns(), st.floats(0.7, 0.95), st.floats(0.001, 0.02))  # High correlation  # Low noise
    @settings(max_examples=30, deadline=4000)
    def test_high_correlation_high_accuracy(self, returns, correlation, noise_level):
        """Property: High correlation should yield high directional accuracy"""
        np.random.seed(42)  # For reproducible noise
        noise = np.random.normal(0, noise_level, len(returns))
        predictions = returns * correlation + noise

        tester = create_robustness_tester()
        accuracy = tester._directional_accuracy(returns, predictions)

        note(f"Correlation: {correlation:.3f}, Noise: {noise_level:.4f}, Accuracy: {accuracy:.3f}")

        # Property: Strong correlation should yield high accuracy
        expected_min_accuracy = max(0.55, correlation * 0.7)  # Reasonable expectation
        assert (
            accuracy >= expected_min_accuracy
        ), f"Expected accuracy >= {expected_min_accuracy:.3f} for correlation {correlation:.3f}, got {accuracy:.3f}"

    @given(financial_returns())
    @settings(max_examples=20, deadline=2000)
    def test_random_predictions_near_fifty_percent(self, actual_returns):
        """Property: Random predictions should have ~50% directional accuracy"""
        np.random.seed(42)
        random_predictions = np.random.normal(0, np.std(actual_returns), len(actual_returns))

        tester = create_robustness_tester()
        accuracy = tester._directional_accuracy(actual_returns, random_predictions)

        note(f"Random accuracy: {accuracy:.3f} for {len(actual_returns)} samples")

        # Property: Random should be around 50% (±20% tolerance for finite samples)
        assert 0.3 <= accuracy <= 0.7, f"Random accuracy {accuracy:.3f} outside expected range [0.3, 0.7]"

    @given(st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=15))
    @settings(max_examples=25)
    def test_multiple_testing_correction_invariants(self, p_values):
        """Property: Multiple testing correction maintains statistical validity"""
        assume(all(0.0 <= p <= 1.0 for p in p_values))

        tester = create_robustness_tester()
        results = tester.multiple_testing_correction(p_values)

        corrected_p = results["corrected_p_values"]

        note(f"Testing {len(p_values)} p-values: {p_values[:3]}...")

        # Core invariants
        assert len(corrected_p) == len(p_values)
        assert all(0.0 <= p <= 1.0 for p in corrected_p)
        assert all(not np.isnan(p) for p in corrected_p)

        # Correction should be more conservative
        original_sig = results["n_significant_original"]
        corrected_sig = results["n_significant_corrected"]
        assert corrected_sig <= original_sig

        # If all p-values are very small, at least some should remain significant
        if all(p < 0.001 for p in p_values):
            assert corrected_sig > 0, "Very small p-values should survive correction"


class TestBootstrapProperties(unittest.TestCase):
    """Test bootstrap statistical properties"""

    @given(returns=financial_returns(min_size=50, max_size=200), block_size=st.integers(2, 15), correlated_data=st.data())
    @settings(max_examples=20, deadline=5000)
    def test_bootstrap_statistical_validity(self, returns, block_size, correlated_data):
        """Property: Bootstrap should produce valid statistical results"""
        assume(block_size < len(returns) // 3)  # Reasonable block size

        predictions, correlation = correlated_data.draw(correlated_predictions(returns, 0.3, 0.8))

        validator = create_robust_validator(
            block_size=block_size, n_bootstrap=50, random_state=42  # Reasonable for testing
        )

        results = validator.moving_block_bootstrap(returns, predictions)

        note(f"Bootstrap with {len(returns)} samples, block_size={block_size}, correlation={correlation:.3f}")

        # Statistical validity properties
        assert isinstance(results["p_value"], (float, np.floating))
        assert 0.0 <= results["p_value"] <= 1.0
        assert not np.isnan(results["p_value"])

        assert isinstance(results["actual_performance"], (float, np.floating))
        assert 0.0 <= results["actual_performance"] <= 1.0

        # Bootstrap distribution properties
        bootstrap_dist = results["bootstrap_distribution"]
        assert len(bootstrap_dist) > 0
        assert all(0.0 <= perf <= 1.0 for perf in bootstrap_dist)
        assert not np.isnan(bootstrap_dist).any()

        # Statistical consistency
        bootstrap_mean = np.mean(bootstrap_dist)
        assert abs(results["bootstrap_mean"] - bootstrap_mean) < 1e-10

    @given(returns=financial_returns(min_size=80, max_size=150), correlated_data=st.data())
    @settings(max_examples=15, deadline=6000)
    def test_bootstrap_significance_detection(self, returns, correlated_data):
        """Property: Bootstrap should distinguish good from random predictions"""
        # Test with good predictions
        good_predictions, _ = correlated_data.draw(correlated_predictions(returns, 0.7, 0.9))

        # Test with random predictions
        np.random.seed(42)
        random_predictions = np.random.normal(0, np.std(returns), len(returns))

        validator = create_robust_validator(n_bootstrap=40, random_state=42)

        good_results = validator.moving_block_bootstrap(returns, good_predictions)
        random_results = validator.moving_block_bootstrap(returns, random_predictions)

        note(f"Good p-value: {good_results['p_value']:.4f}, Random p-value: {random_results['p_value']:.4f}")

        # Property: Good predictions should generally have lower p-values
        # (Not always true due to randomness, but should be true more often than not)
        if good_results["actual_performance"] > random_results["actual_performance"] + 0.1:
            assert (
                good_results["p_value"] <= random_results["p_value"] + 0.1
            ), "Significantly better predictions should tend to have lower p-values"

    @given(returns=financial_returns(min_size=60, max_size=120), block_size=st.integers(3, 10), correlated_data=st.data())
    @settings(max_examples=12, deadline=4000)
    def test_bootstrap_determinism(self, returns, block_size, correlated_data):
        """Property: Bootstrap should be deterministic with same random seed"""
        assume(block_size < len(returns) // 4)

        predictions, _ = correlated_data.draw(correlated_predictions(returns))

        # Run twice with same seed
        validator1 = create_robust_validator(block_size=block_size, n_bootstrap=30, random_state=123)
        validator2 = create_robust_validator(block_size=block_size, n_bootstrap=30, random_state=123)

        results1 = validator1.moving_block_bootstrap(returns, predictions)
        results2 = validator2.moving_block_bootstrap(returns, predictions)

        note(f"Determinism test with {len(returns)} samples, block_size={block_size}")

        # Should be exactly identical
        assert abs(results1["p_value"] - results2["p_value"]) < 1e-15
        np.testing.assert_array_equal(results1["bootstrap_distribution"], results2["bootstrap_distribution"])


class TestCrossValidationProperties(unittest.TestCase):
    """Test cross-validation temporal properties"""

    @given(
        st.integers(2, 6),  # n_splits
        st.floats(0.15, 0.35),  # test_size
        st.integers(2, 12),  # gap_size
        st.integers(150, 400),
    )  # n_samples
    @settings(max_examples=25, deadline=3000)
    def test_cv_temporal_properties(self, n_splits, test_size, gap_size, n_samples):
        """Property: CV must maintain temporal order and gap constraints"""
        cv = create_gapped_cv(n_splits=n_splits, test_size=test_size, gap_size=gap_size, expanding_window=True)

        X = np.random.random((n_samples, 5))
        splits = list(cv.split(X))

        note(f"CV: {len(splits)} splits from {n_samples} samples, gap={gap_size}")

        # Basic properties
        assert len(splits) <= n_splits  # May be fewer due to size constraints

        for i, (train_idx, test_idx) in enumerate(splits):
            # Each split must be valid
            assert len(train_idx) > 0, f"Split {i}: empty training set"
            assert len(test_idx) > 0, f"Split {i}: empty test set"

            # No temporal overlap
            assert len(set(train_idx) & set(test_idx)) == 0, f"Split {i}: train/test overlap"

            # Temporal ordering
            max_train = max(train_idx)
            min_test = min(test_idx)
            assert max_train < min_test, f"Split {i}: train must come before test"

            # Gap enforcement
            actual_gap = min_test - max_train - 1
            assert actual_gap >= gap_size, f"Split {i}: gap {actual_gap} < required {gap_size}"

    @given(st.integers(150, 350), st.integers(2, 5))  # n_samples  # n_splits
    @settings(max_examples=15, deadline=2000)
    def test_expanding_window_property(self, n_samples, n_splits):
        """Property: Expanding window should have non-decreasing train sizes"""
        cv = create_gapped_cv(n_splits=n_splits, expanding_window=True, test_size=0.2, gap_size=5)

        X = np.random.random((n_samples, 8))
        splits = list(cv.split(X))

        if len(splits) > 1:
            train_sizes = [len(train_idx) for train_idx, _ in splits]

            note(f"Expanding window train sizes: {train_sizes}")

            # Property: sizes should be non-decreasing
            for i in range(1, len(train_sizes)):
                assert train_sizes[i] >= train_sizes[i - 1], f"Expanding window violated: sizes {train_sizes}"

    @given(st.integers(100, 300))
    @settings(max_examples=10, deadline=2000)
    def test_cv_no_data_leakage(self, n_samples):
        """Property: CV should never allow data leakage"""
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

        cv = create_gapped_cv(n_splits=3, gap_size=7, test_size=0.25)

        # Should pass leakage validation
        no_leakage = cv.validate_no_leakage(dates)

        note(f"Leakage test with {n_samples} samples: {'PASS' if no_leakage else 'FAIL'}")

        # Property: properly configured CV should never leak
        assert no_leakage, "Gapped CV should prevent data leakage"


class TestPipelineValidationProperties(unittest.TestCase):
    """Test pipeline validation robustness"""

    @given(ohlcv_data())
    @settings(max_examples=20, deadline=4000)
    def test_ohlcv_validation_consistency(self, ohlcv_df):
        """Property: Clean OHLCV data should pass validation consistently"""
        validator = create_pipeline_validator()

        is_valid, issues = validator.validate_raw_data(ohlcv_df, "TEST")

        note(f"OHLCV validation: {len(ohlcv_df)} records, valid={is_valid}, issues={len(issues)}")

        # Property: well-formed OHLCV should generally pass
        if is_valid:
            assert len(issues) == 0
        else:
            assert len(issues) > 0
            assert isinstance(issues[0], str)

    @given(st.integers(50, 200), st.integers(5, 20), st.floats(0.0, 0.5))  # n_samples  # n_features  # nan_fraction
    @settings(max_examples=15, deadline=3000)
    def test_feature_validation_nan_detection(self, n_samples, n_features, nan_fraction):
        """Property: High NaN fractions should be reliably detected"""
        np.random.seed(42)

        feature_data = {}
        for i in range(n_features):
            data = np.random.normal(i, 1, n_samples)

            # Introduce NaNs
            n_nans = int(nan_fraction * n_samples)
            if n_nans > 0:
                nan_indices = np.random.choice(n_samples, n_nans, replace=False)
                data[nan_indices] = np.nan

            feature_data[f"feature_{i}"] = data

        df = pd.DataFrame(feature_data)

        validator = create_pipeline_validator(high_nan_threshold=0.15)  # 15% threshold
        is_valid, issues = validator.validate_feature_data(df, list(df.columns))

        note(f"Feature validation: {nan_fraction:.2f} NaN fraction, valid={is_valid}")

        if nan_fraction > 0.2:  # Well above threshold
            assert not is_valid, f"Should detect {nan_fraction:.2f} NaN fraction"
            assert any("NaN" in issue for issue in issues)
        elif nan_fraction < 0.1:  # Well below threshold
            if not is_valid:
                nan_issues = [issue for issue in issues if "NaN" in issue]
                assert len(nan_issues) == 0, f"Shouldn't detect {nan_fraction:.2f} NaN fraction"

    @given(st.integers(100, 300), st.integers(60, 90), st.integers(5, 15))  # n_samples  # sequence_length  # n_features
    @settings(max_examples=12, deadline=3000)
    def test_sequence_validation_properties(self, n_samples, sequence_length, n_features):
        """Property: Valid 3D sequences should pass validation"""
        assume(sequence_length < n_samples)

        np.random.seed(42)
        X = np.random.normal(0, 1, (n_samples, sequence_length, n_features))
        y = np.random.normal(0, 0.02, n_samples)

        validator = create_pipeline_validator()
        is_valid, issues = validator.validate_sequences(X, y)

        note(f"Sequence validation: shape {X.shape}, valid={is_valid}")

        # Property: clean synthetic sequences should pass
        assert is_valid, f"Clean sequences should pass validation: {issues}"
        assert len(issues) == 0


class TestValidationRobustness(unittest.TestCase):
    """Test validation robustness across edge cases"""

    @given(returns=financial_returns(min_size=30, max_size=100), correlated_data=st.data())
    @settings(max_examples=20, deadline=3000)
    def test_validation_never_crashes(self, returns, correlated_data):
        """Property: Validation should handle any input without crashing"""
        predictions, _ = correlated_data.draw(correlated_predictions(returns))

        # Test all major validation components
        try:
            # Robust validator
            validator = create_robust_validator(n_bootstrap=20, random_state=42)
            bootstrap_results = validator.moving_block_bootstrap(returns, predictions)

            assert "p_value" in bootstrap_results
            assert isinstance(bootstrap_results["p_value"], (float, np.floating))

            # Pipeline validator for prediction validation
            pipeline_validator = create_pipeline_validator()
            pred_valid, pred_issues = pipeline_validator.validate_model_predictions(returns, predictions)

            assert isinstance(pred_valid, bool)
            assert isinstance(pred_issues, list)

            note(f"Robustness test passed for {len(returns)} samples")

        except Exception as e:
            self.fail(f"Validation crashed with data shape {returns.shape}: {e}")

    @given(st.integers(5, 25), st.floats(0.1, 0.9))  # Very small datasets  # Various correlations
    @settings(max_examples=15, deadline=2000)
    def test_small_dataset_handling(self, n_samples, correlation):
        """Property: Validation should handle small datasets gracefully"""
        assume(n_samples >= 5)

        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n_samples)
        predictions = returns * correlation + np.random.normal(0, 0.005, n_samples)

        tester = create_robustness_tester()

        # Should not crash with small datasets
        try:
            accuracy = tester._directional_accuracy(returns, predictions)
            assert 0.0 <= accuracy <= 1.0

            # Test that the enhanced robustness tester methods work
            results = tester.comprehensive_robustness_test(returns, predictions)
            assert isinstance(results, dict)
            assert "overall_assessment" in results

            note(f"Small dataset test: {n_samples} samples, correlation={correlation:.2f}")

        except Exception as e:
            self.fail(f"Small dataset handling failed for n={n_samples}: {e}")


if __name__ == "__main__":
    # Configure hypothesis for CI environments
    settings.register_profile("ci", max_examples=20, deadline=2000)
    settings.register_profile("dev", max_examples=50, deadline=5000)
    settings.register_profile("thorough", max_examples=100, deadline=10000)

    # Use CI profile by default, can override with env var
    import os

    profile = os.environ.get("HYPOTHESIS_PROFILE", "ci")
    settings.load_profile(profile)

    print(f"Running property-based tests with profile: {profile}")
    unittest.main(verbosity=2)

