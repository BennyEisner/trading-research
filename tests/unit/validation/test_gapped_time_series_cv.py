#!/usr/bin/env python3

"""
Unit tests for GappedTimeSeriesCV
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from validation.gapped_time_series_cv import GappedTimeSeriesCV, WalkForwardValidator, create_gapped_cv


class TestGappedTimeSeriesCV(unittest.TestCase):
    """Core tests for gapped time-series cross-validation"""

    def setUp(self):
        """Set up simple test environment"""
        self.cv = GappedTimeSeriesCV(n_splits=3, test_size=0.2, gap_size=5)
        self.X = np.random.random((200, 10))  # Simple 2D array
        self.dates = pd.date_range("2020-01-01", periods=200, freq="D")

    def test_initialization(self):
        """Test CV initializes with correct parameters"""
        cv = GappedTimeSeriesCV(n_splits=4, test_size=0.15, gap_size=10, expanding_window=False)

        self.assertEqual(cv.n_splits, 4)
        self.assertEqual(cv.test_size, 0.15)
        self.assertEqual(cv.gap_size, 10)
        self.assertFalse(cv.expanding_window)

    def test_helper_function(self):
        """Test create_gapped_cv helper function"""
        cv = create_gapped_cv(n_splits=2, gap_size=7)

        self.assertIsInstance(cv, GappedTimeSeriesCV)
        self.assertEqual(cv.n_splits, 2)
        self.assertEqual(cv.gap_size, 7)

    def test_basic_split_generation(self):
        """Test that splits are generated correctly"""
        splits = list(self.cv.split(self.X))

        # Should generate some splits
        self.assertGreater(len(splits), 0)
        self.assertLessEqual(len(splits), 3)

        # Each split should have valid train/test indices
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)

            # No overlap between train and test
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)

            self.assertLess(max(train_idx), min(test_idx))

    def test_gap_enforcement(self):
        """Test that gaps are enforced between train and test"""
        for train_idx, test_idx in self.cv.split(self.X):
            gap = min(test_idx) - max(train_idx) - 1
            self.assertGreaterEqual(gap, self.cv.gap_size)

    def test_no_data_leakage_validation(self):
        """Test leakage validation works"""
        result = self.cv.validate_no_leakage(self.dates)
        self.assertIsInstance(result, bool)
        # With proper gaps, should not have leakage
        self.assertTrue(result)

    def test_split_date_info(self):
        """Test getting split information with dates"""
        split_info = self.cv.get_split_dates(self.dates)

        self.assertIsInstance(split_info, list)
        self.assertGreater(len(split_info), 0)

        for info in split_info:
            # Check basic structure
            self.assertIn("train_start", info)
            self.assertIn("train_end", info)
            self.assertIn("test_start", info)
            self.assertIn("test_end", info)

            # Check chronological order
            self.assertLess(info["train_end"], info["test_start"])

    def test_expanding_vs_fixed_window(self):
        """Test difference between expanding and fixed windows"""
        cv_expanding = GappedTimeSeriesCV(n_splits=2, expanding_window=True)
        cv_fixed = GappedTimeSeriesCV(n_splits=2, expanding_window=False)

        splits_exp = list(cv_expanding.split(self.X))
        splits_fix = list(cv_fixed.split(self.X))

        # Both should generate splits
        self.assertGreater(len(splits_exp), 0)
        self.assertGreater(len(splits_fix), 0)

        # Expanding window should start from index 0
        if len(splits_exp) > 0:
            train_idx, _ = splits_exp[0]
            self.assertEqual(min(train_idx), 0)


class TestGappedCVEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def setUp(self):
        """Set up test environment"""
        self.cv = GappedTimeSeriesCV(n_splits=3, test_size=0.2, gap_size=5)
        self.dates = pd.date_range("2020-01-01", periods=100, freq="D")

    def test_small_dataset(self):
        """Test behavior with small datasets"""
        small_X = np.random.random((30, 5))
        cv = GappedTimeSeriesCV(n_splits=3, test_size=0.3, gap_size=5)

        splits = list(cv.split(small_X))
        # May generate fewer splits or none due to size constraints
        self.assertLessEqual(len(splits), 3)

    def test_large_gap(self):
        """Test behavior with gap larger than reasonable"""
        cv = GappedTimeSeriesCV(n_splits=2, gap_size=50, test_size=0.2)
        X = np.random.random((100, 5))

        splits = list(cv.split(X))
        # Should handle gracefully (may generate no splits)
        self.assertIsInstance(splits, list)

    def test_print_summary_no_crash(self):
        """Test that printing doesn't crash"""
        try:
            self.cv.print_split_summary(self.dates[:50])  # Small dataset
        except Exception as e:
            self.fail(f"print_split_summary crashed: {e}")


class TestWalkForwardValidator(unittest.TestCase):
    """Simplified tests for walk-forward validation"""

    def setUp(self):
        """Set up walk-forward validator"""
        self.wf = WalkForwardValidator(retrain_freq=20, min_train_size=50, test_size=10)
        self.X = np.random.random((200, 5))

    def test_initialization(self):
        """Test walk-forward validator initializes correctly"""
        wf = WalkForwardValidator(retrain_freq=30, min_train_size=100, test_size=15)

        self.assertEqual(wf.retrain_freq, 30)
        self.assertEqual(wf.min_train_size, 100)
        self.assertEqual(wf.test_size, 15)

    def test_split_generation(self):
        """Test walk-forward generates valid splits"""
        splits = list(self.wf.split(self.X))

        self.assertGreater(len(splits), 0)

        for train_idx, test_idx in splits:
            # Train set should be at least min size
            self.assertGreaterEqual(len(train_idx), self.wf.min_train_size)

            self.assertGreater(len(test_idx), 0)
            self.assertLessEqual(len(test_idx), self.wf.test_size)

            # Train should start from beginning (expanding window)
            self.assertEqual(min(train_idx), 0)

            # Train should come before test
            self.assertLess(max(train_idx), min(test_idx))

    def test_progressive_nature(self):
        """Test that splits progress forward in time"""
        splits = list(self.wf.split(self.X))

        if len(splits) > 1:
            # Training sets should get larger (expanding window)
            train_sizes = [len(train_idx) for train_idx, _ in splits]
            for i in range(1, len(train_sizes)):
                self.assertGreaterEqual(train_sizes[i], train_sizes[i - 1])

            # Test periods should progress forward
            test_starts = [min(test_idx) for _, test_idx in splits]
            for i in range(1, len(test_starts)):
                self.assertGreater(test_starts[i], test_starts[i - 1])

    def test_retrain_decision(self):
        """Test retraining decision logic"""
        # Good performance - no retrain needed
        good_result = self.wf.should_retrain(0.65, [0.6, 0.62, 0.64], threshold=0.1)
        self.assertFalse(good_result)

        # Poor performance - retrain needed
        poor_result = self.wf.should_retrain(0.45, [0.6, 0.62, 0.64], threshold=0.1)
        self.assertTrue(poor_result)

        # Insufficient history - no retrain
        short_result = self.wf.should_retrain(0.5, [0.6], threshold=0.1)
        self.assertFalse(short_result)

    def test_edge_cases(self):
        """Test edge cases"""
        tiny_X = np.random.random((40, 3))
        splits_tiny = list(self.wf.split(tiny_X))

        # Should generate no splits if data insufficient
        if len(tiny_X) < (self.wf.min_train_size + self.wf.test_size):
            self.assertEqual(len(splits_tiny), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
