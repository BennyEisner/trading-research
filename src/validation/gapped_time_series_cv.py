#!/usr/bin/env python3

"""
Gapped Time Series Cross-Validator
Prevents look ahead bias with gaps between train/test periods
"""

from datetime import datetime, timedelta
from typing import Dict, Generator, List, Tuple

import numpy as np
import pandas as pd


class GappedTimeSeriesCV:

    def __init__(self, n_splits: int = 5, test_size: float = 0.2, gap_size: int = 10, expanding_window: bool = True):
        """Initialize gapped CV via backwards window generation"""

        self.n_splits = n_splits
        self.test_size = test_size
        self.gap_size = gap_size
        self.expanding_window = expanding_window

    def split(self, X: np.ndarray, y: np.ndarray = None) -> Generator[Tuple[List[int], List[int]], None, None]:
        """Generate time-series splits with gaps"""

        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)

        # calculate split points working backwards from end
        splits = []
        for i in range(self.n_splits):
            # Calculate test period - work backwards without overlap
            test_end = n_samples - i * (test_size_samples + self.gap_size)
            test_start = test_end - test_size_samples

            # Training period ends with gap before test
            train_end = test_start - self.gap_size

            if self.expanding_window:
                train_start = 0
            else:
                train_window_size = test_size_samples * 2
                train_start = max(0, train_end - train_window_size)

            # Index validation
            if train_start >= train_end or train_end >= test_start or test_start >= test_end:
                continue
            if train_start < 0 or test_end > n_samples:
                continue

            train_indices = list(range(train_start, train_end))
            test_indices = list(range(test_start, test_end))

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        # Return splits in correct chronological order (earliest first)
        for train_indices, test_indices in reversed(splits):
            yield train_indices, test_indices

    def get_split_dates(self, dates: pd.DatetimeIndex) -> List[Dict]:
        """Get split information with dates"""

        split_info = []
        for i, (train_idx, test_idx) in enumerate(self.split(np.arange(len(dates)))):
            info = {
                "split": i,
                "train_start": dates[train_idx[0]],
                "train_end": dates[train_idx[-1]],
                "gap_start": dates[train_idx[-1] + 1] if train_idx[-1] + 1 < len(dates) else None,
                "gap_end": dates[test_idx[0] - 1] if test_idx[0] > 0 else None,
                "test_start": dates[test_idx[0]],
                "test_end": dates[test_idx[-1]],
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "gap_days": self.gap_size,
            }
            split_info.append(info)

        return split_info

    def validate_no_leakage(self, dates: pd.DatetimeIndex) -> bool:
        """Validate no look ahead bias exists in splits"""

        splits = list(self.split(np.arange(len(dates))))

        for i, (train_idx, test_idx) in enumerate(splits):
            max_train_date = dates[train_idx[-1]]
            min_test_date = dates[test_idx[0]]

            # Check training data ends before test data starts
            if max_train_date >= min_test_date:
                print(
                    f"Data Leakage Found in split {i}: Training data ({max_train_date}) overlaps with test data ({min_test_date})"
                )
                return False

            # Check gap size
            actual_gap = (min_test_date - max_train_date).days
            if actual_gap < self.gap_size:
                print(f"Insufficient gap in split {i}: Only {actual_gap} day gap when should be {self.gap_size}")
                return False

        print(f"No lookahead bias found in {len(splits)} splits")
        return True

    def print_split_summary(self, dates: pd.DatetimeIndex):
        """Print splits summary"""

        split_info = self.get_split_dates(dates)

        print("\n" + "=" * 80)
        print("GAPPED TIME-SERIES CV SPLIT SUMMARY")
        print("=" * 80)

        for info in split_info:
            print(f"\nSplit {info['split']}:")
            print(f"  Train: {info['train_start'].date()} to {info['train_end'].date()} ({info['train_size']} samples)")
            print(f"  Gap:   {info['gap_days']} days")
            print(f"  Test:  {info['test_start'].date()} to {info['test_end'].date()} ({info['test_size']} samples)")

        print("\n" + "=" * 80)


class WalkForwardValidator:
    """Walk-forward validation with automatic retraining"""

    def __init__(self, retrain_freq: int = 60, min_train_size: int = 252, test_size: int = 20):
        self.retrain_freq = retrain_freq
        self.min_train_size = min_train_size
        self.test_size = test_size

    def split(self, X: np.ndarray, y: np.ndarray = None) -> Generator[Tuple[List[int], List[int]], None, None]:
        """Generate walk-forward splits"""

        n_samples = len(X)

        # Start with minimum training size
        current_pos = self.min_train_size

        while current_pos + self.test_size <= n_samples:
            # Training indices (expanding window)
            train_indices = list(range(0, current_pos))

            # Test indices
            test_end = min(current_pos + self.test_size, n_samples)
            test_indices = list(range(current_pos, test_end))

            yield train_indices, test_indices

            # Move forward
            current_pos += self.retrain_freq

    def should_retrain(
        self, current_performance: float, historical_performance: List[float], threshold: float = 0.1
    ) -> bool:
        """Determine if model should be retrained based on performance degradation"""

        if len(historical_performance) < 3:
            return False

        recent_avg = np.mean(historical_performance[-3:])

        # Check for significant degrdation
        if current_performance < recent_avg * (1 - threshold):
            return True

        return False


def create_gapped_cv(n_splits: int = 5, test_size: float = 0.2, gap_size: int = 10, expanding_window: bool = True):
    """Helper function to create gapped CV"""

    return GappedTimeSeriesCV(
        n_splits=n_splits, test_size=test_size, gap_size=gap_size, expanding_window=expanding_window
    )
