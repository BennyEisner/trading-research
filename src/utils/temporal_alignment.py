#!/usr/bin/env python3

"""
Ensures all tickers have identical date ranges and sample counts
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TemporalAligner:
    """
    Ensures perfect temporal alignment across multiple ticker datasets
    """

    def __init__(self):
        self.common_dates = None
        self.alignment_stats = {}

    def find_common_date_range(self, ticker_data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """
        Find the intersection of valid dates across all tickers

        Args:
            ticker_data: Dict mapping ticker -> DataFrame with date index

        Returns:
            DatetimeIndex of common dates across all tickers
        """
        if not ticker_data:
            raise ValueError("No ticker data provided")

        # Get date ranges for each ticker
        date_ranges = {}
        for ticker, data in ticker_data.items():
            if not isinstance(data.index, pd.DatetimeIndex):
                if "date" in data.columns:
                    dates = pd.to_datetime(data["date"])
                    # Convert Series to DatetimeIndex
                    dates = pd.DatetimeIndex(dates)
                else:
                    raise ValueError(f"No date information found for {ticker}")
            else:
                dates = data.index

            # Remove NaN dates and sort, ensure it's a DatetimeIndex
            dates = dates.dropna().sort_values()
            if isinstance(dates, pd.Series):
                dates = pd.DatetimeIndex(dates)

            date_ranges[ticker] = dates

            print(f"   {ticker}: {len(dates):,} dates from {dates.min()} to {dates.max()}")

        # Find intersection of all date ranges
        # Start with first ticker's dates as DatetimeIndex
        first_ticker = list(ticker_data.keys())[0]
        common_dates = date_ranges[first_ticker]
        print(f"   Starting with {first_ticker}: {len(common_dates):,} dates")

        # Find intersection with each subsequent ticker
        for ticker, dates in date_ranges.items():
            if ticker == first_ticker:
                continue  # Skip the first ticker we already used
            common_dates = common_dates.intersection(dates)
            print(f"   After {ticker}: {len(common_dates):,} common dates")

        if len(common_dates) == 0:
            raise ValueError("No common dates found across all tickers")

        self.common_dates = common_dates
        self.alignment_stats = {
            "common_dates": len(common_dates),
            "date_range": f"{common_dates.min()} to {common_dates.max()}",
            "individual_ranges": date_ranges,
        }

        print(f"Common date range: {len(common_dates):,} dates from {common_dates.min()} to {common_dates.max()}")

        return common_dates

    def align_ticker_data(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all ticker DataFrames to common date range

        Args:
            ticker_data: Dict mapping ticker -> DataFrame

        Returns:
            Dict mapping ticker -> aligned DataFrame with identical date ranges
        """
        if self.common_dates is None:
            self.find_common_date_range(ticker_data)

        aligned_data = {}

        for ticker, data in ticker_data.items():
            # Ensure date column exists and is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                if "date" in data.columns:
                    data = data.set_index("date")
                else:
                    raise ValueError(f"No date information found for {ticker}")

            # Filter to common dates only
            aligned = data.loc[data.index.isin(self.common_dates)].copy()

            # Ensure exact same date index
            aligned = aligned.reindex(self.common_dates)

            # Verify no NaN values in critical columns after alignment
            critical_cols = ["open", "high", "low", "close", "volume"]
            available_critical = [col for col in critical_cols if col in aligned.columns]

            if aligned[available_critical].isna().any().any():
                print(f"⚠️ Warning: {ticker} has NaN values after alignment, forward filling...")
                aligned[available_critical] = aligned[available_critical].fillna(method="ffill").fillna(method="bfill")

            aligned_data[ticker] = aligned

            print(f"{ticker}: {len(aligned):,} records aligned ({aligned.index.min()} to {aligned.index.max()})")

        # Final verification: ensure all tickers have identical indices
        self.verify_alignment(aligned_data)

        return aligned_data

    def verify_alignment(self, aligned_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Verify that all ticker DataFrames have identical date indices

        Args:
            aligned_data: Dict of aligned ticker DataFrames

        Returns:
            True if perfectly aligned, raises ValueError if not
        """
        if not aligned_data:
            raise ValueError("No data to verify")

        # Get reference index from first ticker
        reference_ticker = list(aligned_data.keys())[0]
        reference_index = aligned_data[reference_ticker].index

        print(f"\n🔍 VERIFYING TEMPORAL ALIGNMENT:")
        print(f"   Reference: {reference_ticker} with {len(reference_index):,} records")

        for ticker, data in aligned_data.items():
            if not data.index.equals(reference_index):
                print(f"❌ ALIGNMENT FAILURE:")
                print(f"   {reference_ticker}: {len(reference_index):,} records")
                print(f"   {ticker}: {len(data.index):,} records")

                # Show specific differences
                missing_in_ticker = reference_index.difference(data.index)
                extra_in_ticker = data.index.difference(reference_index)

                if len(missing_in_ticker) > 0:
                    print(f"   Missing from {ticker}: {len(missing_in_ticker)} dates")
                if len(extra_in_ticker) > 0:
                    print(f"   Extra in {ticker}: {len(extra_in_ticker)} dates")

                raise ValueError(f"Ticker {ticker} is not properly aligned with {reference_ticker}")
            else:
                print(f"   {ticker}: {len(data.index):,} records - PERFECTLY ALIGNED")

        print(f"ALL TICKERS PERFECTLY ALIGNED: {len(reference_index):,} identical dates")
        return True

    def get_alignment_stats(self) -> Dict:
        """
        Get statistics about the alignment process

        Returns:
            Dict with alignment statistics
        """
        return self.alignment_stats.copy()

    def align_features_data(self, ticker_features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align feature-engineered data ensuring all tickers have same samples

        Args:
            ticker_features: Dict mapping ticker -> features DataFrame

        Returns:
            Dict mapping ticker -> aligned features DataFrame
        """
        print(f"\nALIGNING FEATURE DATA ACROSS TICKERS:")

        # Find common dates after feature engineering
        common_dates = self.find_common_date_range(ticker_features)

        # Align all ticker feature data
        aligned_features = self.align_ticker_data(ticker_features)

        # Additional verification for feature data
        print(f"\nFEATURE ALIGNMENT SUMMARY:")
        for ticker, features in aligned_features.items():
            feature_count = len(
                [
                    col
                    for col in features.columns
                    if col not in ["date", "open", "high", "low", "close", "volume", "ticker", "daily_return"]
                ]
            )
            print(f"   {ticker}: {len(features):,} samples × {feature_count} features")

        return aligned_features


def align_multi_ticker_data(ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to align multiple ticker datasets

    Args:
        ticker_data: Dict mapping ticker -> DataFrame

    Returns:
        Dict mapping ticker -> aligned DataFrame with identical date ranges
    """
    aligner = TemporalAligner()
    return aligner.align_ticker_data(ticker_data)


def verify_temporal_consistency(data: Dict[str, pd.DataFrame]) -> bool:
    """
    Verify that datasets are temporally consistent

    Args:
        data: Dict mapping ticker -> DataFrame

    Returns:
        True if consistent, False otherwise
    """
    aligner = TemporalAligner()
    try:
        aligner.verify_alignment(data)
        return True
    except ValueError:
        return False

