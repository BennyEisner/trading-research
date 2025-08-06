#!/usr/bin/env python3

"""
Data Loading Utilities for Tests
"""

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import yfinance as yf


def load_test_data(tickers: List[str], days: int = 500) -> Dict[str, pd.DataFrame]:
    """
    Load test data using the same method as production pipeline

    Args:
        tickers: List of ticker symbols to load
        days: Number of days of historical data

    Returns:
        Dictionary {ticker: ohlcv_dataframe} matching expected format
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    ticker_data = {}

    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date.isoformat(), end=end_date.isoformat())

            if data.empty:
                print(f"WARNING: No data retrieved for {ticker}")
                continue

            # Reset index to make Date a column
            data = data.reset_index()

            data.columns = [col.lower() for col in data.columns]

            # Validate required OHLCV columns exist
            required_columns = ["date", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                print(f"WARNING: Missing columns for {ticker}: {missing_columns}")
                continue

            ticker_data[ticker] = data
            print(f"Loaded {len(data)} records for {ticker}: {data['date'].min()} to {data['date'].max()}")

        except Exception as e:
            print(f"ERROR: Failed to load data for {ticker}: {e}")
            continue

    return ticker_data


def validate_data_format(ticker_data: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate that data format matches expectations

    Args:
        ticker_data: Dictionary of ticker dataframes to validate

    Returns:
        True if all data is valid, False otherwise
    """
    required_columns = ["date", "open", "high", "low", "close", "volume"]

    for ticker, data in ticker_data.items():
        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"VALIDATION FAILED: {ticker} missing columns: {missing_columns}")
            return False

        # Check data types and ranges
        if data["close"].isna().all():
            print(f"VALIDATION FAILED: {ticker} has all NaN close prices")
            return False

        if len(data) < 50:  # Need minimum data for pattern detection
            print(f"VALIDATION FAILED: {ticker} has insufficient data ({len(data)} records)")
            return False

    print(f"DATA VALIDATION PASSED: {len(ticker_data)} tickers with proper OHLCV format")
    return True


if __name__ == "__main__":
    # Test data loading with a small sample
    test_tickers = ["AAPL", "MSFT"]
    data = load_test_data(test_tickers, days=100)

    if validate_data_format(data):
        print("Data loading test PASSED")
        for ticker, df in data.items():
            print(f"  {ticker}: {df.shape} - {df.columns.tolist()}")
    else:
        print("Data loading test FAILED")

