#!/usr/bin/env python3

"""
Test expanded universe data loading for LSTM training
"""

import sys
from pathlib import Path

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from config.config import get_config
from tests.data_loader import load_test_data, validate_data_format


def test_expanded_universe_loading():
    """Test loading expanded universe tickers"""

    config = get_config()

    # Get expanded universe tickers (exclude VIX fpr now for yfinance compatibility)
    expanded_tickers = [t for t in config.model.expanded_universe if t != "VIX"]

    print(f"Testing expanded universe data loading...")
    print(f"Total tickers to test: {len(expanded_tickers)}")
    print(f"Expanded universe: {expanded_tickers}")
    print()

    # Test with smaller dataset first
    print("Loading 200 days of data for all expanded universe tickers...")

    try:
        data = load_test_data(expanded_tickers, days=200)

        print(f"Successfully loaded data for {len(data)} out of {len(expanded_tickers)} tickers")
        print()

        
        total_records = 0
        for ticker, df in data.items():
            records = len(df)
            total_records += records
            date_range = f"{df['date'].min()} to {df['date'].max()}"
            print(f"  {ticker}: {records} records ({date_range})")

        print()
        print(f"SUMMARY:")
        print(f"  Successful tickers: {len(data)}/{len(expanded_tickers)} ({len(data)/len(expanded_tickers)*100:.1f}%)")
        print(f"  Total records: {total_records:,}")
        print(f"  Avg records per ticker: {total_records/len(data):.0f}")

        # Validate data format
        if validate_data_format(data):
            print(f"  Data validation: PASS")
        else:
            print(f"  Data validation: FAIL")

        # Success criteria
        success_rate = len(data) / len(expanded_tickers)
        if success_rate >= 0.8:  # 80% success rate
            print()
            print("SUCCESS: Expanded universe data loading validated")
            print("Ready for expanded universe LSTM training")
            return True
        else:
            print()
            print(f"WARNING: Only {success_rate:.1%} success rate")
            print("May need to proceed with available tickers")
            return False

    except Exception as e:
        print(f"ERROR: Failed to load expanded universe data: {e}")
        return False


if __name__ == "__main__":
    success = test_expanded_universe_loading()
    exit(0 if success else 1)

