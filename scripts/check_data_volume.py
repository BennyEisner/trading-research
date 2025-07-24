#!/usr/bin/env python3

"""
Check how much historical data we have in the database
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.config.config import Config
from src.data.data_loader import DataLoader

def check_data_volume():
    """Check data volume for each ticker"""
    
    config = Config({
        "database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db"
    })
    
    data_loader = DataLoader(config)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA"]
    
    print("=" * 60)
    print("DATABASE DATA VOLUME CHECK")
    print("=" * 60)
    
    total_records = 0
    
    for ticker in tickers:
        try:
            # Load maximum available data
            data = data_loader.load_single_ticker_data(ticker, years=20)  # Try 20 years
            
            date_range = f"{data['date'].min().date()} to {data['date'].max().date()}"
            years_span = (data['date'].max() - data['date'].min()).days / 365.25
            
            print(f"{ticker}:")
            print(f"  Records: {len(data)}")
            print(f"  Date range: {date_range}")
            print(f"  Years span: {years_span:.1f} years")
            print(f"  Has daily_return: {'daily_return' in data.columns}")
            if 'daily_return' in data.columns:
                non_zero_returns = (data['daily_return'] != 0).sum()
                print(f"  Non-zero returns: {non_zero_returns}")
            print()
            
            total_records += len(data)
            
        except Exception as e:
            print(f"{ticker}: ERROR - {e}")
            print()
    
    print("=" * 60)
    print(f"TOTAL RECORDS ACROSS ALL TICKERS: {total_records}")
    print(f"AVERAGE RECORDS PER TICKER: {total_records / len(tickers):.0f}")
    
    expected_per_ticker = 252 * 15  # 15 years × 252 trading days
    print(f"EXPECTED (15 years): {expected_per_ticker} per ticker")
    
    if total_records > expected_per_ticker * len(tickers) * 0.8:  # 80% threshold
        print("✅ SUFFICIENT DATA FOR TRAINING")
    else:
        print("❌ INSUFFICIENT DATA - Re-run pipeline to populate more data")
    
    print("=" * 60)

if __name__ == "__main__":
    check_data_volume()