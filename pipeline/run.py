from datetime import date, timedelta
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add config import
sys.path.append(str(Path(__file__).parent.parent))
from config.config import get_config
from api.models import Base  # Import Base to create tables
from pipeline.fetch_data import get_price_data
from pipeline.load import store_prices_and_returns

# Set up database
engine = create_engine("sqlite:///./returns.db", echo=False)

# Create all tables
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)


def main():
    """
    Load expanded universe data for enhanced LSTM training
    Uses configuration-driven ticker selection
    """
    config = get_config()
    
    # Use expanded universe for comprehensive training data
    # VIX removed due to yfinance compatibility issues
    tickers = [t for t in config.model.expanded_universe if t != "VIX"]
    
    end = date.today()
    start = end - timedelta(days=365 * 15)  # 15 years of data
    session = SessionLocal()

    print(f"Loading data for {len(tickers)} securities in expanded universe:")
    print(f"MAG7: {config.model.mag7_tickers}")
    print(f"Expanded: {tickers}")
    
    for symbol in tickers:
        try:
            df = get_price_data(symbol, start, end)
            store_prices_and_returns(session, symbol, df)
            print(f"Loaded returns for {symbol}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    print(f"\nData loading complete. {len(tickers)} securities processed.")
    print("Use MAG7 subset for final validation and specialization.")

def load_mag7_only():
    """Load only MAG7 data for testing/validation"""
    config = get_config()
    tickers = config.model.mag7_tickers
    
    end = date.today()
    start = end - timedelta(days=365 * 15)
    session = SessionLocal()

    print(f"Loading MAG7 data only: {tickers}")
    
    for symbol in tickers:
        try:
            df = get_price_data(symbol, start, end)
            store_prices_and_returns(session, symbol, df)
            print(f"Loaded returns for {symbol}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")


if __name__ == "__main__":
    main()
