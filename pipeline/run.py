from datetime import date, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pipeline.fetch_data import get_price_data
from pipeline.load import store_prices_and_returns

# Set up database
engine = create_engine("sqlite:///./returns.db", echo=False)
SessionLocal = sessionmaker(bind=engine)


def main():
    # Can adjust later, these are just top 50 by market cap from gpt
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA"]
    end = date.today()
    start = end - timedelta(days=365)
    session = SessionLocal()

    for symbol in tickers:
        try:
            df = get_price_data(symbol, start, end)
            store_prices_and_returns(session, symbol, df)
            print(f"Loaded returns for {symbol}")
        except Exception as e:
            print(f"Error proccessing {symbol}: {e}")


if __name__ == "__main__":
    main()
