from datetime import date

import yfinance as yf


def get_price_data(ticker: str, start: date, end: date):
    df = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat())
    return df


def main():
    pass


if __name__ == "__main__":
    main()
