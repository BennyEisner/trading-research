import yfinance as yf


def get_price_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5d")
    return df


if __name__ == "__main__":
    sample = get_price_data("AAPL")
    print(sample)
