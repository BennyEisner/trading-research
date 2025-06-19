from datetime import date

from fastapi import FastAPI, HTTPException

from fetch_data import get_price_data

app = FastAPI(title="Simple yfinance API")


@app.get("/prices/{ticker}/{start_date}/{end_date}")
def read_prices(ticker: str, start_date: date, end_date: date):
    """HTTP GET returns history of ticker data as JSON"""
    try:
        df = get_price_data(ticker, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    records = df.reset_index().to_dict(
        orient="records"
    )  # convers date to column then converts to dict
    return {"ticker": ticker, "data": records}
