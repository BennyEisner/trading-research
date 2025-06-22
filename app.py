from datetime import date

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.models import ReturnData, Ticker
from pipeline.run import SessionLocal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/prices/{ticker}/{start_date}/{end_date}")
def read_returns(
    ticker: str, start_date: date, end_date: date, db: Session = Depends(get_db)
):
    if start_date > end_date:
        raise HTTPException(400, detail="start_date after end date")

    rows = (
        db.query(ReturnData)
        .filter(ReturnData.ticker_symbol == ticker)
        .filter(ReturnData.date.between(start_date, end_date))
        .order_by(ReturnData.date)
        .all()
    )

    if not rows:
        raise HTTPException(
            404, f"No returns for {ticker} in ({start_date}, {end_date}"
        )

    data = [{"date": r.date, "daily_return": r.daily_return} for r in rows]
    return {"symbol": ticker, "returns": data}


@app.get("/tickers")
def get_tickers(db: Session = Depends(get_db)):
    tickers = db.query(Ticker).all()
    return [{"symbol": t.symbol, "name": t.name, "sector": t.sector} for t in tickers]


@app.get("/tickers/{symbol}")
def get_ticker(symbol: str, db: Session = Depends(get_db)):
    ticker = db.query(Ticker).filter(Ticker.symbol == symbol).first()
    if not ticker:
        raise HTTPException(404, f"Ticker {symbol} not found")

    return {"symbol": ticker.symbol, "name": ticker.name, "sector": ticker.sector}
