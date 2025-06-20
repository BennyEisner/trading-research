from datetime import date

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from api.models import ReturnData
from pipeline.run import SessionLocal

app = FastAPI()


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
