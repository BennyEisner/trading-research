from sqlalchemy.orm import Session

from api.models import PriceData, ReturnData, Ticker
from pipeline.transform import compute_returns


def store_prices_and_returns(session: Session, symbol: str, df):
    # Create ticker if it doesn't exist
    ticker = session.query(Ticker).filter(Ticker.symbol == symbol).first()
    if not ticker:
        ticker = Ticker(symbol=symbol)
        session.add(ticker)
        session.flush()
    price_records = []
    for row in df.reset_index().itertuples():
        price_record = PriceData(
            ticker_symbol=symbol,
            date=row.Date.date(),
            open=row.Open,
            high=row.High,
            low=row.Low,
            close=row.Close,
            volume=row.Volume,
        )
        session.add(price_record)
        price_records.append(price_record)

    session.flush()  # Flush to get IDs without committing

    rt = compute_returns(df)

    for i, row in enumerate(rt.reset_index().itertuples()):
        session.add(
            ReturnData(
                ticker_symbol=symbol,
                date=row.Date.date(),
                daily_return=row.daily_return,
                price_data_id=price_records[i].id,
            )
        )

    session.commit()
