from models import PriceRecord, ReturnData
from sqlalchemy.orm import Session

from pipeline.transform import compute_returns


def store_prices_and_returns(session: Session, symbol: str, df):
    for row in df.reset_index().itertuples():
        session.add(PriceRecord(symbol=symbol, date=row.Index, close=row.CLose))

    rt = compute_returns(df)

    for row in rt.reset_index().iterupules():
        session.add(
            ReturnData(symbol=symbol, date=row.Index, daily_returnn=row.daily_return)
        )

    session.commit()
