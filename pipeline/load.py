from sqlalchemy.exc import IntegrityError
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

    # Check for existing data to avoid duplicates
    existing_dates = set(
        row[0]
        for row in session.query(PriceData.date)
        .filter(PriceData.ticker_symbol == symbol)
        .all()
    )

    price_records = []
    for row in df.reset_index().itertuples():
        row_date = row.Date.date()

        # Skip if data already exists for this date
        if row_date in existing_dates:
            continue

        price_record = PriceData(
            ticker_symbol=symbol,
            date=row_date,
            open=row.Open,
            high=row.High,
            low=row.Low,
            close=row.Close,
            volume=row.Volume,
        )
        session.add(price_record)
        price_records.append(price_record)

    if not price_records:
        print(f"No new data to load for {symbol}")
        return

    session.flush()  # Flush to get IDs without committing

    rt = compute_returns(df)

    for i, row in enumerate(rt.reset_index().itertuples()):
        row_date = row.Date.date()

        # Skip if data already exists for this date
        if row_date in existing_dates:
            continue

        try:
            session.add(
                ReturnData(
                    ticker_symbol=symbol,
                    date=row_date,
                    daily_return=row.daily_return,
                    price_data_id=price_records[
                        len([p for p in price_records if p.date <= row_date]) - 1
                    ].id,
                )
            )
        except IntegrityError:
            # Handle case where unique constraint is violated
            session.rollback()
            print(f"Duplicate data detected for {symbol} on {row_date}, skipping...")
            continue

    try:
        session.commit()
        print(f"Successfully loaded data for {symbol}")
    except IntegrityError:
        session.rollback()
        print(f"Failed to load data for {symbol} due to integrity constraint violation")
