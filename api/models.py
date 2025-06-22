import enum

from sqlalchemy import (Column, Date, DateTime, Enum, Float, ForeignKey,
                        Integer, String, Text, UniqueConstraint, create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class RunStatus(enum.Enum):
    SUCCESS = "success"
    FAIL = "fail"


# Stores data about each security
class Ticker(Base):
    __tablename__ = "tickers"

    symbol = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    sector = Column(String, nullable=True)

    # Define relationships between db tables
    prices = relationship("PriceData", back_populates="ticker")
    returns = relationship("ReturnData", back_populates="ticker")


# Stores data for each smybol and date
class PriceData(Base):
    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_symbol = Column(String, ForeignKey("tickers.symbol"), index=True)
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    ticker = relationship("Ticker", back_populates="prices")
    return_record = relationship("ReturnData", back_populates="price", uselist=False)


# Computes daily return for each symbol
class ReturnData(Base):
    __tablename__ = "return_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_symbol = Column(String, ForeignKey("tickers.symbol"), index=True)
    date = Column(Date, index=True)
    price_data_id = Column(Integer, ForeignKey("price_data.id"), index=True)
    daily_return = Column(Float)

    ticker = relationship("Ticker", back_populates="returns")
    price = relationship("PriceData", back_populates="return_record", uselist=False)

    __table_args__ = (
        UniqueConstraint("ticker_symbol", "date", name="unique_ticker_date"),
    )


# Tracks each execution of ETL pipeline
class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_timestamp = Column(DateTime, index=True)
    status = Column(Enum(RunStatus), nullable=False)
    log = Column(Text)


# Bootstraps database
def get_session(database_url: str):
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
