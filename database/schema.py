#!/usr/bin/env python3

"""
Database Schema Definition
Simple schema optimized for financial time-series data
"""

from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, Boolean, 
    JSON, Index, UniqueConstraint, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class MarketData(Base):
    """Market data with TimescaleDB hypertable"""
    __tablename__ = 'market_data'
    
    # Primary key components
    time = Column(DateTime(timezone=True), nullable=False, primary_key=True)
    ticker = Column(String(10), nullable=False, primary_key=True)
    
    # OHLCV data
    open = Column(Numeric(10, 4))
    high = Column(Numeric(10, 4))
    low = Column(Numeric(10, 4))
    close = Column(Numeric(10, 4))
    volume = Column(Integer)
    
    # Computed fields
    daily_return = Column(Numeric(12, 8))
    adjusted_close = Column(Numeric(10, 4))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_market_data_ticker_time', 'ticker', 'time'),
        Index('idx_market_data_time', 'time'),
        {'timescaledb_hypertable': {'time_column_name': 'time'}}
    )


class Features(Base):
    """Computed features table"""
    __tablename__ = 'features'
    
    # Primary key components
    time = Column(DateTime(timezone=True), nullable=False, primary_key=True)
    ticker = Column(String(10), nullable=False, primary_key=True)
    feature_name = Column(String(50), nullable=False, primary_key=True)
    
    # Feature data
    value = Column(Numeric(15, 8))
    
    # Metadata
    feature_type = Column(String(20))  # 'technical', 'fundamental', 'macro', etc.
    computation_time = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_features_ticker_time', 'ticker', 'time'),
        Index('idx_features_name_time', 'feature_name', 'time'),
        {'timescaledb_hypertable': {'time_column_name': 'time'}}
    )


class Predictions(Base):
    """Model predictions table"""
    __tablename__ = 'predictions'
    
    # Primary key components
    time = Column(DateTime(timezone=True), nullable=False, primary_key=True)
    ticker = Column(String(10), nullable=False, primary_key=True)
    model_name = Column(String(50), nullable=False, primary_key=True)
    horizon_days = Column(Integer, nullable=False, primary_key=True)
    prediction_type = Column(String(20), nullable=False, primary_key=True)
    
    # Prediction data
    value = Column(Numeric(12, 8))
    confidence = Column(Numeric(5, 4))
    
    # Model metadata
    model_version = Column(String(20))
    feature_count = Column(Integer)
    
    # Timestamps
    prediction_time = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_predictions_ticker_time', 'ticker', 'time'),
        Index('idx_predictions_model_time', 'model_name', 'time'),
        {'timescaledb_hypertable': {'time_column_name': 'time'}}
    )


class BacktestResults(Base):
    """Backtesting results and performance metrics"""
    __tablename__ = 'backtest_results'
    
    # Primary key
    run_id = Column(String(36), primary_key=True)  # UUID
    
    # Strategy identification
    strategy_name = Column(String(50), nullable=False)
    model_name = Column(String(50))
    config_hash = Column(String(64))  # Hash of configuration
    
    # Time period
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    
    # Performance metrics
    total_return = Column(Numeric(8, 6))
    annualized_return = Column(Numeric(8, 6))
    volatility = Column(Numeric(8, 6))
    sharpe_ratio = Column(Numeric(6, 4))
    max_drawdown = Column(Numeric(6, 4))
    calmar_ratio = Column(Numeric(6, 4))
    
    # Trade statistics
    total_trades = Column(Integer)
    win_rate = Column(Numeric(5, 4))
    avg_win = Column(Numeric(8, 6))
    avg_loss = Column(Numeric(8, 6))
    
    # Configuration and metadata
    config = Column(JSON)
    notes = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_backtest_strategy_date', 'strategy_name', 'start_date'),
        Index('idx_backtest_sharpe', 'sharpe_ratio'),
    )


class ModelRegistry(Base):
    """Model version registry"""
    __tablename__ = 'model_registry'
    
    # Primary key
    model_id = Column(String(36), primary_key=True)  # UUID
    
    # Model identification
    model_name = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)
    
    # Model metadata
    model_type = Column(String(30))  # 'lstm', 'ensemble', 'cross_sectional'
    architecture = Column(JSON)
    parameters = Column(JSON)
    
    # Training metadata
    training_data_hash = Column(String(64))
    training_samples = Column(Integer)
    validation_accuracy = Column(Numeric(6, 4))
    
    # File paths
    model_path = Column(String(200))
    config_path = Column(String(200))
    
    # Status
    is_active = Column(Boolean, default=False)
    is_production = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    deployed_at = Column(DateTime(timezone=True))
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('model_name', 'version'),
        Index('idx_model_registry_name_active', 'model_name', 'is_active'),
    )


def create_tables(engine):
    """Create all tables and setup TimescaleDB hypertables"""
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Setup TimescaleDB hypertables
    with engine.connect() as conn:
        # Market data hypertable
        conn.execute(text("""
            SELECT create_hypertable('market_data', 'time', 
                                   chunk_time_interval => INTERVAL '1 week',
                                   if_not_exists => TRUE);
        """))
        
        # Features hypertable
        conn.execute(text("""
            SELECT create_hypertable('features', 'time',
                                   chunk_time_interval => INTERVAL '1 week',
                                   if_not_exists => TRUE);
        """))
        
        # Predictions hypertable
        conn.execute(text("""
            SELECT create_hypertable('predictions', 'time',
                                   chunk_time_interval => INTERVAL '1 week',
                                   if_not_exists => TRUE);
        """))
        
        conn.commit()
    
    print("Database tables and hypertables created successfully")