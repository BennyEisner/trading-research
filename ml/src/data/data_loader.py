#!/usr/bin/env python3

"""
Data loading and preprocessing for financial time series
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add API path for PriceData model
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from api.models import PriceData, ReturnData


# Separates Database loading, Handles Database errors and validation,
# uses config object for flexibility
class DataLoader:
    """Handles loading and preprocessing of financial data"""

    def __init__(self, config):
        self.config = config
        self.database_url = config.get("database_url", "sqlite:///./returns.db")
        self.engine = None
        self.session_maker = None
        self._setup_database()

    def _setup_database(self):
        """Setup database connection"""
        # Extract database path from URL
        if self.database_url.startswith("sqlite:///"):
            db_path = self.database_url.replace("sqlite:///", "")
            
            # Handle relative paths - look in parent directory
            if not os.path.isabs(db_path):
                # Look for database in parent directory (main API directory)
                parent_db_path = os.path.join(os.path.dirname(__file__), "../../../", db_path)
                if os.path.exists(parent_db_path):
                    db_path = parent_db_path
                    self.database_url = f"sqlite:///{os.path.abspath(db_path)}"

            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database file not found: {db_path}")

            if not os.access(db_path, os.R_OK):
                raise PermissionError(f"Cannot read database file: {db_path}")

        try:
            self.engine = create_engine(self.database_url, echo=False)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.session_maker = sessionmaker(bind=self.engine)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")

    def load_single_ticker_data(self, ticker, years):
        """Load data for a single ticker"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365 * years)

        session = self.session_maker()
        try:
            # Join PriceData with ReturnData to get both price and return data
            records = (
                session.query(PriceData, ReturnData.daily_return)
                .outerjoin(ReturnData, 
                          (PriceData.ticker_symbol == ReturnData.ticker_symbol) & 
                          (PriceData.date == ReturnData.date))
                .filter(PriceData.ticker_symbol == ticker)
                .filter(PriceData.date >= start_date)
                .filter(PriceData.date <= end_date)
                .order_by(PriceData.date)
                .all()
            )

            if not records:
                raise ValueError(f"No {ticker} data available for date range {start_date} to {end_date}")

            # Convert to DataFrame
            stock_data = pd.DataFrame(
                [
                    {
                        "date": record[0].date,  # PriceData.date
                        "open": record[0].open,
                        "high": record[0].high,
                        "low": record[0].low,
                        "close": record[0].close,
                        "volume": record[0].volume,
                        "daily_return": record[1] if record[1] is not None else 0.0,  # ReturnData.daily_return
                    }
                    for record in records
                ]
            )

            stock_data["date"] = pd.to_datetime(stock_data["date"])
            stock_data = stock_data.dropna()

            return stock_data

        finally:
            session.close()

    def convert_to_weekly(self, data):
        """Convert daily data to weekly data to reduce noise"""
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])
        data.set_index("date", inplace=True)

        # Aggregate to weekly data (Friday close or last available)
        weekly_data = (
            data.resample("W-FRI")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        weekly_data.reset_index(inplace=True)
        return weekly_data

    def create_sequences(self, features, targets, lookback_window):
        """Create LSTM sequences with proper temporal alignment"""
        X, y = [], []
        max_idx = len(features) - 1

        # FIXED: Use i+1 to predict next day's return (prevent data leakage)
        for i in range(lookback_window, max_idx):
            X.append(features[i - lookback_window : i, :])
            y.append(targets[i + 1])  # Predict next period's return

        return np.array(X), np.array(y)

    def load_multi_ticker_data(self, tickers, years, feature_engineer=None):
        """Load and process data from multiple tickers"""
        all_sequences = []
        all_targets = []
        all_ticker_indices = []

        for ticker_idx, ticker in enumerate(tickers):
            try:
                # Load raw data
                ticker_data = self.load_single_ticker_data(ticker, years)
                print(f"Loaded {len(ticker_data)} daily records for {ticker}")

                # Convert to weekly if specified
                if self.config.get("prediction_horizon") == "weekly":
                    ticker_data = self.convert_to_weekly(ticker_data)
                    print(f"Converted to {len(ticker_data)} weekly records for {ticker}")

                # Feature engineering
                if feature_engineer is None:
                    from ..features.factory import create_feature_engineer
                    feature_engineer = create_feature_engineer(self.config)
                
                ticker_data = feature_engineer.calculate_all_features(ticker_data)
                feature_columns = feature_engineer.select_stable_features(ticker_data)
                ticker_features = ticker_data[feature_columns].ffill().bfill()

                # Scale features - FIXED: Use StandardScaler to preserve negative/positive relationships
                scaler = StandardScaler()
                ticker_features_scaled = scaler.fit_transform(ticker_features)

                # Compute targets - FIXED: Use forward/backward fill instead of fillna(0)
                if self.config.get("prediction_horizon") == "weekly":
                    target_values = ticker_data["close"].pct_change(periods=1).fillna(method='ffill').fillna(method='bfill').values
                else:
                    target_values = ticker_data["daily_return"].fillna(method='ffill').fillna(method='bfill').values

                # Create sequences
                X_ticker, y_ticker = self.create_sequences(
                    ticker_features_scaled,
                    target_values,
                    self.config.get("lookback_window"),
                )

                if len(X_ticker) > 0:
                    all_sequences.append(X_ticker)
                    all_targets.append(y_ticker)
                    ticker_identity = np.full(len(X_ticker), ticker_idx)
                    all_ticker_indices.append(ticker_identity)
                    print(f"Created {len(X_ticker)} sequences for {ticker}")

            except Exception as e:
                print(f"Warning: Could not process data for {ticker}: {e}")
                continue

        if not all_sequences:
            raise ValueError("No ticker data could be processed")

        # Combine all data
        X_all = np.concatenate(all_sequences, axis=0)
        y_all = np.concatenate(all_targets, axis=0)
        ticker_indices = np.concatenate(all_ticker_indices, axis=0)

        # Create ticker identity features
        num_tickers = len(tickers)
        ticker_identity_matrix = np.eye(num_tickers)[ticker_indices]

        # Broadcast ticker identity across timesteps
        timesteps = X_all.shape[1]
        ticker_identity_broadcast = np.repeat(ticker_identity_matrix[:, np.newaxis, :], timesteps, axis=1)

        # Concatenate features
        X_all_with_ticker = np.concatenate([X_all, ticker_identity_broadcast], axis=2)

        return X_all_with_ticker, y_all, len(feature_columns)
