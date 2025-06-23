import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

sys.path.append("/Users/beneisner/financial-returns-api")

from api.models import Base, PriceData, get_session

np.random.seed(42)
tf.random.set_seed(42)

ticker = "NVDA"  # Set for testing purposes

DATABASE_URL = "sqlite:///./returns.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

end_date = datetime.now().date()
start_date = end_date - timedelta(365)

print(f"Querying {ticker} data from {start_date} to {end_date}")

session = SessionLocal()

try:
    ticker_records = (
        session.query(PriceData)
        .filter(PriceData.ticker_symbol == ticker)
        .filter(PriceData.date >= start_date)
        .filter(PriceData.date <= end_date)
        .order_by(PriceData.date)
        .all()
    )

    if not ticker_records:
        raise ValueError(f"No {ticker} data available in the database")

    stock_data = pd.DataFrame(
        [
            {
                "date": record.date,
                "open": record.open,
                "high": record.high,
                "low": record.low,
                "close": record.close,
                "volume": record.volume,
            }
            for record in ticker_records
        ]
    )
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    stock_data = stock_data.dropna()

    missing_data = stock_data.isnull().sum()
    if missing_data.any():
        print(f"Missing data found: ")
        print(missing_data[missing_data > 0])
    else:
        print("No missing data found")

    print("\n Sample data:")
    print(stock_data.head())

    print(f"Data Validation: {len(stock_data)} days available for training")

except Exception as e:
    print(f"Error loading data: {e}")

finally:
    session.close()


print("\n= PREPROCESSING DATA FOR LSTM")
print("-" * 30)

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(stock_data[["close"]])

print(
    f"Original price range: {stock_data['close'].min():.2f} - {stock_data['close'].max():.2f}"
)
print(f"Scaled price range: {scaled_prices.min():.3f} - {scaled_prices.max():.3f}")


# Create sequences for training
def create_sequences(data, lookback_window):
    X, y = [], []
    for i in range(lookback_window, len(data)):
        X.append(data[i - lookback_window : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# Set sequence length
lookback_window = 120

# Create sequences
X, y = create_sequences(scaled_prices, lookback_window)

# Reshape for LSTM input
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"Created {len(X)} training sequences")
print(f"Input shape: {X.shape}")  # (samples, timesteps, features)
print(f"Output shape: {y.shape}")  # (samples,)


print("\n<?  BUILDING LSTM MODEL")
print("-" * 30)

# Create Sequential model
model = Sequential(
    [
        # First LSTM layer
        LSTM(
            units=50,
            return_sequences=True,  # Pass sequences to next layer
            input_shape=(lookback_window, 1),
        ),
        # Dropout for regularization to prevent overfitting
        Dropout(0.2),
        # Second LSTM layer to learn from first layer
        LSTM(units=50, return_sequences=False),  # Don't return sequences
        Dropout(0.2),
        # Dense layer for prediction
        Dense(units=1, activation="linear"),
    ]
)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Adaptive learning rate
    loss="mean_squared_error",
    metrics=["mean_absolute_error"],  # Track prediction accuracy
)

# Display model architecture
print("Model Architecture:")
model.summary()


print("\n=? TRAINING MODEL (QUICK TEST)")
print("-" * 30)

# Spliit data 80/20
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train the model
print("Starting training...")
start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1,  # Show training progress
    shuffle=False,
)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")


print("\n=? EVALUATING MODEL PERFORMANCE")
print("-" * 30)

# Make predictions on test data
y_pred = model.predict(X_test)

# Transform predictions back to original scale
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_original = scaler.inverse_transform(y_pred)

# Calculate error metrics
mse = mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)

print(f"Test Results:")
print(f"  Mean Squared Error: ${mse:.2f}")
print(f"  Mean Absolute Error: ${mae:.2f}")
print(f"  Root Mean Squared Error: ${rmse:.2f}")

percentage_error = (mae / y_test_original.mean()) * 100
print(f"  Average Percentage Error: {percentage_error:.2f}%")


# Check env success
success_checks = [
    ("TensorFlow imported", True),
    ("Model created successfully", model is not None),
    ("Training completed", training_time > 0),
    ("Predictions generated", len(y_pred) > 0),
    ("Reasonable accuracy", percentage_error < 50),
]
