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


def calculate_sma(data, window):
    return data.rolling(window=window).mean()


# ema better for catching trend changes quickly than sma
def calculate_ema(data, window):
    return data.ewm(span=window).mean()


# Relative strength index (|| recent price change ||)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # AVG gains
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # AVG loss
    rs = gain / loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Relationsip between moving averages
def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)  # 9 day EMA of MACD
    histogram = macd_line - signal_line  # difference
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = calculate_sma(data, window)  # Middle band
    std = data.rolling(window=window).std()
    uppper_band = sma + (std * num_std)  # +2 std dev
    lower_band = sma - (std * num_std)  # -2 std dev
    return uppper_band, lower_band, sma


def calculate_price_features(data):
    data["high_low_pct"] = (
        (data["high"] - data["low"]) / data["close"] * 100
    )  # Volatility
    data["open_close_pct"] = (
        (data["close"] - data["open"]) / data["open"] * 100
    )  # Daily Performance
    data["daily_return"] = data["close"].pct_change()  # Price Momentum

    data["price_position"] = (data["close"] - data["low"]) / (
        data["high"] - data["low"]
    )  # Close price within daily range
    return data


print("calculating indicators")
stock_data = calculate_price_features(stock_data)

# Moving averages
stock_data["sma_5"] = calculate_sma(stock_data["close"], 5)  # Short term
stock_data["sma_20"] = calculate_sma(stock_data["close"], 20)  # Med term
stock_data["sma_50"] = calculate_sma(stock_data["close"], 50)  # Long term

stock_data["ema_12"] = calculate_ema(stock_data["close"], 12)  # Fast
stock_data["ema_26"] = calculate_ema(stock_data["close"], 26)  # Slow

# Momentum Indicators
stock_data["rsi"] = calculate_rsi(stock_data["close"])

# MACD
macd_line, signal_line, histogram = calculate_macd(stock_data["close"])
stock_data["macd_line"] = macd_line
stock_data["macd_signal"] = signal_line
stock_data["macd_histogram"] = histogram

# Bollinger Bands
upper_band, lower_band, bb_middle = calculate_bollinger_bands(stock_data["close"])
stock_data["bb_upper"] = upper_band
stock_data["bb_lower"] = lower_band
stock_data["bb_middle"] = bb_middle

# Relative Position within Bollinger Bands
stock_data["bb_position"] = (stock_data["close"] - stock_data["bb_lower"]) / (
    stock_data["bb_upper"] - stock_data["bb_lower"]
)
# Feature Matrix
feature_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "high_low_pct",
    "open_close_pct",
    "daily_return",
    "price_position",
    "sma_5",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_lower",
    "bb_middle",
    "bb_position",
]
print(f"Total Features pre cleaning: {len(feature_columns)}")


# Data Cleaning
print(f"Total data point pre cleaning: {len(stock_data)}")
stock_data = stock_data.dropna()
print(f"Total data point post cleaning: {len(stock_data)}")

feature_data = stock_data[feature_columns].values
print(f"Feature Matrix Shape: {feature_data.shape}")

if np.isnan(feature_data).any():
    print("Warning: NaN values found in feature data")
if np.isinf(feature_data).any():
    print("Warning: Infinite values found in feature data")


# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(feature_data)
print(f"\nScaled feature ranges:")
print(f"  All features: {scaled_features.min():.3f} to {scaled_features.max():.3f}")

# Create separate scaler for close price only (for inverse transformation)
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = close_scaler.fit_transform(stock_data[["close"]].values)

print("Scalers created:")
print(f"  Main scaler: handles {len(feature_columns)} features")
print(f"  Close scaler: handles close price only")

print(
    f"Original price range: {stock_data['close'].min():.2f} - {stock_data['close'].max():.2f}"
)
print(
    f"Feature matrix contains {feature_data.shape[1]} features with {feature_data.shape[0]} samples"
)


# Create sequences for training
def create_sequences(
    data, lookback_window, target_col_idx=3, forecast_horizon=1
):  # close at index 3
    X, y = [], []
    for i in range(lookback_window, len(data) - forecast_horizon + 1):
        X.append(
            data[i - lookback_window : i, :]
        )  # Input sequence: lookback window days of all features
        y.append(
            data[i + forecast_horizon - 1, target_col_idx]
        )  # Target: next days close price

    return np.array(X), np.array(y)


# Set sequence length
lookback_window = 120


def create_time_series_split(X, y, train_ratio=0.7, validation_ratio=0.15):
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    validation_end = int(n_samples * (train_ratio + validation_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:validation_end]
    y_val = y[train_end:validation_end]
    X_test = X[validation_end:]
    y_train = y[validation_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Create Sequences with all features
X, y = create_sequences(scaled_features, lookback_window, target_col_idx=3)

print(f"Created {len(X)} training sequences")
print(f"X shape: {X.shape}")  # (samples, timesteps, features)
print(f"Y shape: {y.shape}")  # (samples,)
print(f"Features per timestep: {X.shape[2]}")
print(f"Lookback window:{X.shape[1]} days")

print("\n<?  BUILDING LSTM MODEL")
print("-" * 30)
def build_model(input_shape, ltsm_units=[64,32], dropout=0.3):
    model = Sequential()
    
    # Layer 1 
    model.add(LTSM(
        units=ltsm_units[0],
        return_sequences=True,
        input_shape=input_shape,
        recurrent_dropout=0.1 # Dropout on recurrent connections 
    ))

    model.add(BatchNormalization()) # To stablize training
    model.add(Dropout(dropout))
   
    # Layer 2
    model.add(LTSM(
        units=ltsm_units[1],
        return_sequences=False,
        recurrent_dropout=0.1 
    ))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))


    # Dense layers for final prediction 
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='linear'))

    return model


model = build_model(
    input_shape=lookback_window, len(feature_columns), 
    lstm_units=[64,32],
    dropout=0.3
)

def setup_training_callbacks(model_save_path='best_model.h5'):
    callbacks = [
        # Stop training id validation loss stps improving 
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
    
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )

        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True
            verbose=1
        )
    ]
        
    return callbacks
    

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

# Transform predictions back to original scale using close_scaler
y_test_original = close_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_original = close_scaler.inverse_transform(y_pred)

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
