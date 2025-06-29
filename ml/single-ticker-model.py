import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.callbacks import (
    EarlyStopping,
    LambdaCallback,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from keras.layers import LSTM, BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append("/Users/beneisner/financial-returns-api")

from api.models import PriceData

np.random.seed(42)
tf.random.set_seed(42)

ticker = "AAPL"

DATABASE_URL = "sqlite:///./returns.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

end_date = datetime.now().date()
start_date = end_date - timedelta(
    1095
)  # 3 years of data for robust training across market cycles

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
        print(f"Missing data found: {missing_data[missing_data > 0]}")
    else:
        print("No missing data found")

    print("\n Sample data:")
    print(stock_data.head())

    print(f"Data Validation: {len(stock_data)} days available for training")

except Exception as e:
    print(f"Error loading data: {e}")
    stock_data = pd.DataFrame()  # Initialize empty DataFrame on error

finally:
    session.close()

# Ensure stock_data is properly initialized
if "stock_data" not in locals() or stock_data.empty:
    raise ValueError(
        "Failed to load stock data. Check database connection and ticker symbol."
    )


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
    upper_band = sma + (std * num_std)  # +2 std dev
    lower_band = sma - (std * num_std)  # -2 std dev
    return upper_band, lower_band, sma


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

# Volatility features
stock_data["volatility_5"] = stock_data["daily_return"].rolling(window=5).std()
stock_data["volatility_20"] = stock_data["daily_return"].rolling(window=20).std()
stock_data["volatility_ratio"] = (
    stock_data["volatility_5"] / stock_data["volatility_20"]
)
stock_data["price_range_volatility"] = (
    stock_data["high"] - stock_data["low"]
) / stock_data["close"]

# volume features
stock_data["volume_sma_20"] = stock_data["volume"].rolling(window=20).mean()
stock_data["volume_ratio"] = stock_data["volume"] / stock_data["volume_sma_20"]
stock_data["volume_price_trend"] = (
    (stock_data["close"] * stock_data["volume"]).rolling(window=20).mean()
)
stock_data["on_balance_volume"] = (
    stock_data["volume"] * np.where(stock_data["daily_return"] > 0, 1, -1)
).cumsum()
stock_data["obv_ema"] = stock_data["on_balance_volume"].ewm(span=20).mean()

# Higher order return metrics
stock_data["returns_skewness"] = stock_data["daily_return"].rolling(window=20).skew()
stock_data["returns_kurtosis"] = stock_data["daily_return"].rolling(window=20).kurt()
stock_data["return_momentum_3"] = stock_data["close"].pct_change(periods=3)
stock_data["return_momentum_10"] = stock_data["close"].pct_change(periods=10)

# Momentum features
stock_data["momentum_2d"] = (stock_data["close"] / stock_data["close"].shift(2)) - 1
stock_data["momentum_5d"] = (stock_data["close"] / stock_data["close"].shift(5)) - 1
stock_data["momentum_10d"] = (stock_data["close"] / stock_data["close"].shift(10)) - 1
stock_data["momentum_20d"] = (stock_data["close"] / stock_data["close"].shift(20)) - 1

# Acceleration
stock_data["price_acceleration"] = stock_data["daily_return"].diff()
stock_data["momentum_persistence"] = (
    stock_data["daily_return"].rolling(window=5).apply(lambda x: (x > 0).sum()) / 5
)

# Support + Resistence
stock_data["resistance_20"] = stock_data["high"].rolling(window=20).max()
stock_data["support_20"] = stock_data["low"].rolling(window=20).min()
stock_data["price_vs_resistance"] = stock_data["close"] / stock_data["resistance_20"]
stock_data["price_vs_support"] = stock_data["close"] / stock_data["support_20"]

stock_data["trend_strength"] = (
    abs(stock_data["close"] - stock_data["sma_20"]) / stock_data["sma_20"]
)
stock_data["market_efficiency"] = (
    stock_data["daily_return"]
    .rolling(window=20)
    .apply(lambda x: abs(x.autocorr()) if len(x.dropna()) > 1 else 0)
)

#
stock_data["gap_up"] = np.where(
    stock_data["open"] > stock_data["close"].shift(1),
    (stock_data["open"] - stock_data["close"].shift(1)) / stock_data["close"].shift(1),
    0,
)
stock_data["gap_down"] = np.where(
    stock_data["open"] < stock_data["close"].shift(1),
    (stock_data["close"].shift(1) - stock_data["open"]) / stock_data["close"].shift(1),
    0,
)

print("Advanced features calculated!")

feature_columns = [
    # OHLCV
    "open",
    "high",
    "low",
    "close",
    "volume",
    # Price Features
    "high_low_pct",
    "open_close_pct",
    "daily_return",
    "price_position",
    # Moving Averages
    "sma_5",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    # Technical Indicators
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_lower",
    "bb_middle",
    "bb_position",
    # Volatility Features
    "volatility_5",
    "volatility_20",
    "volatility_ratio",
    "price_range_volatility",
    # Volume Analysis
    "volume_ratio",
    "volume_price_trend",
    "on_balance_volume",
    "obv_ema",
    # HigherOrder Returns
    "returns_skewness",
    "returns_kurtosis",
    "return_momentum_3",
    "return_momentum_10",
    # Multi Timeframe Momentum
    "momentum_2d",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    # Price Dynamics
    "price_acceleration",
    "momentum_persistence",
    # Support + Resistance
    "resistance_20",
    "support_20",
    "price_vs_resistance",
    "price_vs_support",
    # Market Statistics
    "trend_strength",
    "market_efficiency",
    # Gap
    "gap_up",
    "gap_down",
]
print(f"Total Features pre cleaning: {len(feature_columns)}")


lookback_window = 180


print(f"Total data points pre-cleaning: {len(stock_data)}")

# Handle infinite values before dropping NaNs
numeric_columns = stock_data.select_dtypes(include=[np.number]).columns
stock_data[numeric_columns] = stock_data[numeric_columns].replace(
    [np.inf, -np.inf], np.nan
)

# Drop rows with NaN values
stock_data = stock_data.dropna()
print(f"Total data points post-cleaning: {len(stock_data)}")

# Verify we have enough data for training
min_required_samples = (
    lookback_window + 100
)  # Minimum for meaningful train/val/test split
if len(stock_data) < min_required_samples:
    raise ValueError(
        f"Insufficient data: {len(stock_data)} samples, need at least {min_required_samples}"
    )

feature_data = stock_data[feature_columns].values
print(f"Feature Matrix Shape: {feature_data.shape}")

# Final data quality checks
if np.isnan(feature_data).any():
    print(" ERROR: NaN values found in feature data after cleaning!")
    nan_cols = [
        feature_columns[i]
        for i in range(len(feature_columns))
        if np.isnan(feature_data[:, i]).any()
    ]
    print(f"Columns with NaN: {nan_cols}")
    raise ValueError("Data quality check failed - NaN values present")

if np.isinf(feature_data).any():
    print(" ERROR: Infinite values found in feature data after cleaning!")
    inf_cols = [
        feature_columns[i]
        for i in range(len(feature_columns))
        if np.isinf(feature_data[:, i]).any()
    ]
    print(f"Columns with infinite values: {inf_cols}")
    raise ValueError("Data quality check failed - Infinite values present")

print(" Data quality checks passed!")


# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(feature_data)
print(f"  All features: {scaled_features.min():.3f} to {scaled_features.max():.3f}")

# Create separate scaler for close price only (for inverse transformation)
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = close_scaler.fit_transform(stock_data[["close"]].values)

print("Scalers created:")
print(f"Main scaler: handles {len(feature_columns)} features")
print("Close scaler: handles close price only")

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


def create_time_series_split(X, y, train_ratio=0.7, validation_ratio=0.15):
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    validation_end = int(n_samples * (train_ratio + validation_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:validation_end]
    y_val = y[train_end:validation_end]
    X_test = X[validation_end:]
    y_test = y[validation_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Create Sequences with all features
X, y = create_sequences(scaled_features, lookback_window, target_col_idx=3)

print(f"Created {len(X)} training sequences")
print(f"X shape: {X.shape}")  # (samples, timesteps, features)
print(f"Y shape: {y.shape}")  # (samples,)
print(f"Features per timestep: {X.shape[2]}")
print(f"Lookback window:{X.shape[1]} days")

# Create time series split
(X_train, y_train), (X_val, y_val), (X_test, y_test) = create_time_series_split(X, y)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples: {len(X_test)}")

print("\n<?  BUILDING LSTM MODEL")
print("-" * 30)


def build_enhanced_model(input_shape, lstm_units=[128, 64, 32], dropout=0.3):
    """
    Enhanced LSTM architecture for better pattern recognition
    with more features and longer sequences
    """
    model = Sequential()

    # First LSTM layer
    # Large capacity for feature extraction
    model.add(
        LSTM(
            units=lstm_units[0],
            return_sequences=True,
            input_shape=input_shape,
            recurrent_dropout=0.1,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    # Second LSTM layer
    # Pattern refinement
    model.add(
        LSTM(
            units=lstm_units[1],
            return_sequences=True,
            recurrent_dropout=0.1,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    # Third LSTM layer
    # Final pattern extraction
    model.add(
        LSTM(
            units=lstm_units[2],
            return_sequences=False,
            recurrent_dropout=0.1,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    # Dense layers
    model.add(
        Dense(
            units=64,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(
        Dense(
            units=32,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(Dropout(0.1))

    # Final prediction layer
    model.add(Dense(units=1, activation="linear"))

    return model


model = build_enhanced_model(
    input_shape=(lookback_window, len(feature_columns)),
    lstm_units=[128, 64, 32],  # Larger architecture for better pattern recognition
    dropout=0.3,
)


def setup_enhanced_training_callbacks(model_save_path="enhanced_model.h5"):
    """Enhanced callbacks for overnight training with better monitoring"""
    callbacks = [
        # Early stopping with more patience for complex model
        EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,  # More aggressive reduction
            patience=10,
            min_lr=1e-8,
            verbose=1,
            mode="min",
            cooldown=3,  # Wait before next reduction
        ),
        # Model checkpointing
        ModelCheckpoint(
            model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min",
            save_weights_only=False,
        ),
        # Additional callback for directional accuracy monitoring
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"Epoch {epoch+1}: Training Loss: {logs['loss']:.6f}, Val Loss: {logs['val_loss']:.6f}"
            )
        ),
    ]

    return callbacks


# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss="huber",
    metrics=["mae", "mse"],
)
callbacks = setup_enhanced_training_callbacks()

print("Starting enhanced model training for overnight session...")
print(
    f"Model architecture: {len(feature_columns)} features, {lookback_window} day lookback"
)
print(
    f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}"
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    verbose=1,
    shuffle=False,
)


def evaluate_model(y_true, y_pred, prices_true, prices_pred):
    mse = mean_squared_error(prices_true, prices_pred)
    mae = mean_absolute_error(prices_true, prices_pred)
    rmse = np.sqrt(mse)

    # Financial specific metrics
    y_true_direction = np.diff(prices_true.flatten()) > 0
    y_pred_direction = np.diff(prices_pred.flatten()) > 0
    directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100

    # Sharpe Ratio
    actual_returns = np.diff(prices_true.flatten()) / prices_true[:-1].flatten()
    predicted_returns = np.diff(prices_pred.flatten()) / prices_true[:-1].flatten()
    if np.std(predicted_returns) > 0:
        sharpe_ratio = (
            np.mean(predicted_returns) / np.std(predicted_returns) * np.sqrt(252)
        )
    else:
        sharpe_ratio = 0

    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + predicted_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100

    results = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "avg_percentage_error": (mae / np.mean(prices_true)) * 100,
    }

    return results


# Make predictions on test data
y_pred = model.predict(X_test)

# Transform predictions back to original scale using close_scaler
y_test_original = close_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_original = close_scaler.inverse_transform(y_pred)

results = evaluate_model(y_test, y_pred, y_test_original, y_pred_original)

print(
    f"Architecture: {len(feature_columns)} features, {lookback_window} day lookback, 3-layer LSTM"
)
print(f"Training completed with {len(history.history['loss'])} epochs")
print("PERFORMANCE METRICS:")
print(f"  RMSE: ${results['rmse']:.2f}")
print(f"  MAE: ${results['mae']:.2f}")
print(f"  Directional Accuracy: {results['directional_accuracy']:.1f}%")
print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {results['max_drawdown']:.1f}%")
print(f"  Avg Percentage Error: {results['avg_percentage_error']:.2f}%")

print("\nFEATURE SUMMARY:")
print(f"  Total features used: {len(feature_columns)}")
print(f"  New advanced features added: {len(feature_columns) - 22}")
print(f"  Lookback window: {lookback_window} days")

print("\nFILES SAVED:")
print("  Best model: enhanced_best_model.h5")
print(
    f"  Ready for ensemble approach: {'YES' if results['directional_accuracy'] > 52 and results['sharpe_ratio'] > -0.5 else 'NO - NEEDS IMPROVEMENT'}"
)
