#!/usr/bin/env python3

"""
Neural network model builder and training utilities
Requires Python 3.12+ and TensorFlow 2.18+
"""
import sys
import time

if sys.version_info < (3, 12):
    raise RuntimeError("This module requires Python 3.12 or higher")

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class ModelBuilder:
    """Builds and configures neural network models with multi-head attention layer"""

    def __init__(self, config):
        self.config = config

    def build_lstm_model(self, input_shape):
        """Build configurable LSTM model based on config parameters"""
        inputs = Input(shape=input_shape)
        
        # Get configuration parameters
        lstm_units = self.config.get("lstm_units", [64, 32])
        enable_attention = self.config.get("enable_attention", False)
        dropout_rate = self.config.get("dropout_rate", 0.3)
        
        # Build LSTM layers according to config
        x = inputs
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1) or enable_attention
            x = LSTM(units, return_sequences=return_sequences)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        # Add attention if enabled
        if enable_attention:
            # Add attention mechanism
            attention = MultiHeadAttention(num_heads=2, key_dim=min(lstm_units[-1], 32))(x, x)
            attention = LayerNormalization()(attention)
            attention = Dropout(0.2)(attention)
            
            # Final LSTM to flatten for dense layers
            x = LSTM(lstm_units[-1], return_sequences=False)(attention)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        # Dense prediction layers
        dense_units = self.config.get("dense_units", [32, 16])
        for units in dense_units:
            x = Dense(units, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
            x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(1, activation="linear")(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def directional_loss(alpha=0.2):
        """Custom loss combining MSE with directional accuracy"""

        def loss(y_true, y_pred):
            # Standard MSE for magnitude accuracy
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

            # Directional penalty
            y_true_sign = tf.sign(y_true)
            y_pred_sign = tf.sign(y_pred)
            directional_error = tf.cast(tf.not_equal(y_true_sign, y_pred_sign), tf.float32)
            directional_penalty = tf.reduce_mean(directional_error)

            return (1 - alpha) * mse_loss + alpha * directional_penalty

        return loss

    @staticmethod
    def directional_accuracy(y_true, y_pred):
        """Calculate directional accuracy metric"""
        abs_threshold = 1e-4
        meaningful_mask = tf.greater(tf.abs(y_true), abs_threshold)

        y_true_filtered = tf.boolean_mask(y_true, meaningful_mask)
        y_pred_filtered = tf.boolean_mask(y_pred, meaningful_mask)

        y_true_sign = tf.sign(y_true_filtered)
        y_pred_sign = tf.sign(y_pred_filtered)

        num_meaningful = tf.reduce_sum(tf.cast(meaningful_mask, tf.float32))

        accuracy = tf.cond(
            tf.greater(num_meaningful, 0),
            lambda: tf.reduce_mean(tf.cast(tf.equal(y_true_sign, y_pred_sign), tf.float32)),
            lambda: 0.5,
        )
        return accuracy

    @staticmethod
    def learning_rate_schedule(epoch, lr):
        """Custom learning rate schedule"""
        initial_lr = 0.002

        # Warmup phase (first 10 epochs)
        if epoch < 10:
            return initial_lr * (epoch + 1) / 10

        # Decay phases
        if epoch < 50:
            return initial_lr * 0.95 ** ((epoch - 10) // 5)
        elif epoch < 100:
            return initial_lr * 0.5 * 0.90 ** ((epoch - 50) // 10)
        elif epoch < 150:
            return initial_lr * 0.25 * 0.85 ** ((epoch - 100) // 15)
        else:
            return initial_lr * 0.1 * 0.80 ** ((epoch - 150) // 20)

    def compile_model(self, model, learning_rate=None):
        """Compile model with custom loss and metrics"""
        if learning_rate is None:
            learning_rate = self.config.get("learning_rate", 0.002)

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=self.directional_loss(alpha=0.2),
            metrics=[self.directional_accuracy, "mae", "mse"],
        )
        return model

    def get_callbacks(self, model_save_path="best_model.h5", logger=None):
        """Get training callbacks including learning rate scheduler"""
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=self.config.get("early_stopping_patience", 80), restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=1, save_weights_only=False),
            LearningRateScheduler(self.learning_rate_schedule, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-7, verbose=1),
        ]

        # Add custom progress callback if logger provided
        if logger:
            callbacks.append(ProgressCallback(logger, "Training", self.config.get("epochs", 500)))

        return callbacks

    def train_model(self, model, X_train, y_train, X_val, y_val, callbacks=None, logger=None):
        """Train the model with given data"""
        if callbacks is None:
            callbacks = self.get_callbacks(logger=logger)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get("epochs", 500),
            batch_size=self.config.get("batch_size", 32),
            callbacks=callbacks,
            verbose=1,
            shuffle=False,  # Important for time series data
        )
        return history


class ProgressCallback(Callback):
    """Custom Callback for detailed training progress"""

    def __init__(self, logger, phase_name, total_epochs):
        super().__init__()
        self.logger = logger
        self.phase_name = phase_name
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        self.training_start_time = None

    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Fix: Ensure logs is not None
        logs = logs or {}

        epoch_time = time.time() - self.epoch_start_time
        elapsed_total = time.time() - self.training_start_time

        # Compute ETA
        epochs_remaining = self.total_epochs - (epoch + 1)
        avg_epoch_time = elapsed_total / (epoch + 1)
        eta_seconds = epochs_remaining * avg_epoch_time
        eta_minutes = eta_seconds / 60

        # Extract metrics
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        dir_acc = logs.get("directional_accuracy", 0)
        val_dir_acc = logs.get("val_directional_accuracy", 0)
        mae = logs.get("mae", 0)
        val_mae = logs.get("val_mae", 0)
        lr = logs.get("lr", logs.get("learning_rate", 0.002))

        # Progress bar
        progress_pct = ((epoch + 1) / self.total_epochs) * 100
        bar_length = 30
        filled_length = int(bar_length * (epoch + 1) // self.total_epochs)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        val_metrics = ""
        if val_loss > 0:
            val_metrics = f" | Val Loss: {val_loss:.4f} | Val Dir: {val_dir_acc:.1%} | Val MAE: {val_mae:.4f}"

        progress_msg = (
            f"[{bar}] {progress_pct:.1f}% "
            f"Epoch {epoch+1}/{self.total_epochs} ({epoch_time:.1f}s) | "
            f"Loss: {loss:.4f} | Dir: {dir_acc:.1%} | MAE: {mae:.4f}"
            f"{val_metrics} | LR: {lr:.2e} | ETA: {eta_minutes:.1f}min"
        )

        self.logger.log(progress_msg)
