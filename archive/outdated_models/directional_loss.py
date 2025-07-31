#!/usr/bin/env python3

"""
Directional-focused loss functions for financial time series prediction
Requires Python 3.12+ and TensorFlow 2.18+
"""

import sys

if sys.version_info < (3, 12):
    raise RuntimeError("This module requires Python 3.12 or higher")

from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras


class DirectionalLoss:
    """
    Custom loss functions that emphasize directional accuracy
    """

    @staticmethod
    def directional_mse_loss(y_true, y_pred, alpha=0.5):
        """
        Combined MSE + Directional loss

        Args:
            y_true: True returns
            y_pred: Predicted returns
            alpha: Weight between MSE (1-alpha) and directional (alpha) components
        """
        # Standard MSE component
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

        # Directional component - penalize wrong directions more heavily
        y_true_sign = tf.sign(y_true)
        y_pred_sign = tf.sign(y_pred)

        # Directional error: 0 if same sign, 1 if different sign
        directional_error = 0.5 * (1 - y_true_sign * y_pred_sign)
        directional_loss = tf.reduce_mean(directional_error)

        # Combined loss
        return (1 - alpha) * mse_loss + alpha * directional_loss

    @staticmethod
    def asymmetric_directional_loss(y_true, y_pred, penalty_factor=2.0):
        """
        Asymmetric loss that heavily penalizes directional errors

        Args:
            y_true: True returns
            y_pred: Predicted returns
            penalty_factor: How much more to penalize wrong directions
        """
        # Calculate squared error
        squared_error = tf.square(y_true - y_pred)

        # Check if prediction direction is wrong
        same_direction = y_true * y_pred >= 0

        # Apply penalty multiplier for wrong directions
        penalty_multiplier = tf.where(
            same_direction,
            1.0,  # Normal penalty for correct direction
            penalty_factor,  # Higher penalty for wrong direction
        )

        return tf.reduce_mean(squared_error * penalty_multiplier)

    @staticmethod
    def sign_prediction_loss(y_true, y_pred):
        """
        Pure sign prediction loss - only cares about direction
        """
        y_true_sign = tf.sign(y_true)
        y_pred_sign = tf.sign(y_pred)

        # Binary cross-entropy style loss for sign prediction
        # Convert signs to probabilities: +1 -> 1, -1 -> 0
        y_true_prob = (y_true_sign + 1) / 2
        y_pred_prob = (y_pred_sign + 1) / 2

        # Clip to avoid log(0)
        y_pred_prob = tf.clip_by_value(y_pred_prob, 1e-7, 1 - 1e-7)

        return -tf.reduce_mean(
            y_true_prob * tf.math.log(y_pred_prob) + (1 - y_true_prob) * tf.math.log(1 - y_pred_prob)
        )

    @staticmethod
    def quantile_directional_loss(y_true, y_pred, quantiles=[0.1, 0.5, 0.9]):
        """
        Quantile regression loss with directional emphasis
        """
        losses = []

        for q in quantiles:
            error = y_true - y_pred
            # Asymmetric quantile loss
            quantile_loss = tf.maximum(q * error, (q - 1) * error)

            # Add directional penalty
            same_direction = y_true * y_pred >= 0
            directional_penalty = tf.where(same_direction, 0.0, 0.5)

            losses.append(quantile_loss + directional_penalty)

        return tf.reduce_mean(tf.stack(losses))


class DirectionalMetrics:
    """
    Enhanced metrics for tracking directional prediction performance
    """

    @staticmethod
    def directional_accuracy(y_true, y_pred):
        """Standard directional accuracy metric"""
        y_true_sign = tf.sign(y_true)
        y_pred_sign = tf.sign(y_pred)
        correct_directions = tf.equal(y_true_sign, y_pred_sign)
        return tf.reduce_mean(tf.cast(correct_directions, tf.float32))

    @staticmethod
    def weighted_directional_accuracy(y_true, y_pred):
        """
        Directional accuracy weighted by magnitude of true returns
        (more important to get direction right for larger moves)
        """
        y_true_sign = tf.sign(y_true)
        y_pred_sign = tf.sign(y_pred)
        correct_directions = tf.cast(tf.equal(y_true_sign, y_pred_sign), tf.float32)

        # Weight by absolute magnitude of true returns
        weights = tf.abs(y_true)
        weights = weights / (tf.reduce_mean(weights) + 1e-8)  # Normalize weights

        weighted_accuracy = tf.reduce_sum(correct_directions * weights) / tf.reduce_sum(weights)
        return weighted_accuracy

    @staticmethod
    def up_down_accuracy(y_true, y_pred):
        """
        Separate accuracy for up moves vs down moves
        """
        y_true_sign = tf.sign(y_true)
        y_pred_sign = tf.sign(y_pred)

        # Up moves (positive returns)
        up_mask = y_true > 0
        up_correct = tf.logical_and(up_mask, tf.equal(y_true_sign, y_pred_sign))
        up_total = tf.reduce_sum(tf.cast(up_mask, tf.float32))
        up_accuracy = tf.reduce_sum(tf.cast(up_correct, tf.float32)) / (up_total + 1e-8)

        # Down moves (negative returns)
        down_mask = y_true < 0
        down_correct = tf.logical_and(down_mask, tf.equal(y_true_sign, y_pred_sign))
        down_total = tf.reduce_sum(tf.cast(down_mask, tf.float32))
        down_accuracy = tf.reduce_sum(tf.cast(down_correct, tf.float32)) / (down_total + 1e-8)

        # Return balanced accuracy
        return (up_accuracy + down_accuracy) / 2.0

    @staticmethod
    def profit_loss_accuracy(y_true, y_pred):
        """
        Simulated P&L based on direction prediction
        (assumes we trade based on predicted direction)
        """
        # Trading signal: +1 if predict up, -1 if predict down
        trade_signal = tf.sign(y_pred)

        # P&L = signal * actual_return
        pnl = trade_signal * y_true

        # Return average P&L (should be positive if directionally accurate)
        return tf.reduce_mean(pnl)


class DirectionalModelBuilder:
    """
    Enhanced model builder with directional focus
    """

    def __init__(self, base_builder):
        self.base_builder = base_builder

    def build_directional_model(self, input_shape, loss_type="directional_mse", **params):
        """
        Build model optimized for directional prediction

        Args:
            input_shape: Input shape for model
            loss_type: Type of directional loss to use
            **params: Model parameters
        """

        # Build base multi-scale model
        model = self.base_builder.build_multi_scale_model(input_shape, **params)

        # Choose directional loss function
        loss_functions = {
            "directional_mse": lambda y_true, y_pred: DirectionalLoss.directional_mse_loss(
                y_true, y_pred, alpha=params.get("directional_alpha", 0.3)
            ),
            "asymmetric": lambda y_true, y_pred: DirectionalLoss.asymmetric_directional_loss(
                y_true, y_pred, penalty_factor=params.get("penalty_factor", 2.0)
            ),
            "sign_only": DirectionalLoss.sign_prediction_loss,
            "quantile": DirectionalLoss.quantile_directional_loss,
        }

        loss_fn = loss_functions.get(loss_type, "directional_mse")

        # Enhanced metrics focused on direction
        metrics = [
            "mae",
            DirectionalMetrics.directional_accuracy,
            DirectionalMetrics.weighted_directional_accuracy,
            DirectionalMetrics.up_down_accuracy,
            DirectionalMetrics.profit_loss_accuracy,
        ]

        # Compile with directional focus
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.get("learning_rate", 0.001), beta_1=0.9, beta_2=0.999, clipnorm=1.0
        )

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        return model

    def get_directional_callbacks(self, **params):
        """
        Get callbacks optimized for directional prediction
        """
        callbacks = [
            # Early stopping based on directional accuracy
            tf.keras.callbacks.EarlyStopping(
                monitor="val_directional_accuracy",
                patience=params.get("patience", 10),
                restore_best_weights=True,
                mode="max",  # Higher directional accuracy is better
                verbose=1,
            ),
            # Reduce learning rate when directional accuracy plateaus
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_directional_accuracy", factor=0.5, patience=5, mode="max", verbose=1, min_lr=1e-6
            ),
            # Model checkpoint based on best directional accuracy
            tf.keras.callbacks.ModelCheckpoint(
                filepath="models/trained/best_directional_model.keras",
                monitor="val_directional_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
        ]

        return callbacks


def create_directional_training_config():
    """
    Create training configuration optimized for directional prediction
    """
    return {
        "loss_type": "directional_mse",  # Balanced MSE + directional loss
        "directional_alpha": 0.4,  # 40% weight on directional component
        "penalty_factor": 2.5,  # 2.5x penalty for wrong directions
        "learning_rate": 0.0005,  # Slightly lower LR for stable directional learning
        "batch_size": 64,  # Smaller batches for better gradient estimates
        "epochs": 100,
        "patience": 15,  # More patience since directional learning is slower
        # Model architecture adjustments
        "dropout_rate": 0.4,  # Higher dropout to prevent overfitting to magnitude
        "l2_regularization": 0.005,  # Stronger regularization
        # Additional training strategies
        "use_class_weights": True,  # Weight up/down classes if imbalanced
        "gradient_clipping": True,
        "mixed_precision": False,  # Disable for stability in directional prediction
    }

