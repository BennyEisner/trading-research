#!/usr/bin/env python3

"""
Pattern focused LSTM Architecture
Simplifies Multi_scale_lstm.py drastically with intention of complementing technical indicator strategies by focusing on temporal patterns and non linear relationships
"""

import sys
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PatternFocusedLSTMBuilder:
    """LSTM Architecture focused on pattern detection for ensemble integration
    - Single Branch for simplicity
    - ~500k parameters
    - Pattern Detection Focused
    - Bounded output of -1 to 1 for voted weighting
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build_model(self, input_shape: Tuple[int, int], **params) -> keras.Model:
        """Build pattern focused LSTM model

        Args:
            input_shape: (sequence_length, n_features) - expected ~(30, 12)
            **params: Model hyperparameters

        Returns:
            Compiled keras model with tanh output
        """

        sequence_length, n_features = input_shape

        # Model hyperparameters
        lstm_units_1 = params.get("lstm_units_1", 96)
        lstm_units_2 = params.get("lstm_units_2", 48)
        dropout_rate = params.get("dropout_rate", 0.3)
        l2_reg = params.get("l2_regularization", 0.003)

        # Input Layer
        main_input = layers.Input(shape=input_shape, name="pattern_input")

        x = layers.LSTM(
            lstm_units_1,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_regularizer=keras.regularizers.l2(l2_reg),
            name="pattern_lstm_1",
        )(main_input)
        x = layers.Dropout(dropout_rate, name="pattern_dropout_1")(x)
        x = layers.LSTM(
            lstm_units_2,
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_regularizer=keras.regularizers.l2(l2_reg),
            name="pattern_lstm_2",
        )(x)

        # Single dense layer for pattern interpretation
        x = layers.Dense(24, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg), name="pattern_dense")(x)
        x = layers.Dropout(dropout_rate / 2, name="pattern_dropout_3")(x)

        # Bound output between -1 1 via tanh
        output = layers.Dense(
            1, activation="tanh", name="pattern_prediction", kernel_initializer="zeros", bias_initializer="zeros"
        )(x)

        model = keras.Model(inputs=main_input, outputs=output, name="PatternFocusedLSTM")

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.get("learning_rate", 0.001), beta_1=0.9, beta_2=0.999, epsilon=1e-7, clipnorm=1.0
        )

        model.compile(
            optimizer=optimizer, loss="mse", metrics=["mae", self._directional_accuracy, self._correlation_metric]
        )

        return model

    def _directional_accuracy(self, y_true, y_pred):
        """Calculate directional accuracy metric"""
        return tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true), tf.sign(y_pred)), tf.float32))

    def _correlation_metric(self, y_true, y_pred):
        """Calculate correlation coefficient metric"""

        # Center predictions and targets
        y_true_centered = y_true - tf.reduce_mean(y_true)
        y_pred_centered = y_pred - tf.reduce_mean(y_pred)

        numerator = tf.reduce_sum(y_true_centered * y_pred_centered)
        denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered)) * tf.reduce_sum(tf.square(y_pred_centered)))

        correlation = tf.cond(tf.equal(denominator, 0.0), lambda: 0.0, lambda: numerator / denominator)

        return correlation

    def get_model_summary(self, input_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Get model architecture summary"""

        model = self.build_model(input_shape)

        # Count params
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

        return {
            "model_name": "PatternFocusedLSTM",
            "input_shape": input_shape,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
            "architecture_type": "simplified_pattern_focused",
            "output_activation": "tanh",
            "output_range": "(-1, 1)",
            "designed_for": "ensemble_integration",
        }


class MultiScaleLSTMBuilder(PatternFocusedLSTMBuilder):
    """Backward compatibility wrapper whuile new LSTM is still in progress"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        print("INFO: Using refactored pattern focused LSTM")
        print("INFO: Previous multi-scale complexity replaced with ensemble optimized")

    def build_optimized_model(self, input_shape: Tuple[int, int], **params) -> keras.Model:
        """Build optimized model"""
        return self.build_model(input_shape, **params)

    def build_efficient_multi_scale_model(self, input_shape: Tuple[int, int], **params) -> keras.Model:
        """Redirects to pattern focused model"""
        return self.build_model(input_shape, **params)

    def build_multi_scale_model(self, input_shape: Tuple[int, int], **params) -> keras.Model:
        """Redirects to pattern focused model"""
        return self.build_model(input_shape, **params)

    def create_pattern_training_config(self) -> Dict[str, Any]:
        """Get pattern training config from the main config system"""
        return {**self.config.model.model_params, **self.config.model.training_params}

    def test_pattern_model_creation(self): 
        try: 
            config = self.create_pattern_training_config()
            builder = PatternFocusedLSTMBuilder(config)
             
            input_shape = (30, 12)
            model = builder.build_model(input_shape, **config)
            print("Pattern focused lstm built")
            
            test_input = np.random.randn(1, 30, 12)
            prediction = model.predict(test_input, verbose=0)
            print(f"Test prediction: {prediction[0][0]:.6f} (range -1 to 1)")
            return True

        except Exception as e: 
            print(f"Error creating pattern focused model: {e}") 



if __name__ == "__main__": 
    from config.config import get_config
    
    # Use existing config system
    config = get_config()
    
    builder = MultiScaleLSTMBuilder(config.dict())
    builder.test_pattern_model_creation()

