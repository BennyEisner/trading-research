#!/usr/bin/env python3

"""
Neural network model builder and training utilities
"""

import tensorflow as tf

# TensorFlow/Keras imports
from tensorflow.keras import regularizers
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


class ModelBuilder:
    """Builds and configures neural network models with multi-head attention layer"""

    def __init__(self, config):
        self.config = config

    def build_lstm_model(self, input_shape):
        """Build advanced multi-branch LSTM model with attention"""
        inputs = Input(shape=input_shape)
        # Branch 1: Depp LSTM
        lstm_branch = LSTM(128, return_sequences=True)(inputs)
        lstm_branch = BatchNormalization()(lstm_branch)  # Keeps mean close to 0, std close to 1
        lstm_branch = Dropout(0.3)(lstm_branch)

        lstm_branch = LSTM(96, return_sequences=True)(lstm_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = Dropout(0.3)(lstm_branch)

        lstm_branch = LSTM(64, return_sequences=True)(lstm_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = Dropout(0.3)(lstm_branch)

        # Branch 2: Forward GRU
        # Applies a Gated Reccurent Unit in only one direction to prevent lookahead bias
        gru_branch = GRU(128, return_sequence=True)(inputs)
        gru_branch = BatchNormalization()(gru_branch)
        gru_branch = Dropout(0.3)(gru_branch)

        # Branch 3: Conv1D
        # Applies a convolution operation along a single dimension
        # Detects short term trends, local correlations, etc
        conv_branch = Conv1D(64, kernel_size=3, activation="relu")(inputs)
        conv_branch = Conv1D(32, kernel_size=3, activation="relu")(conv_branch)
        conv_branch = GlobalMaxPooling1D()(conv_branch)  # Takes max value across all time steps/feature
        conv_branch = Reshape((-1, -1))(conv_branch)  # Changes tensor dimensions without changing data
        conv_branch = Lambda(lambda x: tf.tile(x, [1, input_shape[0], 1]))(conv_branch)

        # Combine Branches
        combined = Concatenate(axis=-1)([lstm_branch, gru_branch, conv_branch])

        # Multi-head attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=32)(combined, combined)
        attention = LayerNormalization()(attention)
        attention = Dropout(0.2)(attention)

        # Final Proccessing
        final_lstm = LSTM(32, return_sequences=False)(attention)
        final_lstm = BatchNormalization()(final_lstm)
        final_lstm = Dropout(0.2)(final_lstm)

        # Dense Prediction layers
        dense = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(final_lstm)
        dense = Dropout(0.4)(dense)
        dense = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01))(dense)
        dense = Dropout(0.4)(dense)
        dense = Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.01))(dense)
        dense = Dropout(0.3)(dense)

        outputs = Dense(1, activation="linear")(dense)

        model = Model(inputs=inputs, outputs=outputs)
        return model
