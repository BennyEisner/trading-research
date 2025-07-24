#!/usr/bin/env python3

"""
Multi-Scale LSTM Architecture for Financial Time Series Prediction
Requires Python 3.12+ and TensorFlow 2.18+
"""

import sys
if sys.version_info < (3, 12):
    raise RuntimeError("This module requires Python 3.12 or higher")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List, Dict, Any


class MultiScaleLSTMBuilder:
    """
    Advanced multi-scale LSTM architecture that processes multiple time horizons
    with attention mechanisms for financial time series prediction
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def build_multi_scale_model(self, input_shape: Tuple[int, int], **params) -> keras.Model:
        """
        Build multi-scale LSTM model with three time horizons and attention
        
        Args:
            input_shape: (sequence_length, features)
            **params: Model hyperparameters
        """
        
        sequence_length, n_features = input_shape
        
        # Model hyperparameters
        lstm_units_1 = params.get('lstm_units_1', 512)
        lstm_units_2 = params.get('lstm_units_2', 256)
        lstm_units_3 = params.get('lstm_units_3', 128)
        dropout_rate = params.get('dropout_rate', 0.3)
        l2_reg = params.get('l2_regularization', 0.003)
        use_attention = params.get('use_attention', True)
        dense_layers = params.get('dense_layers', [256, 128, 64])
        
        # Input layer
        main_input = layers.Input(shape=input_shape, name='main_input')
        
        # ====================================================================
        # MULTI-SCALE PROCESSING BRANCHES
        # ====================================================================
        
        # Branch 1: Short-term patterns (10 days)
        short_term_input = layers.Lambda(
            lambda x: x[:, -10:, :],
            name='short_term_slice'
        )(main_input)
        
        short_term_lstm = layers.LSTM(
            lstm_units_3,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_regularizer=keras.regularizers.l2(l2_reg),
            name='short_term_lstm'
        )(short_term_input)
        short_term_lstm = layers.Dropout(dropout_rate, name='short_term_dropout')(short_term_lstm)
        
        # Branch 2: Medium-term patterns (30 days) - Full sequence
        medium_term_lstm = layers.LSTM(
            lstm_units_1,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_regularizer=keras.regularizers.l2(l2_reg),
            name='medium_term_lstm_1'
        )(main_input)
        medium_term_lstm = layers.Dropout(dropout_rate, name='medium_term_dropout_1')(medium_term_lstm)
        
        medium_term_lstm = layers.LSTM(
            lstm_units_2,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_regularizer=keras.regularizers.l2(l2_reg),
            name='medium_term_lstm_2'
        )(medium_term_lstm)
        medium_term_lstm = layers.Dropout(dropout_rate, name='medium_term_dropout_2')(medium_term_lstm)
        
        # Branch 3: Long-term patterns (subsampled to capture longer trends)
        long_term_input = layers.Lambda(
            lambda x: x[:, ::2, :],  # Take every 2nd timestep
            name='long_term_subsample'
        )(main_input)
        
        long_term_lstm = layers.LSTM(
            lstm_units_3,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_regularizer=keras.regularizers.l2(l2_reg),
            name='long_term_lstm'
        )(long_term_input)
        long_term_lstm = layers.Dropout(dropout_rate, name='long_term_dropout')(long_term_lstm)
        
        # ====================================================================
        # ATTENTION MECHANISMS
        # ====================================================================
        
        if use_attention:
            # Self-attention for each branch
            short_term_att = layers.MultiHeadAttention(
                num_heads=4, 
                key_dim=lstm_units_3//4,
                name='short_term_attention'
            )(short_term_lstm, short_term_lstm)
            
            medium_term_att = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=lstm_units_2//8, 
                name='medium_term_attention'
            )(medium_term_lstm, medium_term_lstm)
            
            long_term_att = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=lstm_units_3//4,
                name='long_term_attention'
            )(long_term_lstm, long_term_lstm)
            
            # Global max pooling to get fixed-size representations
            short_term_pooled = layers.GlobalAveragePooling1D(name='short_term_pool')(short_term_att)
            medium_term_pooled = layers.GlobalAveragePooling1D(name='medium_term_pool')(medium_term_att)
            long_term_pooled = layers.GlobalAveragePooling1D(name='long_term_pool')(long_term_att)
            
        else:
            # Without attention, just use the last output
            short_term_pooled = layers.GlobalAveragePooling1D(name='short_term_pool')(short_term_lstm)
            medium_term_pooled = layers.GlobalAveragePooling1D(name='medium_term_pool')(medium_term_lstm)
            long_term_pooled = layers.GlobalAveragePooling1D(name='long_term_pool')(long_term_lstm)
        
        # ====================================================================
        # FEATURE FUSION WITH CROSS-ATTENTION
        # ====================================================================
        
        # Concatenate all time-scale features
        multi_scale_features = layers.Concatenate(name='multi_scale_concat')([
            short_term_pooled,
            medium_term_pooled, 
            long_term_pooled
        ])
        
        # Cross-scale attention to learn interactions between time horizons
        if use_attention:
            # Create learnable cross-attention instead of reshaping
            # Apply dense layer to get proper dimensions for attention
            attention_input = layers.Dense(384, activation='relu', name='attention_prep')(multi_scale_features)
            attention_reshaped = layers.Reshape((3, 128), name='reshape_for_cross_attention')(attention_input)
            
            cross_attention = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=32,
                name='cross_scale_attention'
            )(attention_reshaped, attention_reshaped)
            
            # Flatten back and project to original dimension
            cross_attended = layers.Flatten(name='flatten_cross_attention')(cross_attention)
            cross_projected = layers.Dense(multi_scale_features.shape[-1], name='cross_attention_projection')(cross_attended)
            
            # Residual connection
            fused_features = layers.Add(name='residual_connection')([multi_scale_features, cross_projected])
            fused_features = layers.LayerNormalization(name='fusion_layer_norm')(fused_features)
        else:
            fused_features = multi_scale_features
        
        # ====================================================================
        # DEEP DENSE LAYERS FOR FINAL PREDICTION
        # ====================================================================
        
        x = fused_features
        
        for i, units in enumerate(dense_layers):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = layers.Dropout(dropout_rate / 2, name=f'dense_dropout_{i+1}')(x)
            x = layers.LayerNormalization(name=f'dense_layer_norm_{i+1}')(x)
        
        # Final prediction layer
        output = layers.Dense(1, name='prediction_output')(x)
        
        # ====================================================================
        # CREATE AND COMPILE MODEL
        # ====================================================================
        
        model = keras.Model(inputs=main_input, outputs=output, name='MultiScaleLSTM')
        
        # Advanced optimizer with learning rate scheduling
        initial_lr = params.get('learning_rate', 0.001)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0  # Gradient clipping for stability
        )
        
        # Compile with custom metrics
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', self._directional_accuracy, self._weighted_directional_accuracy, self._correlation_metric]
        )
        
        return model
    
    def build_directional_focused_model(self, input_shape: Tuple[int, int], **params) -> keras.Model:
        """
        Build multi-scale model with directional loss and enhanced directional metrics
        """
        # Build base architecture
        model = self.build_multi_scale_model(input_shape, **params)
        
        # Directional loss function
        def directional_mse_loss(y_true, y_pred):
            alpha = params.get('directional_alpha', 0.4)  # Weight for directional component
            
            # MSE component
            mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
            
            # Directional component
            y_true_sign = tf.sign(y_true)
            y_pred_sign = tf.sign(y_pred)
            directional_error = 0.5 * (1 - y_true_sign * y_pred_sign)
            directional_loss = tf.reduce_mean(directional_error)
            
            return (1 - alpha) * mse_loss + alpha * directional_loss
        
        # Enhanced optimizer for directional learning
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.get('learning_rate', 0.0005),  # Lower LR for stable directional learning
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        
        # Compile with directional focus
        model.compile(
            optimizer=optimizer,
            loss=directional_mse_loss,
            metrics=[
                'mae',
                self._directional_accuracy,
                self._weighted_directional_accuracy,
                self._up_down_accuracy,
                self._correlation_metric
            ]
        )
        
        return model
    
    @staticmethod
    def _up_down_accuracy(y_true, y_pred):
        """Custom metric: Balanced accuracy for up vs down moves"""
        y_true_sign = tf.sign(y_true)
        y_pred_sign = tf.sign(y_pred)
        
        # Up moves accuracy
        up_mask = y_true > 0
        up_correct = tf.logical_and(up_mask, tf.equal(y_true_sign, y_pred_sign))
        up_total = tf.reduce_sum(tf.cast(up_mask, tf.float32))
        up_accuracy = tf.reduce_sum(tf.cast(up_correct, tf.float32)) / (up_total + 1e-8)
        
        # Down moves accuracy  
        down_mask = y_true < 0
        down_correct = tf.logical_and(down_mask, tf.equal(y_true_sign, y_pred_sign))
        down_total = tf.reduce_sum(tf.cast(down_mask, tf.float32))
        down_accuracy = tf.reduce_sum(tf.cast(down_correct, tf.float32)) / (down_total + 1e-8)
        
        # Return balanced accuracy
        return (up_accuracy + down_accuracy) / 2.0
    
    def build_ensemble_model(self, input_shape: Tuple[int, int], **params) -> keras.Model:
        """
        Build ensemble model combining LSTM with CNN-1D and Transformer components
        """
        
        main_input = layers.Input(shape=input_shape, name='ensemble_input')
        
        # LSTM branch (simplified multi-scale)
        lstm_branch = self._build_lstm_branch(main_input, params)
        
        # CNN-1D branch for local patterns
        cnn_branch = self._build_cnn_branch(main_input, params)
        
        # Transformer branch for long-range dependencies
        transformer_branch = self._build_transformer_branch(main_input, params)
        
        # Combine all branches
        combined = layers.Concatenate(name='ensemble_concat')([
            lstm_branch,
            cnn_branch,
            transformer_branch
        ])
        
        # Final layers
        x = layers.Dense(256, activation='relu', name='ensemble_dense_1')(combined)
        x = layers.Dropout(0.3, name='ensemble_dropout_1')(x)
        x = layers.Dense(128, activation='relu', name='ensemble_dense_2')(x)
        x = layers.Dropout(0.2, name='ensemble_dropout_2')(x)
        
        output = layers.Dense(1, name='ensemble_output')(x)
        
        model = keras.Model(inputs=main_input, outputs=output, name='EnsembleLSTM')
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001))
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', self._directional_accuracy]
        )
        
        return model
    
    def _build_lstm_branch(self, input_tensor, params) -> layers.Layer:
        """Build LSTM branch for ensemble"""
        x = layers.LSTM(256, return_sequences=True)(input_tensor)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(128)(x)
        x = layers.Dense(64, activation='relu')(x)
        return x
    
    def _build_cnn_branch(self, input_tensor, params) -> layers.Layer:
        """Build CNN-1D branch for ensemble"""
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(input_tensor)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        return x
    
    def _build_transformer_branch(self, input_tensor, params) -> layers.Layer:
        """Build Transformer branch for ensemble"""
        # Simple transformer encoder
        attention = layers.MultiHeadAttention(num_heads=8, key_dim=32)(input_tensor, input_tensor)
        attention = layers.Dropout(0.2)(attention)
        
        # Add & Norm
        x = layers.Add()([input_tensor, attention])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff = layers.Dense(256, activation='relu')(x)
        ff = layers.Dropout(0.2)(ff)
        ff = layers.Dense(input_tensor.shape[-1])(ff)
        
        # Add & Norm
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        return x
    
    def create_learning_rate_schedule(self, initial_lr: float = 0.001) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """Create cosine annealing learning rate schedule"""
        
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=1000,  # Adjust based on training steps
            alpha=0.1  # Minimum learning rate factor
        )
    
    @staticmethod
    def _directional_accuracy(y_true, y_pred):
        """Custom metric: Directional accuracy (percentage of correct direction predictions)"""
        y_true_sign = tf.sign(y_true)
        y_pred_sign = tf.sign(y_pred)
        correct_directions = tf.equal(y_true_sign, y_pred_sign)
        return tf.reduce_mean(tf.cast(correct_directions, tf.float32))
    
    @staticmethod
    def _weighted_directional_accuracy(y_true, y_pred):
        """Custom metric: Directional accuracy weighted by magnitude"""
        y_true_sign = tf.sign(y_true)
        y_pred_sign = tf.sign(y_pred)
        correct_directions = tf.cast(tf.equal(y_true_sign, y_pred_sign), tf.float32)
        
        # Weight by absolute magnitude of true returns
        weights = tf.abs(y_true)
        weights = weights / (tf.reduce_mean(weights) + 1e-8)  # Normalize weights
        
        weighted_accuracy = tf.reduce_sum(correct_directions * weights) / tf.reduce_sum(weights)
        return weighted_accuracy
    
    @staticmethod
    def _correlation_metric(y_true, y_pred):
        """Custom metric: Pearson correlation coefficient"""
        x = y_true - tf.reduce_mean(y_true)
        y = y_pred - tf.reduce_mean(y_pred)
        
        numerator = tf.reduce_sum(x * y)
        denominator = tf.sqrt(tf.reduce_sum(x**2) * tf.reduce_sum(y**2))
        
        correlation = numerator / (denominator + 1e-8)
        return correlation
    
    def get_model_summary_stats(self, model: keras.Model) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'layers': len(model.layers),
            'architecture_type': 'MultiScaleLSTM'
        }