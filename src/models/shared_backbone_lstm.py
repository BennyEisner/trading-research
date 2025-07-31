#!/usr/bin/env python3

"""
Shared Backbone LSTM Architecture
Enhanced regularization with shared learning across correlated securities
Designed for preventing overfitting in expanded universe training
"""

import sys
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SharedBackboneLSTMBuilder:
    """
    Enhanced LSTM architecture with shared backbone for cross-stock pattern learning
    - Increased regularization to prevent overfitting on expanded universe
    - Batch normalization for training stability
    - Shared pattern recognition layers
    - Stock-specific adaptation layers
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build_model(self, input_shape: Tuple[int, int], **params) -> keras.Model:
        """Build enhanced pattern focused LSTM with shared backbone
        
        Args:
            input_shape: (sequence_length, n_features) - expected ~(20, 17) for swing trading
            **params: Model hyperparameters
            
        Returns:
            Compiled keras model with enhanced regularization
        """
        
        sequence_length, n_features = input_shape
        
        # Enhanced regularization parameters
        lstm_units_1 = params.get("lstm_units_1", 64)  # Reduced from 96 to prevent overfitting
        lstm_units_2 = params.get("lstm_units_2", 32)  # Reduced from 48 to prevent overfitting
        dropout_rate = params.get("dropout_rate", 0.45)  # Increased from 0.3 to 0.45
        l2_reg = params.get("l2_regularization", 0.006)  # Increased from 0.003 to 0.006
        use_batch_norm = params.get("use_batch_norm", True)
        use_recurrent_dropout = params.get("use_recurrent_dropout", True)
        
        # Input Layer
        main_input = layers.Input(shape=input_shape, name="pattern_input")
        
        # Shared backbone for pattern recognition
        x = main_input
        
        # Optional input batch normalization
        if use_batch_norm:
            x = layers.BatchNormalization(name="input_batch_norm")(x)
        
        # First shared LSTM layer - pattern detection backbone
        x = layers.LSTM(
            lstm_units_1,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_dropout=0.2 if use_recurrent_dropout else 0.0,
            name="shared_pattern_lstm_1",
        )(x)
        
        # Batch normalization after first LSTM
        if use_batch_norm:
            x = layers.BatchNormalization(name="lstm1_batch_norm")(x)
        
        x = layers.Dropout(dropout_rate, name="shared_dropout_1")(x)
        
        # Second shared LSTM layer - temporal pattern consolidation
        x = layers.LSTM(
            lstm_units_2,
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_regularizer=keras.regularizers.l2(l2_reg),
            recurrent_dropout=0.1 if use_recurrent_dropout else 0.0,
            name="shared_pattern_lstm_2",
        )(x)
        
        # Batch normalization after second LSTM
        if use_batch_norm:
            x = layers.BatchNormalization(name="lstm2_batch_norm")(x)
        
        # Shared pattern interpretation layer
        x = layers.Dense(
            16,  # Reduced from 24 to prevent overfitting
            activation="relu", 
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name="shared_pattern_dense"
        )(x)
        
        if use_batch_norm:
            x = layers.BatchNormalization(name="dense_batch_norm")(x)
        
        x = layers.Dropout(dropout_rate * 0.8, name="shared_dropout_2")(x)
        
        # Final prediction layer with stronger regularization
        output = layers.Dense(
            1, 
            activation="tanh", 
            name="pattern_prediction",
            kernel_initializer="glorot_uniform",  # Better initialization
            bias_initializer="zeros",
            kernel_regularizer=keras.regularizers.l2(l2_reg * 1.5),  # Extra regularization on output
            activity_regularizer=keras.regularizers.l1(0.001)  # L1 for sparsity
        )(x)
        
        model = keras.Model(inputs=main_input, outputs=output, name="SharedBackboneLSTM")
        
        # Enhanced optimizer with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.get("learning_rate", 0.0008),  # Slightly reduced for stability
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=0.8  # Tighter gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae", self._pattern_detection_accuracy, self._correlation_metric]
        )
        
        return model

    def build_stock_specific_model(self, 
                                  shared_backbone: keras.Model,
                                  input_shape: Tuple[int, int], 
                                  stock_symbol: str,
                                  **params) -> keras.Model:
        """
        Build stock-specific model using shared backbone weights
        
        Args:
            shared_backbone: Pre-trained shared backbone model
            input_shape: Input shape for the specific stock
            stock_symbol: Stock symbol for naming
            **params: Model hyperparameters
            
        Returns:
            Stock-specific model with frozen shared layers
        """
        
        # Extract shared layers (freeze backbone, adapt prediction head)
        main_input = layers.Input(shape=input_shape, name=f"{stock_symbol}_input")
        
        # Use shared backbone layers (frozen)
        x = main_input
        for i, layer in enumerate(shared_backbone.layers[1:-2]):  # Skip input and final layers
            if hasattr(layer, 'trainable'):
                layer.trainable = False  # Freeze shared backbone
            x = layer(x)
        
        # Stock-specific adaptation layers
        adaptation_units = params.get("adaptation_units", 8)
        l2_reg = params.get("l2_regularization", 0.006)
        
        # Stock-specific adaptation layer
        x = layers.Dense(
            adaptation_units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f"{stock_symbol}_adaptation"
        )(x)
        
        x = layers.Dropout(0.3, name=f"{stock_symbol}_adaptation_dropout")(x)
        
        # Stock-specific prediction head
        output = layers.Dense(
            1,
            activation="tanh",
            name=f"{stock_symbol}_prediction",
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)
        
        model = keras.Model(inputs=main_input, outputs=output, name=f"SpecializedLSTM_{stock_symbol}")
        
        # Use same optimizer as shared model but with different learning rate for fine-tuning
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params.get("finetune_learning_rate", 0.0002),  # Lower LR for fine-tuning
            clipnorm=0.5
        )
        
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae", self._pattern_detection_accuracy, self._correlation_metric]
        )
        
        return model

    def create_regularization_schedule(self, epochs: int) -> Dict[str, callable]:
        """
        Create regularization schedule for curriculum learning
        
        Args:
            epochs: Total number of training epochs
            
        Returns:
            Dictionary of schedulers for dropout and L2 regularization
        """
        
        def dropout_schedule(epoch):
            """Increase dropout over time to prevent overfitting"""
            if epoch < epochs * 0.3:  # First 30% of training
                return 0.3
            elif epoch < epochs * 0.6:  # Middle 30% of training  
                return 0.4
            else:  # Final 40% of training
                return 0.5
        
        def l2_schedule(epoch):
            """Increase L2 regularization over time"""
            if epoch < epochs * 0.3:
                return 0.003
            elif epoch < epochs * 0.6:
                return 0.005
            else:
                return 0.008
        
        return {
            "dropout_schedule": dropout_schedule,
            "l2_schedule": l2_schedule
        }

    def get_regularization_callbacks(self, epochs: int) -> List[keras.callbacks.Callback]:
        """
        Get callbacks for enhanced regularization during training
        
        Args:
            epochs: Total training epochs
            
        Returns:
            List of Keras callbacks for regularization
        """
        
        callbacks = [
            # Early stopping with patience for overfitting prevention
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,  # Increased patience for swing trading
                restore_best_weights=True,
                verbose=1,
                mode="min"
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.7,  # More conservative reduction
                patience=8,
                min_lr=1e-6,
                verbose=1,
                mode="min"
            ),
            
            # Model checkpoint for best validation performance
            keras.callbacks.ModelCheckpoint(
                filepath="models/best_shared_backbone_model.keras",
                monitor="val_correlation_metric",
                save_best_only=True,
                save_weights_only=False,
                mode="max",
                verbose=1
            )
        ]
        
        return callbacks

    def _pattern_detection_accuracy(self, y_true, y_pred):
        """Calculate pattern detection accuracy metric for pattern confidence scores"""
        # Convert pattern confidence scores to binary classification (threshold = 0.5)
        y_true_binary = tf.cast(y_true > 0.5, tf.float32)
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(y_true_binary, y_pred_binary), tf.float32))

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
        """Get enhanced model architecture summary"""
        
        model = self.build_model(input_shape)
        
        # Count params
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        return {
            "model_name": "SharedBackboneLSTM",
            "input_shape": input_shape,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
            "architecture_type": "shared_backbone_regularized",
            "regularization_features": [
                "enhanced_dropout_0.45",
                "increased_l2_regularization_0.006", 
                "batch_normalization",
                "recurrent_dropout",
                "gradient_clipping_0.8",
                "activity_regularization_l1"
            ],
            "output_activation": "tanh",
            "output_range": "(-1, 1)",
            "designed_for": "expanded_universe_training",
            "target_parameter_count": "~400k",
            "overfitting_prevention": "high"
        }


def create_shared_backbone_lstm_builder(config: Dict[str, Any]) -> SharedBackboneLSTMBuilder:
    """Convenience function to create shared backbone LSTM builder"""
    return SharedBackboneLSTMBuilder(config)


# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add config import
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config.config import get_config
    
    # Test with swing trading configuration
    config = get_config()
    
    builder = create_shared_backbone_lstm_builder(config.dict())
    
    # Test model creation with swing trading input shape
    input_shape = (20, 17)  # 20-day sequences, 17 pattern features
    model = builder.build_model(input_shape)
    
    # Get model summary
    summary = builder.get_model_summary(input_shape)
    print(f"Model Summary: {summary}")
    
    # Test model prediction
    test_input = np.random.randn(1, 20, 17)
    prediction = model.predict(test_input, verbose=0)
    print(f"Test prediction: {prediction[0][0]:.6f} (range -1 to 1)")
    
    # Test regularization callbacks
    callbacks = builder.get_regularization_callbacks(epochs=100)
    print(f"Regularization callbacks: {len(callbacks)} configured")
    
    print("Shared backbone LSTM architecture validated for expanded universe training")