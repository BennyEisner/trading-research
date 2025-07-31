#!/usr/bin/env python3

"""
Test pattern model creation and basic functionality
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.pattern_focused_lstm import PatternFocusedLSTMBuilder


class TestPatternModelCreation(unittest.TestCase):
    """Test pattern focused LSTM model creation and basic operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "epochs": 50,
            "batch_size": 64,
            "sequence_length": 30,
            "patience": 8,
            "min_delta": 0.001,
            "learning_rate": 0.001,
            "reduce_lr_patience": 4,
            "reduce_lr_factor": 0.5,
            "validation_split": 0.2,
            "shuffle": False,
            "lstm_units_1": 96,
            "lstm_units_2": 48,
            "dropout_rate": 0.3,
            "l2_regularization": 0.003,
            "model_type": "pattern_focused",
            "output_activation": "tanh",
        }
        self.builder = PatternFocusedLSTMBuilder(self.config)
        self.input_shape = (30, 12)

    def test_model_creation(self):
        """Test that model can be created without errors"""
        model = self.builder.build_model(self.input_shape, **self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "PatternFocusedLSTM")

    def test_model_input_shape(self):
        """Test model accepts correct input shape"""
        model = self.builder.build_model(self.input_shape, **self.config)
        expected_input_shape = (None, 30, 12)
        self.assertEqual(model.input_shape, expected_input_shape)

    def test_model_output_shape(self):
        """Test model produces correct output shape"""
        model = self.builder.build_model(self.input_shape, **self.config)
        expected_output_shape = (None, 1)
        self.assertEqual(model.output_shape, expected_output_shape)

    def test_model_prediction(self):
        """Test model can make predictions"""
        model = self.builder.build_model(self.input_shape, **self.config)
        test_input = np.random.randn(1, 30, 12)
        prediction = model.predict(test_input, verbose=0)
        
        # Check prediction shape
        self.assertEqual(prediction.shape, (1, 1))
        
        # Check prediction is in tanh range [-1, 1]
        self.assertGreaterEqual(prediction[0][0], -1.0)
        self.assertLessEqual(prediction[0][0], 1.0)

    def test_model_parameter_count(self):
        """Test model has expected parameter count (~500K)"""
        model = self.builder.build_model(self.input_shape, **self.config)
        total_params = model.count_params()
        
        # Should be around 500K parameters
        self.assertGreater(total_params, 300000)  # At least 300K
        self.assertLess(total_params, 800000)     # At most 800K

    def test_model_summary_info(self):
        """Test model summary returns correct information"""
        summary = self.builder.get_model_summary(self.input_shape)
        
        self.assertEqual(summary["model_name"], "PatternFocusedLSTM")
        self.assertEqual(summary["input_shape"], self.input_shape)
        self.assertEqual(summary["architecture_type"], "simplified_pattern_focused")
        self.assertEqual(summary["output_activation"], "tanh")
        self.assertEqual(summary["output_range"], "(-1, 1)")
        self.assertEqual(summary["designed_for"], "ensemble_integration")

    def test_batch_prediction(self):
        """Test model can handle batch predictions"""
        model = self.builder.build_model(self.input_shape, **self.config)
        batch_size = 5
        test_input = np.random.randn(batch_size, 30, 12)
        predictions = model.predict(test_input, verbose=0)
        
        # Check batch prediction shape
        self.assertEqual(predictions.shape, (batch_size, 1))
        
        # Check all predictions are in valid range
        for pred in predictions:
            self.assertGreaterEqual(pred[0], -1.0)
            self.assertLessEqual(pred[0], 1.0)


def create_pattern_training_config():
    """Create pattern training configuration"""
    return {
        "epochs": 50,
        "batch_size": 64,
        "sequence_length": 30,
        "patience": 8,
        "min_delta": 0.001,
        "learning_rate": 0.001,
        "reduce_lr_patience": 4,
        "reduce_lr_factor": 0.5,
        "validation_split": 0.2,
        "shuffle": False,
        "lstm_units_1": 96,
        "lstm_units_2": 48,
        "dropout_rate": 0.3,
        "l2_regularization": 0.003,
        "model_type": "pattern_focused",
        "output_activation": "tanh",
    }


def test_pattern_model_creation():
    """Standalone test function for quick validation"""
    try:
        config = create_pattern_training_config()
        builder = PatternFocusedLSTMBuilder(config)
        
        input_shape = (30, 12)
        model = builder.build_model(input_shape, **config)
        print("Pattern focused LSTM built successfully")
        
        test_input = np.random.randn(1, 30, 12)
        prediction = model.predict(test_input, verbose=0)
        print(f"Test prediction: {prediction[0][0]:.6f} (range -1 to 1)")
        return True

    except Exception as e:
        print(f"Error creating pattern focused model: {e}")
        return False


if __name__ == "__main__":
    # Run standalone test
    test_pattern_model_creation()
    
    # Run unit tests
    unittest.main()