#!/usr/bin/env python3
"""
Model Components Unit Tests
Tests individual model components without overlap

Consolidates functionality from:
- src/models/tests/test_lstm_baseline.py
- Various model-related tests scattered across the codebase
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.utilities.data_fixtures import MockModelFixtures, TestDataGenerator
from tests.utilities.test_helpers import TestAssertions, TestEnvironment, TestMetrics


class TestModelComponents(unittest.TestCase):
    """Test individual model components and architectures"""

    def setUp(self):
        """Set up test fixtures"""
        self.input_shape = (20, 17)
        self.batch_size = 32
        self.samples = 100

    def test_lstm_builder_initialization(self):
        """Test LSTM builder can be initialized with various configurations"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from config.config import get_config
            from src.models.shared_backbone_lstm import SharedBackboneLSTMBuilder

            config = get_config()
            builder = SharedBackboneLSTMBuilder(config.model_dump())

            # Test builder was created successfully
            self.assertIsNotNone(builder)
            self.assertTrue(hasattr(builder, "build_model"))

    def test_model_architecture_creation(self):
        """Test model architecture can be created with correct shapes"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from config.config import get_config
            from src.models.shared_backbone_lstm import SharedBackboneLSTMBuilder

            config = get_config()
            builder = SharedBackboneLSTMBuilder(config.model_dump())

            # Build model
            model = builder.build_model(self.input_shape)

            # Validate model properties
            self.assertIsNotNone(model)
            self.assertEqual(len(model.inputs), 1)
            self.assertEqual(len(model.outputs), 1)
            self.assertEqual(model.input_shape[1:], self.input_shape)

            # Test model has trainable parameters
            self.assertGreater(model.count_params(), 0)

    def test_correlation_loss_function(self):
        """Test correlation-optimized loss function works correctly"""

        with TestEnvironment.suppress_tensorflow_warnings():
            import tensorflow as tf

            from config.config import get_config
            from src.models.shared_backbone_lstm import SharedBackboneLSTMBuilder

            # Create test data with known correlation
            y_true, y_pred = TestDataGenerator.generate_correlation_test_data(correlation=0.5, samples=100)

            y_true_tf = tf.constant(y_true)
            y_pred_tf = tf.constant(y_pred)

            config = get_config()
            builder = SharedBackboneLSTMBuilder(config.model_dump())
            model = builder.build_model(self.input_shape)

            # Get the loss function
            loss_fn = model.loss
            self.assertIsNotNone(loss_fn)

            # Test loss function returns a scalar
            loss_value = loss_fn(y_true_tf, y_pred_tf)
            self.assertIsInstance(loss_value.numpy(), (int, float, np.number))

    def test_correlation_metric_calculation(self):
        """Test correlation metric calculation accuracy"""

        with TestEnvironment.suppress_tensorflow_warnings():
            import tensorflow as tf

            from config.config import get_config
            from src.models.shared_backbone_lstm import SharedBackboneLSTMBuilder

            # Test with perfect correlation
            y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
            y_pred = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)  # Perfect correlation

            expected_correlation = np.corrcoef(y_true, y_pred)[0, 1]

            config = get_config()
            builder = SharedBackboneLSTMBuilder(config.model_dump())

            # Test correlation metric directly
            correlation_metric = builder._correlation_metric(tf.constant(y_true), tf.constant(y_pred))

            # Allow for small numerical differences
            TestAssertions.assert_correlation_close(correlation_metric.numpy(), expected_correlation, tolerance=0.01)

    def test_model_prediction_shape(self):
        """Test model produces correctly shaped predictions"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from config.config import get_config
            from src.models.shared_backbone_lstm import SharedBackboneLSTMBuilder

            X_test, _ = TestDataGenerator.generate_feature_matrix(
                samples=self.batch_size, features=self.input_shape[1], sequence_length=self.input_shape[0]
            )

            config = get_config()
            builder = SharedBackboneLSTMBuilder(config.model_dump())
            model = builder.build_model(self.input_shape)

            # predictions
            predictions = model.predict(X_test, verbose=0)

            # Validate prediction shape
            expected_shape = (self.batch_size, 1)  # Single output
            self.assertEqual(predictions.shape, expected_shape)

            # Validate prediction range
            TestAssertions.assert_array_properties(predictions, min_val=0.0, max_val=1.0, no_nan=True, no_inf=True)

    def test_model_gradient_flow(self):
        """Test model gradients can be computed (not blocked)"""

        with TestEnvironment.suppress_tensorflow_warnings():
            import tensorflow as tf

            from config.config import get_config
            from src.models.shared_backbone_lstm import SharedBackboneLSTMBuilder

            # Generate small test batch
            X_test, y_test = TestDataGenerator.generate_feature_matrix(
                samples=8,  # Small batch for gradient test
                features=self.input_shape[1],
                sequence_length=self.input_shape[0],
            )

            config = get_config()
            builder = SharedBackboneLSTMBuilder(config.model_dump())
            model = builder.build_model(self.input_shape)

            # Test gradient computation
            with tf.GradientTape() as tape:
                predictions = model(X_test, training=True)
                loss = model.loss(y_test, predictions)

            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Validate gradients exist and are not all zero
            self.assertIsNotNone(gradients)
            self.assertGreater(len(gradients), 0)

            # Check that at least some gradients are non-zero
            non_zero_gradients = sum(
                1 for grad in gradients if grad is not None and tf.reduce_sum(tf.abs(grad)) > 1e-10
            )
            self.assertGreater(non_zero_gradients, 0)


class TestModelIntegration(unittest.TestCase):
    """Test model integration with training components"""

    def test_model_training_compatibility(self):
        """Test model can be used with trainer without errors"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            # Create minimal trainer
            trainer = SharedBackboneTrainer(tickers=["AAPL"], use_expanded_universe=False)

            # Test trainer initialization worked
            self.assertIsNotNone(trainer)
            self.assertIsNotNone(trainer.lstm_builder)

            # Test model building through trainer
            input_shape = (20, 17)
            model = trainer.lstm_builder.build_model(input_shape)

            self.assertIsNotNone(model)
            self.assertTrue(model.built)

    def test_pattern_detection_accuracy_metric(self):
        """Test pattern detection accuracy calculation"""

        # Generate test predictions and targets
        y_true, y_pred = MockModelFixtures.create_mock_predictions(samples=100, correlation=0.3)

        # Calculate pattern accuracy (should be > 50% for reasonable predictions)
        accuracy = TestMetrics.calculate_pattern_accuracy(y_true, y_pred)

        # Basic validation
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

        # For correlated predictions, should be better than random
        self.assertGreater(accuracy, 0.4)  # Allow some tolerance


if __name__ == "__main__":
    # Run tests with minimal output
    unittest.main(verbosity=1)
