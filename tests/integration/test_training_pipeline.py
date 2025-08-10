#!/usr/bin/env python3
"""
Training Pipeline Integration Test
Tests complete training pipeline integration without redundancy

Consolidates functionality from:
- tests/test_data_pipeline.py
- src/strategies/tests/test_integration.py
- Multiple scattered integration tests
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.utilities.data_fixtures import TestConfigurationFixtures, TestDataGenerator
from tests.utilities.test_helpers import TestAssertions, TestEnvironment, TestMetrics, TestTimer


class TestTrainingPipelineIntegration(unittest.TestCase):
    """Test complete training pipeline integration"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.test_tickers = ["AAPL", "MSFT"]
        self.test_days = 60  # Minimal for fast integration testing
        self.test_epochs = 2  # Minimal epochs for integration test

        # Generate consistent test data
        self.ticker_data = TestDataGenerator.generate_multi_ticker_data(tickers=self.test_tickers, days=self.test_days)

    def test_data_loading_to_training_pipeline(self):
        """Test complete pipeline from data loading to training preparation"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            # Initialize trainer
            trainer = SharedBackboneTrainer(tickers=self.test_tickers, use_expanded_universe=False)

            # Test data preparation
            training_data = trainer.prepare_training_data(self.ticker_data)

            # Validate training data structure
            self.assertIsInstance(training_data, dict)
            self.assertGreater(len(training_data), 0)

            # Validate each ticker's data
            for ticker, (X, y) in training_data.items():
                # Check data types and shapes
                TestAssertions.assert_array_properties(
                    X,
                    shape=(X.shape[0], 20, 17),  # Expected sequence structure
                    dtype=np.float32,
                    no_nan=True,
                    no_inf=True,
                )

                TestAssertions.assert_array_properties(
                    y, min_val=0.0, max_val=1.0, shape=(len(X),), dtype=np.float32, no_nan=True, no_inf=True
                )

                # Validate we have reasonable amount of data
                self.assertGreater(len(X), 10)  # At least some sequences

    def test_training_data_to_model_training(self):
        """Test pipeline from prepared training data to actual model training"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            trainer = SharedBackboneTrainer(tickers=self.test_tickers, use_expanded_universe=False)

            # Prepare training data
            training_data = trainer.prepare_training_data(self.ticker_data)

            # Test model training
            training_results = trainer.train_shared_backbone(
                training_data=training_data, validation_split=0.2, epochs=self.test_epochs
            )

            # Validate training results structure
            self.assertIsInstance(training_results, dict)
            required_keys = ["model", "history", "final_metrics"]
            for key in required_keys:
                self.assertIn(key, training_results, f"Missing key: {key}")

            # Validate model exists and is trained
            model = training_results["model"]
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "predict"))

            # Validate training history
            history = training_results["history"]
            self.assertIn("loss", history)
            self.assertEqual(len(history["loss"]), self.test_epochs)

            # Validate final metrics
            final_metrics = training_results["final_metrics"]
            required_metrics = ["correlation", "pattern_detection_accuracy"]
            for metric in required_metrics:
                self.assertIn(metric, final_metrics, f"Missing metric: {metric}")
                self.assertIsInstance(final_metrics[metric], (int, float, np.number))

    def test_model_inference_pipeline(self):
        """Test inference pipeline from trained model to predictions"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            trainer = SharedBackboneTrainer(tickers=self.test_tickers, use_expanded_universe=False)

            # Full pipeline
            training_data = trainer.prepare_training_data(self.ticker_data)
            training_results = trainer.train_shared_backbone(training_data=training_data, epochs=self.test_epochs)

            model = training_results["model"]

            # Test inference on each ticker
            for ticker, (X, y) in training_data.items():
                # Generate predictions
                predictions = model.predict(X[:10], verbose=0)  # Small sample for speed

                # Validate predictions
                TestAssertions.assert_array_properties(
                    predictions, min_val=0.0, max_val=1.0, shape=(10, 1), no_nan=True, no_inf=True
                )

                # Test predictions have some variance (not constant)
                pred_variance = np.var(predictions)
                self.assertGreater(pred_variance, 1e-6, "Predictions are too constant")

    def test_cross_ticker_validation_pipeline(self):
        """Test cross-ticker validation pipeline"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            trainer = SharedBackboneTrainer(tickers=self.test_tickers, use_expanded_universe=False)

            # Full pipeline to trained model
            training_data = trainer.prepare_training_data(self.ticker_data)
            training_results = trainer.train_shared_backbone(training_data=training_data, epochs=self.test_epochs)

            # Test cross-ticker validation
            validation_results = trainer.validate_cross_ticker_performance(
                training_data=training_data, trained_model=training_results["model"]
            )

            # Validate validation results structure
            self.assertIn("ticker_performance", validation_results)
            self.assertIn("overall_stats", validation_results)

            ticker_performance = validation_results["ticker_performance"]
            overall_stats = validation_results["overall_stats"]

            # Validate each ticker has results
            for ticker in self.test_tickers:
                if ticker in ticker_performance:
                    ticker_results = ticker_performance[ticker]
                    required_metrics = ["mean_pattern_detection_accuracy", "mean_correlation", "mean_mae"]

                    for metric in required_metrics:
                        self.assertIn(metric, ticker_results)
                        self.assertIsInstance(ticker_results[metric], (int, float, np.number))

            # Validate overall stats
            required_overall = ["mean_pattern_detection_accuracy", "mean_correlation", "successful_tickers"]
            for stat in required_overall:
                self.assertIn(stat, overall_stats)

    def test_pipeline_error_handling(self):
        """Test pipeline handles errors gracefully"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            # Test with minimal/invalid data
            invalid_data = {"INVALID": TestDataGenerator.generate_ohlcv_data(days=5)}  # Too little data

            trainer = SharedBackboneTrainer(tickers=["INVALID"], use_expanded_universe=False)

            # Should handle insufficient data gracefully
            training_data = trainer.prepare_training_data(invalid_data)

            # Either produces empty training data or handles the error
            if training_data:
                # If data was produced, it should be valid
                for ticker, (X, y) in training_data.items():
                    self.assertGreater(len(X), 0)
            else:
                # Empty data is acceptable for invalid input
                self.assertEqual(len(training_data), 0)

    def test_pipeline_performance_benchmarks(self):
        """Test pipeline performance meets basic benchmarks"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            timer = TestTimer()

            with timer.measure():
                trainer = SharedBackboneTrainer(tickers=self.test_tickers, use_expanded_universe=False)

                training_data = trainer.prepare_training_data(self.ticker_data)
                training_results = trainer.train_shared_backbone(training_data=training_data, epochs=self.test_epochs)

            # Basic performance validation
            total_time = timer.elapsed()

            # Should complete integration test in reasonable time
            self.assertLess(total_time, 120, f"Integration test took too long: {total_time:.1f}s")

            # Validate training actually happened
            final_metrics = training_results["final_metrics"]

            # Training should produce some learning (loss should decrease from first to last epoch)
            history = training_results["history"]
            if len(history["loss"]) > 1:
                initial_loss = history["loss"][0]
                final_loss = history["loss"][-1]

                # Allow for some cases where loss might not decrease in just 2 epochs
                # Just check it's not wildly increasing
                self.assertLess(final_loss, initial_loss * 2, "Training loss increased too much")


class TestPipelineIntegrationCornerCases(unittest.TestCase):
    """Test pipeline integration corner cases and edge conditions"""

    def test_single_ticker_pipeline(self):
        """Test pipeline works with single ticker"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            single_ticker_data = TestDataGenerator.generate_multi_ticker_data(tickers=["AAPL"], days=50)

            trainer = SharedBackboneTrainer(tickers=["AAPL"], use_expanded_universe=False)

            training_data = trainer.prepare_training_data(single_ticker_data)
            self.assertEqual(len(training_data), 1)

            training_results = trainer.train_shared_backbone(training_data=training_data, epochs=2)

            self.assertIn("model", training_results)
            self.assertIn("final_metrics", training_results)

    def test_minimal_data_pipeline(self):
        """Test pipeline with minimal data requirements"""

        with TestEnvironment.suppress_tensorflow_warnings():
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            # Generate minimal data (just enough for sequence creation)
            minimal_data = TestDataGenerator.generate_multi_ticker_data(
                tickers=["AAPL"], days=25  # Just enough for 20-day sequences + targets
            )

            trainer = SharedBackboneTrainer(tickers=["AAPL"], use_expanded_universe=False)

            training_data = trainer.prepare_training_data(minimal_data)

            if training_data:  # If data was generated successfully
                # Should be able to train even with minimal data
                training_results = trainer.train_shared_backbone(
                    training_data=training_data,
                    epochs=1,  # Single epoch for minimal test
                    validation_split=0.1,  # Smaller validation split
                )

                self.assertIn("model", training_results)


if __name__ == "__main__":
    unittest.main(verbosity=1)

