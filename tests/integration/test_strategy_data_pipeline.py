#!/usr/bin/env python3

"""
Tests for end-to-end ensemble data pipeline
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.pipeline.orchestration.ensemble_data_pipeline import (
    EnsembleDataPipeline,
    PipelineBatchProcessor,
    PipelineError,
    create_ensemble_pipeline,
)
from src.strategies.ensemble import EnsembleConfig
from src.strategies.implementations.macd_momentum_strategy import MACDMomentumStrategy, MACDStrategyConfig
from src.strategies.implementations.rsi_strategy import RSIMeanReversionStrategy, RSIStrategyConfig


class TestEnsembleDataPipeline:
    """Test complete end-to-end data pipeline"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for pipeline testing"""
        # Use recent dates and sufficient data for realistic backtesting
        dates = pd.date_range("2024-01-01", periods=300, freq="D")  # 300 days for backtesting

        # Create realistic price data
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0, 0.02, 300)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLCV
        close_prices = np.array(prices)
        open_prices = close_prices * (1 + np.random.normal(0, 0.005, 300))
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.01, 300))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.01, 300))
        volumes = np.random.uniform(1000000, 5000000, 300)

        return pd.DataFrame(
            {
                "date": dates,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            }
        )

    @pytest.fixture
    def strategies(self):
        """Create strategy instances for testing"""
        rsi_config = RSIStrategyConfig()
        macd_config = MACDStrategyConfig()

        return [RSIMeanReversionStrategy(rsi_config), MACDMomentumStrategy(macd_config)]

    @pytest.fixture
    def ensemble_config(self):
        """Create ensemble configuration"""
        return EnsembleConfig()

    @pytest.fixture
    def pipeline(self, strategies, ensemble_config):
        """Create pipeline instance with production-like settings"""
        return EnsembleDataPipeline(strategies=strategies, ensemble_config=ensemble_config, validation_enabled=True)

    def test_pipeline_creation(self, strategies, ensemble_config):
        """Test pipeline can be created successfully"""
        pipeline = EnsembleDataPipeline(strategies=strategies, ensemble_config=ensemble_config)

        assert len(pipeline.strategies) == 2
        assert pipeline.ensemble_config == ensemble_config
        assert pipeline.validation_enabled is True
        assert len(pipeline.execution_results) == 0
        assert len(pipeline.pipeline_metrics) == 0

    def test_create_ensemble_pipeline_convenience_function(self, strategies):
        """Test convenience function for pipeline creation"""
        pipeline = create_ensemble_pipeline(strategies)

        assert isinstance(pipeline, EnsembleDataPipeline)
        assert len(pipeline.strategies) == 2
        assert pipeline.validation_enabled is True

    def test_pipeline_execute_success(self, pipeline, sample_ohlcv_data):
        """Test successful end-to-end pipeline execution"""

        results = pipeline.execute(sample_ohlcv_data)

        # Verify result structure
        assert "ensemble_signals" in results
        assert "individual_signals" in results
        assert "engineered_features" in results
        assert "pipeline_metrics" in results
        assert "execution_results" in results
        assert "validation_summary" in results
        assert "timestamp" in results

        # Verify ensemble signals
        ensemble_signals = results["ensemble_signals"]
        assert isinstance(ensemble_signals, pd.DataFrame)
        assert len(ensemble_signals) == len(sample_ohlcv_data)
        assert "position" in ensemble_signals.columns
        assert "confidence" in ensemble_signals.columns

        # Verify individual signals
        individual_signals = results["individual_signals"]
        assert isinstance(individual_signals, dict)
        assert len(individual_signals) >= 1  # At least one strategy should work

        for strategy_name, signals in individual_signals.items():
            assert isinstance(signals, pd.DataFrame)
            assert "position" in signals.columns
            assert len(signals) == len(sample_ohlcv_data)

        # Verify engineered features
        engineered_features = results["engineered_features"]
        assert isinstance(engineered_features, pd.DataFrame)
        assert len(engineered_features) == len(sample_ohlcv_data)
        assert len(engineered_features.columns) > len(sample_ohlcv_data.columns)

        # Verify pipeline metrics
        metrics = results["pipeline_metrics"]
        assert metrics["data_points_processed"] == len(sample_ohlcv_data)
        assert metrics["execution_time_seconds"] > 0
        assert metrics["features_engineered"] > 0
        assert metrics["strategies_executed"] >= 1
        assert metrics["ensemble_signals_generated"] == len(ensemble_signals)

        # Verify execution results
        execution_results = results["execution_results"]
        assert "raw_data_validation" in execution_results
        assert "feature_engineering" in execution_results
        assert "strategy_validation" in execution_results
        assert "signal_generation" in execution_results

        # All stages should have completed successfully
        for stage_name, stage_result in execution_results.items():
            if isinstance(stage_result, dict):
                status = stage_result.get("status")
                assert status in ["PASSED", "COMPLETED"], f"Stage {stage_name} failed with result: {stage_result}"

    def test_pipeline_execute_invalid_data(self, pipeline):
        """Test pipeline execution with invalid data"""

        # Create invalid data (missing required columns)
        invalid_data = pd.DataFrame({"invalid_column": [1, 2, 3]})

        with pytest.raises(PipelineError) as exc_info:
            pipeline.execute(invalid_data)

        assert "missing required ohlcv columns" in str(exc_info.value).lower()

    def test_pipeline_execute_with_validation_disabled(self, strategies, sample_ohlcv_data):
        """Test pipeline execution with validation disabled"""

        pipeline = EnsembleDataPipeline(strategies=strategies, validation_enabled=False)

        results = pipeline.execute(sample_ohlcv_data)

        # Should still work but without validation summary
        assert "ensemble_signals" in results
        assert "individual_signals" in results
        assert "engineered_features" in results
        assert results["pipeline_metrics"]["validation_enabled"] is False

        # Validation results should be minimal when validation is disabled
        assert "validation_summary" not in results

    def test_pipeline_execute_with_failing_strategy(self, sample_ohlcv_data, ensemble_config):
        """Test pipeline execution with one failing strategy"""

        # Create one good strategy and one mock failing strategy
        good_strategy = RSIMeanReversionStrategy(RSIStrategyConfig())

        failing_strategy = Mock()
        failing_strategy.name = "failing_strategy"
        failing_strategy.get_required_features.return_value = ["nonexistent_feature"]
        failing_strategy.validate_parameters.return_value = True

        pipeline = EnsembleDataPipeline(
            strategies=[good_strategy, failing_strategy], ensemble_config=ensemble_config, validation_enabled=True
        )

        # Should still execute successfully with graceful degradation
        results = pipeline.execute(sample_ohlcv_data)

        assert "ensemble_signals" in results
        assert "individual_signals" in results

        # Should have signals from the working strategy
        individual_signals = results["individual_signals"]
        assert good_strategy.name in individual_signals
        assert failing_strategy.name not in individual_signals

        # Execution results should show the issue
        signal_gen_results = results["execution_results"]["signal_generation"]
        assert len(signal_gen_results["invalid_strategies"]) == 1
        assert failing_strategy.name in signal_gen_results["invalid_strategies"]

    def test_pipeline_reset(self, pipeline, sample_ohlcv_data):
        """Test pipeline reset functionality"""

        # Execute pipeline to populate results
        pipeline.execute(sample_ohlcv_data)
        assert len(pipeline.execution_results) > 0
        assert len(pipeline.pipeline_metrics) > 0

        # Reset pipeline
        pipeline.reset_pipeline()
        assert len(pipeline.execution_results) == 0
        assert len(pipeline.pipeline_metrics) == 0

    def test_pipeline_get_status(self, pipeline):
        """Test pipeline status reporting"""

        status = pipeline.get_pipeline_status()

        assert "strategies_registered" in status
        assert "ensemble_config" in status
        assert "validation_enabled" in status
        assert "execution_results" in status
        assert "pipeline_metrics" in status
        assert "last_execution" in status

        assert status["strategies_registered"] == len(pipeline.strategies)
        assert status["validation_enabled"] is True
        assert status["last_execution"] == "Never"

    def test_feature_engineering_stage(self, pipeline, sample_ohlcv_data):
        """Test feature engineering stage in isolation"""

        # Execute feature engineering
        engineered_data = pipeline._execute_feature_engineering(sample_ohlcv_data)

        # Verify engineered data
        assert isinstance(engineered_data, pd.DataFrame)
        assert len(engineered_data) == len(sample_ohlcv_data)
        assert len(engineered_data.columns) > len(sample_ohlcv_data.columns)

        # Should have technical indicators
        expected_features = ["rsi", "rsi_14", "macd", "macd_signal", "atr"]
        for feature in expected_features:
            assert feature in engineered_data.columns, f"Missing feature: {feature}"

        # Verify execution results were recorded
        assert "feature_engineering" in pipeline.execution_results
        fe_result = pipeline.execution_results["feature_engineering"]
        assert fe_result["status"] == "COMPLETED"
        assert fe_result["new_features"] > 0

    def test_signal_generation_stage(self, pipeline, sample_ohlcv_data):
        """Test signal generation stage"""

        # First engineer features
        engineered_data = pipeline._execute_feature_engineering(sample_ohlcv_data)

        # Then generate signals
        ensemble_signals, individual_signals = pipeline._execute_signal_generation(engineered_data)

        # Verify ensemble signals
        assert isinstance(ensemble_signals, pd.DataFrame)
        assert len(ensemble_signals) == len(engineered_data)
        assert "position" in ensemble_signals.columns

        # Verify individual signals
        assert isinstance(individual_signals, dict)
        assert len(individual_signals) >= 1

        for strategy_name, signals in individual_signals.items():
            assert isinstance(signals, pd.DataFrame)
            assert len(signals) == len(engineered_data)
            assert "position" in signals.columns

        # Verify execution results
        assert "signal_generation" in pipeline.execution_results
        sg_result = pipeline.execution_results["signal_generation"]
        assert sg_result["status"] == "COMPLETED"
        assert sg_result["ensemble_signals_generated"] == len(ensemble_signals)


class TestPipelineBatchProcessor:
    """Test batch processing functionality"""

    @pytest.fixture
    def sample_datasets(self):
        """Create multiple sample datasets for batch processing"""
        datasets = []

        for i in range(3):
            dates = pd.date_range(f"2024-0{i+1}-01", periods=300, freq="D")  # Sufficient for backtesting
            np.random.seed(42 + i)  # Different seed for each dataset

            # Create realistic price walk
            base_price = 100
            returns = np.random.normal(0, 0.02, 300)
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            close_prices = np.array(prices)
            open_prices = close_prices * (1 + np.random.normal(0, 0.005, 300))
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.01, 300))
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.01, 300))
            volumes = np.random.uniform(1000000, 5000000, 300)

            data = pd.DataFrame(
                {
                    "date": dates,
                    "open": open_prices,
                    "high": high_prices,
                    "low": low_prices,
                    "close": close_prices,
                    "volume": volumes,
                }
            )

            datasets.append((f"dataset_{i}", data))

        return datasets

    @pytest.fixture
    def batch_processor(self):
        """Create batch processor instance"""
        strategies = [RSIMeanReversionStrategy(RSIStrategyConfig()), MACDMomentumStrategy(MACDStrategyConfig())]

        pipeline = create_ensemble_pipeline(strategies, validation_enabled=False)  # Disable validation for speed
        return PipelineBatchProcessor(pipeline)

    def test_batch_processor_creation(self, batch_processor):
        """Test batch processor can be created"""
        assert isinstance(batch_processor, PipelineBatchProcessor)
        assert isinstance(batch_processor.pipeline, EnsembleDataPipeline)
        assert len(batch_processor.batch_results) == 0

    def test_batch_processing_success(self, batch_processor, sample_datasets):
        """Test successful batch processing"""

        results = batch_processor.process_batch(sample_datasets)

        # Verify batch results
        assert len(results) == len(sample_datasets)

        for i, result in enumerate(results):
            assert result["dataset_identifier"] == f"dataset_{i}"
            assert result["batch_status"] == "SUCCESS"
            assert "ensemble_signals" in result
            assert "individual_signals" in result
            assert "pipeline_metrics" in result

        # Verify batch processor state
        assert len(batch_processor.batch_results) == len(sample_datasets)

    def test_batch_processing_with_failures(self, batch_processor):
        """Test batch processing with some failing datasets"""

        # Create mix of valid and invalid datasets
        valid_data = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=50),
                "open": np.random.uniform(95, 105, 50),
                "high": np.random.uniform(100, 110, 50),
                "low": np.random.uniform(90, 100, 50),
                "close": np.random.uniform(95, 105, 50),
                "volume": np.random.uniform(1000000, 5000000, 50),
            }
        )

        invalid_data = pd.DataFrame({"invalid_column": [1, 2, 3]})

        datasets = [("valid_dataset", valid_data), ("invalid_dataset", invalid_data)]

        results = batch_processor.process_batch(datasets)

        # Verify mixed results
        assert len(results) == 2
        assert results[0]["batch_status"] == "SUCCESS"
        assert results[1]["batch_status"] == "FAILED"
        assert "error" in results[1]

    def test_batch_summary(self, batch_processor, sample_datasets):
        """Test batch processing summary"""

        # Process batch
        batch_processor.process_batch(sample_datasets)

        # Get summary
        summary = batch_processor.get_batch_summary()

        assert summary["total_datasets"] == len(sample_datasets)
        assert summary["successful"] == len(sample_datasets)
        assert summary["failed"] == 0
        assert summary["success_rate"] == 1.0
        assert summary["average_execution_time"] > 0
        assert len(summary["failed_datasets"]) == 0

    def test_batch_summary_empty(self, batch_processor):
        """Test batch summary with no processing done"""

        summary = batch_processor.get_batch_summary()

        assert "status" in summary
        assert summary["status"] == "No batch processing completed"


class TestPipelineErrorHandling:
    """Test error handling in pipeline execution"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for error testing"""
        dates = pd.date_range("2025-01-01", periods=50, freq="D")
        np.random.seed(42)

        close_prices = np.random.uniform(95, 105, 50)
        open_prices = close_prices * (1 + np.random.normal(0, 0.005, 50))
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.01, 50))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.01, 50))
        volumes = np.random.uniform(1000000, 5000000, 50)

        return pd.DataFrame(
            {
                "date": dates,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            }
        )

    def test_pipeline_error_creation(self):
        """Test PipelineError can be created"""
        error = PipelineError("Test error message")
        assert str(error) == "Test error message"

    def test_pipeline_error_propagation(self, sample_ohlcv_data):
        """Test error propagation through pipeline"""

        # Create pipeline with mock strategy that will fail
        failing_strategy = Mock()
        failing_strategy.name = "failing_strategy"
        failing_strategy.get_required_features.side_effect = Exception("Mock failure")
        failing_strategy.validate_parameters.return_value = True

        pipeline = EnsembleDataPipeline(strategies=[failing_strategy], validation_enabled=True)

        with pytest.raises(PipelineError) as exc_info:
            pipeline.execute(sample_ohlcv_data)

        assert "Pipeline execution failed" in str(exc_info.value)
        assert "Mock failure" in str(exc_info.value)

    @patch("src.strategies.pipeline.data_pipeline.PriceFeaturesProcessor")
    def test_feature_engineering_error_handling(self, mock_processor, sample_ohlcv_data):
        """Test error handling in feature engineering stage"""

        # Mock processor to raise exception
        mock_processor.return_value.calculate.side_effect = Exception("Feature engineering failed")

        strategies = [RSIMeanReversionStrategy(RSIStrategyConfig())]
        pipeline = EnsembleDataPipeline(strategies=strategies)

        with pytest.raises(PipelineError) as exc_info:
            pipeline.execute(sample_ohlcv_data)

        assert "Feature engineering failed" in str(exc_info.value)

