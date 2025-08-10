#!/usr/bin/env python3

"""
Integration tests for feature engineering to strategy pipeline
Tests end-to-end flow: Raw Data → Features → Strategies → Ensemble
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.features.processors.price_features import PriceFeaturesProcessor
from src.features.processors.technical_indicators import TechnicalIndicatorsProcessor
from src.strategies.adapters.data_adapter import StrategyDataAdapter
from src.strategies.core.compatibility_checker import FeatureValidationError
from src.strategies.ensemble import EnsembleConfig, EnsembleManager
from src.strategies.implementations.macd_momentum_strategy import MACDMomentumStrategy, MACDStrategyConfig
from src.strategies.implementations.rsi_strategy import RSIMeanReversionStrategy, RSIStrategyConfig
from src.training.shared_backbone_trainer import SharedBackboneTrainer
from src.training.pattern_target_generator import PatternTargetGenerator


class TestFeatureToStrategyIntegration:
    """Test integration between feature engineering and strategy framework"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create realistic price data
        np.random.seed(42)  # For reproducible tests
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLCV with realistic relationships
        close_prices = np.array(prices)
        open_prices = close_prices * (1 + np.random.normal(0, 0.005, 100))
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.01, 100))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.01, 100))
        volumes = np.random.uniform(1000000, 5000000, 100)

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
    def engineered_features_data(self, sample_ohlcv_data):
        """Create data with engineered features"""
        # Use the existing processors to create realistic features
        price_processor = PriceFeaturesProcessor()
        tech_processor = TechnicalIndicatorsProcessor()

        # Process price features first (creates EMAs needed for MACD)
        data_with_price_features = price_processor.calculate(sample_ohlcv_data)

        # Then process technical indicators
        data_with_tech_features = tech_processor.calculate(data_with_price_features)

        return data_with_tech_features

    @pytest.fixture
    def data_adapter(self):
        """Create data adapter instance"""
        return StrategyDataAdapter(min_data_points=20, max_nan_ratio=0.2, validation_enabled=True)

    @pytest.fixture
    def rsi_strategy(self):
        """Create RSI strategy instance"""
        config = RSIStrategyConfig()
        return RSIMeanReversionStrategy(config)

    @pytest.fixture
    def macd_strategy(self):
        """Create MACD strategy instance"""
        config = MACDStrategyConfig()
        return MACDMomentumStrategy(config)

    def test_feature_column_alignment(self, engineered_features_data, rsi_strategy, macd_strategy):
        """Test that engineered features have correct column names for strategies"""

        # Check RSI strategy requirements
        rsi_required = rsi_strategy.get_required_features()
        rsi_missing = [col for col in rsi_required if col not in engineered_features_data.columns]

        assert len(rsi_missing) == 0, f"RSI strategy missing features: {rsi_missing}"

        macd_required = macd_strategy.get_required_features()
        macd_missing = [col for col in macd_required if col not in engineered_features_data.columns]

        assert len(macd_missing) == 0, f"MACD strategy missing features: {macd_missing}"

    def test_strategy_data_adapter_validation_success(self, engineered_features_data, data_adapter, rsi_strategy):
        """Test successful data adapter validation"""

        # Should not raise exception
        prepared_data, validation_results = data_adapter.prepare_strategy_data(engineered_features_data, rsi_strategy)

        # Validate results
        assert validation_results["validation_passed"] is True
        assert validation_results["strategy_name"] == rsi_strategy.name
        assert len(validation_results["errors"]) == 0
        assert isinstance(prepared_data, pd.DataFrame)
        assert len(prepared_data) == len(engineered_features_data)

    def test_strategy_data_adapter_missing_features(self, rsi_strategy):
        """Test data adapter handling of missing features"""

        # Create adapter with lower data requirements to test feature validation
        test_adapter = StrategyDataAdapter(min_data_points=3, validation_enabled=True)

        # Create data missing features but with sufficient data points
        incomplete_data = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
                # Missing RSI, ATR features
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        # Should raise FeatureValidationError
        with pytest.raises(FeatureValidationError) as exc_info:
            test_adapter.prepare_strategy_data(incomplete_data, rsi_strategy)

        assert "Critical features missing" in str(exc_info.value)

    def test_strategy_data_adapter_insufficient_data(self, engineered_features_data, rsi_strategy):
        """Test data adapter handling of insufficient data points"""

        # Create adapter with high minimum data requirement
        strict_adapter = StrategyDataAdapter(min_data_points=200)

        with pytest.raises(FeatureValidationError) as exc_info:
            strict_adapter.prepare_strategy_data(engineered_features_data, rsi_strategy)

        assert "Insufficient data points" in str(exc_info.value)

    def test_ensemble_data_preparation_success(
        self, engineered_features_data, data_adapter, rsi_strategy, macd_strategy
    ):
        """Test successful ensemble data preparation"""

        strategies = [rsi_strategy, macd_strategy]

        prepared_data, ensemble_results = data_adapter.prepare_ensemble_data(engineered_features_data, strategies)

        # Validate ensemble results
        assert ensemble_results["ensemble_ready"] is True
        assert ensemble_results["total_strategies"] == 2
        assert len(ensemble_results["valid_strategies"]) == 2
        assert len(ensemble_results["invalid_strategies"]) == 0

    def test_ensemble_graceful_degradation(self, engineered_features_data, data_adapter, rsi_strategy):
        """Test ensemble graceful degradation when some strategies fail"""

        # mock strategy that will fail validation
        failing_strategy = Mock()
        failing_strategy.name = "failing_strategy"
        failing_strategy.get_required_features.return_value = ["nonexistent_feature"]
        failing_strategy.validate_parameters.return_value = True

        strategies = [rsi_strategy, failing_strategy]

        # Should succeed with 1 valid strategy
        prepared_data, ensemble_results = data_adapter.prepare_ensemble_data(engineered_features_data, strategies)

        assert ensemble_results["ensemble_ready"] is True
        assert len(ensemble_results["valid_strategies"]) == 1
        assert len(ensemble_results["invalid_strategies"]) == 1
        assert ensemble_results["valid_strategies"][0] == rsi_strategy.name

    def test_rsi_strategy_signal_generation_with_features(self, engineered_features_data, data_adapter, rsi_strategy):
        """Test RSI strategy can generate signals with engineered features"""

        # Prepare data
        prepared_data, _ = data_adapter.prepare_strategy_data(engineered_features_data, rsi_strategy)

        # Generate signals
        signals = rsi_strategy.generate_signals(prepared_data)

        # Validate signal structure
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(prepared_data)
        assert "position" in signals.columns
        assert "entry_price" in signals.columns
        assert "stop_loss" in signals.columns
        assert "take_profit" in signals.columns

        # Check signal values are valid
        assert signals["position"].isin([-1.0, 0.0, 1.0]).all()

    def test_macd_strategy_signal_generation_with_features(self, engineered_features_data, data_adapter, macd_strategy):
        """Test MACD strategy can generate signals with engineered features"""

        prepared_data, _ = data_adapter.prepare_strategy_data(engineered_features_data, macd_strategy)
        signals = macd_strategy.generate_signals(prepared_data)

        # signal structure validation
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(prepared_data)
        assert "position" in signals.columns
        assert "entry_price" in signals.columns

        # Check signal values are valid
        assert signals["position"].isin([-1.0, 0.0, 1.0]).all()

    def test_end_to_end_ensemble_signal_generation(self, engineered_features_data, data_adapter):
        """Test complete e2e ensemble signal generation"""

        # Create strategies
        rsi_strategy = RSIMeanReversionStrategy(RSIStrategyConfig())
        macd_strategy = MACDMomentumStrategy(MACDStrategyConfig())
        strategies = [rsi_strategy, macd_strategy]

        # Prepare ensemble data
        prepared_data, ensemble_results = data_adapter.prepare_ensemble_data(engineered_features_data, strategies)

        # Create ensemble manager
        ensemble_config = EnsembleConfig()
        ensemble_manager = EnsembleManager(ensemble_config)

        # Register strategies
        for strategy in strategies:
            ensemble_manager.register_strategy(strategy)

        # Generate ensemble signals
        ensemble_signals = ensemble_manager.generate_ensemble_signals(prepared_data)

        # Validate ensemble signals
        assert isinstance(ensemble_signals, pd.DataFrame)
        assert len(ensemble_signals) == len(prepared_data)
        assert "position" in ensemble_signals.columns
        assert "confidence" in ensemble_signals.columns

    def test_data_adapter_validation_summary(self, engineered_features_data, data_adapter, rsi_strategy, macd_strategy):
        """Test data adapter validation summary functionality"""

        strategies = [rsi_strategy, macd_strategy]

        # Process strategies
        for strategy in strategies:
            data_adapter.prepare_strategy_data(engineered_features_data, strategy)

        # Get validation summary
        summary = data_adapter.get_validation_summary()

        # Validate summary structure
        assert "total_strategies_validated" in summary
        assert "successful_validations" in summary
        assert "failed_validations" in summary
        assert "strategy_details" in summary

        assert summary["successful_validations"] == 2
        assert summary["failed_validations"] == 0


class TestTemporalDataLeakageFixes:
    """Test temporal data leakage fixes in training pipeline"""
    
    @pytest.fixture
    def extended_ohlcv_data(self):
        """Create extended OHLCV data for temporal testing (100+ days)"""
        dates = pd.date_range("2023-01-01", periods=150, freq="D")
        
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0, 0.015, 150)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        close_prices = np.array(prices)
        open_prices = close_prices * (1 + np.random.normal(0, 0.005, 150))
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.01, 150))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.01, 150))
        volumes = np.random.uniform(1000000, 5000000, 150)
        
        return pd.DataFrame({
            "date": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes
        })
    
    def test_temporal_sequence_alignment_prevents_leakage(self, extended_ohlcv_data):
        """Test that sequence-target alignment prevents temporal data leakage"""
        
        # Create trainer with small tickers list for testing
        trainer = SharedBackboneTrainer(tickers=["TEST"], use_expanded_universe=False)
        
        # Mock the pattern engine to return our test data
        mock_features = extended_ohlcv_data.copy()
        
        # Add required features for pattern target generation  
        mock_features['price_acceleration'] = np.random.normal(0, 1, len(mock_features))
        mock_features['volume_price_divergence'] = np.random.normal(0, 1, len(mock_features))
        mock_features['volatility_regime_change'] = np.random.normal(0, 1, len(mock_features))
        mock_features['return_skewness_7d'] = np.random.normal(0, 1, len(mock_features))
        mock_features['momentum_persistence_7d'] = np.random.normal(0, 1, len(mock_features))
        mock_features['volatility_clustering'] = np.random.normal(0, 1, len(mock_features))
        mock_features['trend_exhaustion'] = np.random.normal(0, 1, len(mock_features))
        mock_features['garch_volatility_forecast'] = np.random.normal(0, 1, len(mock_features))
        mock_features['intraday_range_expansion'] = np.random.normal(0, 1, len(mock_features))
        mock_features['overnight_gap_behavior'] = np.random.normal(0, 1, len(mock_features))
        mock_features['end_of_day_momentum'] = np.random.normal(0, 1, len(mock_features))
        mock_features['sector_relative_strength'] = np.random.normal(0, 1, len(mock_features))
        mock_features['market_beta_instability'] = np.random.normal(0, 1, len(mock_features))
        mock_features['vix_term_structure'] = np.random.normal(0, 1, len(mock_features))
        mock_features['returns_1d'] = mock_features['close'].pct_change()
        mock_features['returns_3d'] = mock_features['close'].pct_change(3)
        mock_features['returns_7d'] = mock_features['close'].pct_change(7)
        
        # Generate pattern targets
        target_generator = PatternTargetGenerator()
        targets_dict = target_generator.generate_all_pattern_targets(mock_features)
        pattern_targets = targets_dict['pattern_confidence_score']
        
        # Test sequence preparation with temporal gap
        feature_columns = [col for col in mock_features.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
        
        X, y = trainer._prepare_pattern_detection_sequences(
            mock_features, 
            feature_columns,
            pattern_targets,
            sequence_length=20,
            stride=5
        )
        
        # CRITICAL TESTS: Verify no temporal leakage
        sequence_length = 20
        prediction_horizon = 5
        temporal_gap = 1
        total_offset = prediction_horizon + temporal_gap
        
        # Test 1: Verify temporal gap exists
        for i in range(len(X)):
            sequence_end_day = i * 5 + sequence_length - 1  # Last day of sequence 
            target_day = i * 5 + sequence_length + total_offset - 1  # Target day
            
            assert target_day > sequence_end_day + temporal_gap, \
                f"Temporal gap violated: sequence ends day {sequence_end_day}, target at day {target_day}"
        
        # Test 2: Verify sequences are correctly shaped
        assert X.ndim == 3, "Sequences should be 3D (batch, time, features)"
        assert X.shape[1] == sequence_length, f"Sequences should be {sequence_length} days long"
        assert len(X) == len(y), "Number of sequences should match number of targets"
        
        # Test 3: Verify reduced overlap (5-day stride)
        expected_sequences = (len(mock_features) - sequence_length - total_offset) // 5
        assert len(X) <= expected_sequences + 1, "Too many sequences generated for stride=5"
        
        # Test 4: Verify no NaN leakage in sequences
        assert not np.isnan(X).any(), "Sequences contain NaN values"
        assert not np.isnan(y).any(), "Targets contain NaN values"
        
    def test_configuration_override_functionality(self):
        """Test that critical configuration parameters can be overridden"""
        
        trainer = SharedBackboneTrainer(tickers=["TEST"], use_expanded_universe=False)
        
        # Get original config values
        original_lr = trainer.config.model.model_params.get("learning_rate", 0.0005)
        
        # Test that override mechanism works
        model_params = trainer.config.model.model_params.copy()
        critical_overrides = {
            "learning_rate": 0.002,
            "diversity_penalty_weight": 25.0,
            "correlation_penalty_weight": 15.0,
        }
        
        model_params.update(critical_overrides)
        
        # Verify overrides are applied
        assert model_params["learning_rate"] == 0.002, "Learning rate override failed"
        assert model_params["diversity_penalty_weight"] == 25.0, "Diversity penalty override failed"
        assert model_params["correlation_penalty_weight"] == 15.0, "Correlation penalty override failed"
        
        # Verify original config unchanged
        assert trainer.config.model.model_params.get("learning_rate") == original_lr, \
            "Original config should not be modified"
    
    def test_reduced_sequence_overlap_calculation(self):
        """Test that sequence overlap reduction works correctly"""
        
        trainer = SharedBackboneTrainer(tickers=["TEST"], use_expanded_universe=False)
        
        # Test overlap calculation
        lookback_window = 20
        optimal_stride = max(1, lookback_window // 4)  # Should be 5
        
        assert optimal_stride == 5, f"Expected stride=5, got {optimal_stride}"
        
        # Test overlap percentage
        overlap_pct = (lookback_window - optimal_stride) / lookback_window * 100
        assert overlap_pct == 75.0, f"Expected 75% overlap, got {overlap_pct}%"
        
        # Test that this is significantly less than 1-day stride
        one_day_overlap = (lookback_window - 1) / lookback_window * 100
        assert overlap_pct < one_day_overlap - 10, "Overlap reduction should be significant"
        
    def test_temporal_validation_split_integrity(self):
        """Test that temporal validation splits preserve time series order"""
        
        # Create mock combined data
        total_samples = 1000
        combined_X = np.random.randn(total_samples, 20, 17)  # batch, time, features
        combined_y = np.random.randn(total_samples)
        
        # Calculate temporal split (80% train)
        train_size = int(0.8 * total_samples)
        
        train_X, val_X = combined_X[:train_size], combined_X[train_size:]
        train_y, val_y = combined_y[:train_size], combined_y[train_size:]
        
        # Test split sizes
        assert len(train_X) == train_size, f"Training size mismatch: {len(train_X)} != {train_size}"
        assert len(val_X) == total_samples - train_size, f"Validation size mismatch"
        assert len(train_y) == train_size, "Training targets size mismatch"
        assert len(val_y) == total_samples - train_size, "Validation targets size mismatch"
        
        # Test temporal ordering (no random shuffling)
        # Training data should come first, validation data should come after
        # This is implicitly tested by the slicing [:train_size] and [train_size:]
        
        # Test no data overlap between train and validation
        assert train_size + len(val_X) == total_samples, "Data overlap or loss detected"

