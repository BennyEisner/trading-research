#!/usr/bin/env python3

"""
Test LSTM Trainer Initialization
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.shared_backbone_trainer import create_shared_backbone_trainer


def test_trainer_initialization():
    """Test that LSTM trainer can be initialized properly"""

    test_tickers = ["AAPL", "MSFT", "GOOG"]

    try:
        print("Testing LSTM Trainer Initialization...")
        print(f"Test tickers: {test_tickers}")

        # Create trainer with small test set
        trainer = create_shared_backbone_trainer(tickers=test_tickers, use_expanded_universe=False)

        # Validate trainer components
        assert trainer is not None, "Trainer should not be None"
        assert trainer.config is not None, "Config should be loaded"
        assert trainer.pattern_engine is not None, "Pattern engine should be initialized"
        assert trainer.lstm_builder is not None, "LSTM builder should be initialized"
        assert len(trainer.tickers) == len(
            test_tickers
        ), f"Expected {len(test_tickers)} tickers, got {len(trainer.tickers)}"

        print(f"Trainer initialized successfully")
        print(f"Tickers: {len(trainer.tickers)}")
        print(f"Config loaded: {trainer.config is not None}")
        print(f"Pattern engine: {trainer.pattern_engine is not None}")
        print(f"LSTM builder: {trainer.lstm_builder is not None}")
        print(f"Use expanded universe: {trainer.use_expanded_universe}")

        return True

    except Exception as e:
        print(f"✗ Trainer initialization FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_trainer_config_validation():
    """Test trainer configuration parameters for optimized infrastructure"""

    try:
        print("\nTesting Optimized Trainer Configuration...")

        trainer = create_shared_backbone_trainer(tickers=["AAPL"], use_expanded_universe=False)

        config = trainer.config

        # Validate key configuration parameters
        assert config.model.lookback_window > 0, "Lookback window should be positive"
        assert config.model.prediction_horizon > 0, "Prediction horizon should be positive"
        assert len(config.model.mag7_tickers) > 0, "Should have MAG7 tickers"
        
        # Validate optimized configuration
        model_params = config.model.model_params
        training_params = config.model.training_params
        
        assert model_params.get('dropout_rate', 0) >= 0.5, "Dropout should be >= 0.5 for high overlap"
        assert training_params.get('batch_size', 0) >= 256, "Batch size should be >= 256 for stability"
        assert training_params.get('monitor_metric') == 'val_loss', "Should monitor val_loss for high overlap"

        print(f"✓ Configuration validation PASSED")
        print(f"  - Lookback window: {config.model.lookback_window}")
        print(f"  - Prediction horizon: {config.model.prediction_horizon}")
        print(f"  - Model size: {config.model.model_size}")
        print(f"  - Dropout rate: {model_params['dropout_rate']}")
        print(f"  - Batch size: {training_params['batch_size']}")
        print(f"  - Monitor metric: {training_params['monitor_metric']}")
        print(f"  - MAG7 tickers: {len(config.model.mag7_tickers)}")
        print(f"  - Infrastructure: Optimized for 1-day stride, 95% overlap")

        return True

    except Exception as e:
        print(f"✗ Configuration validation FAILED: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("LSTM TRAINER INITIALIZATION TESTS")
    print("=" * 60)

    success = True

    # Test basic initialization
    if not test_trainer_initialization():
        success = False

    # Test configuration
    if not test_trainer_config_validation():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("ALL TRAINER TESTS PASSED ✓")
    else:
        print("SOME TRAINER TESTS FAILED ✗")
    print("=" * 60)

    sys.exit(0 if success else 1)

