#!/usr/bin/env python3

"""
Run expanded universe LSTM training
"""

import sys
from pathlib import Path

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from scripts.run_lstm_training import LSTMTrainingPipeline
from src.training.shared_backbone_trainer import create_shared_backbone_trainer


def run_expanded_universe_training():
    """Execute expanded universe LSTM training"""

    print("=" * 70)
    print("EXPANDED UNIVERSE LSTM TRAINING")
    print("=" * 70)
    print()

    try:
        # Create training pipeline
        pipeline = LSTMTrainingPipeline()

        print("Configuring trainer for expanded universe (34 securities)...")
        pipeline.trainer = create_shared_backbone_trainer(
            tickers=None, use_expanded_universe=True  # Use config default
        )

        print(f"Training universe: {len(pipeline.trainer.tickers)} securities")
        print(f"Securities: {pipeline.trainer.tickers}")
        print()

        expected_sequences = len(pipeline.trainer.tickers) * 100
        print(f"Expected training scale:")
        print(f"  Sequences: ~{expected_sequences:,} (vs 740 for MAG7)")
        print(f"  Multiplier: ~{expected_sequences/740:.1f}x more training data")
        print(f"  Expected training time: 15-25 minutes")
        print()

        # Run full pipeline
        print("Starting expanded universe training pipeline...")
        success = pipeline.run_full_pipeline(epochs=50)

        if success:
            print()
            print("=" * 70)
            print("EXPANDED UNIVERSE TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 70)

            results_path = pipeline.get_latest_results_path()
            print(f"Results saved to: {results_path}")

            return True
        else:
            print()
            print("=" * 70)
            print("EXPANDED UNIVERSE TRAINING FAILED")
            print("=" * 70)
            return False

    except Exception as e:
        print(f"ERROR: Expanded universe training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_expanded_universe_training()
    exit(0 if success else 1)

