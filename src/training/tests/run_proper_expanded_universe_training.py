#!/usr/bin/env python3

"""
Direct Expanded Universe LSTM Training
Bypasses the hardcoded MAG7 pipeline to properly train on 34 securities
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from config.config import get_config
from tests.data_loader import load_test_data, validate_data_format
from src.training.shared_backbone_trainer import create_shared_backbone_trainer

def run_proper_expanded_universe_training():
    """Execute proper expanded universe training with 34 securities"""
    
    print("=" * 80)
    print("PROPER EXPANDED UNIVERSE LSTM TRAINING")
    print("Direct training on 34 securities (bypassing MAG7 pipeline)")
    print("=" * 80)
    print()
    
    try:
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"lstm_training_runs/expanded_universe_{timestamp}")
        
        # Create subdirectories
        subdirs = ["data", "models", "evaluation", "logs", "artifacts"]
        for subdir in subdirs:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {output_dir}")
        print()
        
        # STAGE 1: Initialize trainer with expanded universe
        print("STAGE 1: Initialize Expanded Universe Trainer")
        print("-" * 50)
        
        trainer = create_shared_backbone_trainer(
            tickers=None,  # Use config default
            use_expanded_universe=True
        )
        
        print(f"Training universe: {len(trainer.tickers)} securities")
        print(f"Securities: {trainer.tickers}")
        print()
        
        # STAGE 2: Load market data for all 34 tickers
        print("STAGE 2: Load Market Data (34 tickers)")
        print("-" * 50)
        
        print("Loading 400 days of data for expanded universe...")
        ticker_data = load_test_data(trainer.tickers, days=400)
        
        print(f"Successfully loaded data for {len(ticker_data)} tickers")
        
        # Validate data
        if not validate_data_format(ticker_data):
            raise ValueError("Data validation failed")
        
        # Calculate expected sequences
        total_expected_sequences = 0
        for ticker, df in ticker_data.items():
            # Each ticker: ~400 days - 20 (lookback) = 380 available samples
            # With stride=5: 380/5 = 76 sequences per ticker
            expected_sequences = max(0, (len(df) - 20) // 5)
            total_expected_sequences += expected_sequences
            print(f"  {ticker}: {len(df)} records → ~{expected_sequences} sequences")
        
        print(f"Expected total sequences: ~{total_expected_sequences:,}")
        print(f"Training data multiplier vs MAG7: {total_expected_sequences/740:.1f}x")
        print()
        
        # STAGE 3: Prepare training data
        print("STAGE 3: Prepare Training Data")
        print("-" * 50)
        
        training_data = trainer.prepare_training_data(ticker_data)
        
        # Calculate actual sequences generated
        actual_sequences = sum(len(X) for X, y in training_data.values())
        print(f"Actual sequences generated: {actual_sequences:,}")
        print(f"Success rate: {len(training_data)}/{len(trainer.tickers)} tickers")
        print()
        
        # STAGE 4: Train shared backbone
        print("STAGE 4: Train Shared Backbone LSTM")
        print("-" * 50)
        
        training_results = trainer.train_shared_backbone(
            training_data=training_data,
            validation_split=0.2,
            epochs=50
        )
        
        # Save model
        model_path = output_dir / "models" / "expanded_universe_lstm.keras"
        training_results["model"].save(str(model_path))
        print(f"Model saved: {model_path}")
        
        # STAGE 5: Cross-ticker validation
        print("STAGE 5: Cross-Ticker Validation")
        print("-" * 50)
        
        validation_results = trainer.validate_cross_ticker_performance(
            training_data, training_results["model"]
        )
        
        # STAGE 6: Generate comprehensive results
        print("STAGE 6: Generate Results")
        print("-" * 50)
        
        # Compile comprehensive results
        pipeline_results = {
            "training_config": {
                "universe": "expanded",
                "tickers": trainer.tickers,
                "num_tickers": len(trainer.tickers),
                "expected_sequences": total_expected_sequences,
                "actual_sequences": actual_sequences,
                "sequence_success_rate": actual_sequences / total_expected_sequences if total_expected_sequences > 0 else 0,
                "epochs": 50
            },
            "training_results": {
                "final_metrics": training_results["final_metrics"],
                "model_params": training_results["model"].count_params(),
                "training_stable": training_results["training_stable"]
            },
            "validation_results": validation_results,
            "comparison_vs_mag7": {
                "mag7_sequences": 740,
                "expanded_sequences": actual_sequences,
                "multiplier": actual_sequences / 740,
                "mag7_accuracy": 0.403,  # From previous run
                "expanded_accuracy": training_results["final_metrics"]["pattern_detection_accuracy"]
            }
        }
        
        # Save results
        results_path = output_dir / "evaluation" / "expanded_universe_results.json"
        with open(results_path, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        # Generate report
        report = trainer.generate_training_report()
        report_path = output_dir / "evaluation" / "expanded_universe_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print()
        print("=" * 80)
        print("EXPANDED UNIVERSE TRAINING COMPLETED")
        print("=" * 80)
        print(f"Results: {results_path}")
        print(f"Report: {report_path}")
        print()
        
        # Print key metrics
        final_metrics = training_results["final_metrics"]
        print("KEY PERFORMANCE METRICS:")
        print(f"  Pattern Detection Accuracy: {final_metrics['pattern_detection_accuracy']:.1%}")
        print(f"  Cross-Ticker Generalization: {validation_results['overall_stats']['pattern_generalization_score']:.1%}")
        print(f"  Training Sequences: {actual_sequences:,}")
        print(f"  Improvement vs MAG7: {(final_metrics['pattern_detection_accuracy']/0.403 - 1)*100:+.1f}%")
        print()
        
        return True, pipeline_results
        
    except Exception as e:
        print(f"ERROR: Expanded universe training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, results = run_proper_expanded_universe_training()
    exit(0 if success else 1)