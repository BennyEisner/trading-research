#!/usr/bin/env python3

"""
LSTM Shared Backbone Training Script
Runs the complete LSTM pattern detection training pipeline using existing infrastructure
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import get_config, load_config
from tests.utilities.data_loader import load_test_data, validate_data_format
from src.training.shared_backbone_trainer import create_shared_backbone_trainer


class LSTMTrainingPipeline:
    """
    LSTM training pipeline following established infrastructure patterns
    """
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path) if config_path else get_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pipeline_results = {}
        
        # Create output directory structure following existing pattern
        self.create_output_structure()
    
    def create_output_structure(self):
        """Create output directory structure following production pipeline pattern"""
        self.output_dir = Path(f"lstm_training_runs/{self.timestamp}")
        
        # Create subdirectories matching production pattern
        subdirs = ["data", "models", "evaluation", "logs", "artifacts"]
        
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        print(f"Created output structure: {self.output_dir}")
    
    def stage_1_initialize_trainer(self):
        """Stage 1: Initialize LSTM trainer with configuration"""
        print(f"\n{'='*60}")
        print(f"STAGE 1: INITIALIZE LSTM TRAINER")
        print(f"{'='*60}")
        
        try:
            # Use MAG7 tickers from config
            tickers = self.config.model.mag7_tickers
            
            self.trainer = create_shared_backbone_trainer(
                tickers=tickers,
                use_expanded_universe=False
            )
            
            print(f"Trainer initialized successfully")
            print(f"  - Tickers: {len(self.trainer.tickers)} ({', '.join(tickers)})")
            print(f"  - Lookback window: {self.config.model.lookback_window}")
            print(f"  - Sequence stride: {self.config.model.sequence_stride}")
            print(f"  - Model parameters: {self.config.model.model_params}")
            
            self.pipeline_results["trainer_initialization"] = {
                "success": True,
                "tickers": tickers,
                "config_params": {
                    "lookback_window": self.config.model.lookback_window,
                    "sequence_stride": self.config.model.sequence_stride,
                    "prediction_horizon": self.config.model.prediction_horizon
                }
            }
            
            return True
            
        except Exception as e:
            print(f"STAGE 1 FAILED: {e}")
            self.pipeline_results["trainer_initialization"] = {"success": False, "error": str(e)}
            return False
    
    def stage_2_load_data(self):
        """Stage 2: Load and validate market data"""
        print(f"\n{'='*60}")
        print(f"STAGE 2: LOAD MARKET DATA")
        print(f"{'='*60}")
        
        try:
            tickers = self.config.model.mag7_tickers
            # Load substantial data for robust training (2+ years)
            days = 800
            
            print(f"Loading {days} days of data for {len(tickers)} tickers...")
            self.ticker_data = load_test_data(tickers, days=days)
            
            if not validate_data_format(self.ticker_data):
                raise ValueError("Data format validation failed")
            
            # Log data statistics
            data_stats = {
                "tickers": list(self.ticker_data.keys()),
                "total_records": sum(len(df) for df in self.ticker_data.values()),
                "date_ranges": {}
            }
            
            print(f"Data loaded successfully for {len(self.ticker_data)} tickers:")
            for ticker, df in self.ticker_data.items():
                date_range = {
                    "start": str(df['date'].min()),
                    "end": str(df['date'].max()),
                    "count": len(df)
                }
                data_stats["date_ranges"][ticker] = date_range
                print(f"  - {ticker}: {len(df)} records ({date_range['start']} to {date_range['end']})")
            
            # Save data statistics
            with open(self.output_dir / "data" / "data_stats.json", 'w') as f:
                json.dump(data_stats, f, indent=2)
            
            self.pipeline_results["data_loading"] = {
                "success": True,
                "stats": data_stats
            }
            
            return True
            
        except Exception as e:
            print(f"STAGE 2 FAILED: {e}")
            self.pipeline_results["data_loading"] = {"success": False, "error": str(e)}
            return False
    
    def stage_3_prepare_training_data(self):
        """Stage 3: Prepare training data with features and patterns"""
        print(f"\n{'='*60}")
        print(f"STAGE 3: PREPARE TRAINING DATA")
        print(f"{'='*60}")
        
        try:
            print("Calculating features, generating pattern targets, and creating sequences...")
            self.training_data = self.trainer.prepare_training_data(self.ticker_data)
            
            if not self.training_data:
                raise ValueError("No training data prepared")
            
            total_sequences = sum(len(X) for X, y in self.training_data.values())
            
            print(f"Training data prepared successfully:")
            print(f"  - Successful tickers: {len(self.training_data)}")
            print(f"  - Total sequences: {total_sequences}")
            
            preparation_stats = {
                "successful_tickers": len(self.training_data),
                "total_sequences": total_sequences,
                "ticker_details": {}
            }
            
            for ticker, (X, y) in self.training_data.items():
                ticker_stats = {
                    "sequences": len(X),
                    "feature_shape": list(X.shape),
                    "target_range": [float(y.min()), float(y.max())],
                    "target_mean": float(y.mean())
                }
                preparation_stats["ticker_details"][ticker] = ticker_stats
                print(f"  - {ticker}: X{X.shape}, y{y.shape}, y_range=[{y.min():.3f}, {y.max():.3f}]")
            
            self.pipeline_results["training_data_preparation"] = {
                "success": True,
                "stats": preparation_stats
            }
            
            return True
            
        except Exception as e:
            print(f"STAGE 3 FAILED: {e}")
            self.pipeline_results["training_data_preparation"] = {"success": False, "error": str(e)}
            import traceback
            traceback.print_exc()
            return False
    
    def stage_4_train_model(self, epochs: int = None):
        """Stage 4: Train the shared backbone LSTM model"""
        print(f"\n{'='*60}")
        print(f"STAGE 4: TRAIN SHARED BACKBONE LSTM")
        print(f"{'='*60}")
        
        try:
            # Use config values or provided epochs
            training_epochs = epochs if epochs is not None else self.config.model.training_params.get("epochs", 50)
            validation_split = self.config.model.training_params.get("validation_split", 0.2)
            
            print(f"Training configuration:")
            print(f"  - Epochs: {training_epochs}")
            print(f"  - Validation split: {validation_split}")
            print(f"  - Model parameters: ~{sum(len(X) * X.shape[1] * X.shape[2] for X, y in self.training_data.values())} total parameters")
            
            print(f"\nStarting LSTM training...")
            self.training_results = self.trainer.train_shared_backbone(
                training_data=self.training_data,
                validation_split=validation_split,
                epochs=training_epochs
            )
            
            model = self.training_results["model"]
            final_metrics = self.training_results["final_metrics"]
            
            print(f"LSTM training completed successfully:")
            print(f"  - Model parameters: {model.count_params():,}")
            print(f"  - Final train loss: {final_metrics['train_loss']:.4f}")
            print(f"  - Final val loss: {final_metrics['val_loss']:.4f}")
            print(f"  - Pattern detection accuracy: {final_metrics['pattern_detection_accuracy']:.3f}")
            print(f"  - Correlation: {final_metrics['correlation']:.3f}")
            
            self.pipeline_results["model_training"] = {
                "success": True,
                "model_params": model.count_params(),
                "final_metrics": final_metrics,
                "training_stable": self.training_results["training_stable"]
            }
            
            return True
            
        except Exception as e:
            print(f"STAGE 4 FAILED: {e}")
            self.pipeline_results["model_training"] = {"success": False, "error": str(e)}
            import traceback
            traceback.print_exc()
            return False
    
    def stage_5_validate_performance(self):
        """Stage 5: Cross-ticker validation and performance assessment"""
        print(f"\n{'='*60}")
        print(f"STAGE 5: VALIDATE PERFORMANCE")
        print(f"{'='*60}")
        
        try:
            model = self.training_results["model"]
            
            print("Running cross-ticker performance validation...")
            validation_results = self.trainer.validate_cross_ticker_performance(
                self.training_data, model
            )
            
            overall_stats = validation_results["overall_stats"]
            
            print(f"Cross-ticker validation completed:")
            print(f"  - Mean pattern accuracy: {overall_stats['mean_pattern_detection_accuracy']:.3f} ± {overall_stats['std_pattern_detection_accuracy']:.3f}")
            print(f"  - Mean correlation: {overall_stats['mean_correlation']:.3f} ± {overall_stats['std_correlation']:.3f}")
            print(f"  - Pattern generalization score: {overall_stats['pattern_generalization_score']:.3f}")
            print(f"  - Successful tickers: {overall_stats['successful_tickers']}")
            
            self.pipeline_results["performance_validation"] = {
                "success": True,
                "overall_stats": overall_stats,
                "validation_results": validation_results
            }
            
            return True
            
        except Exception as e:
            print(f"STAGE 5 FAILED: {e}")
            self.pipeline_results["performance_validation"] = {"success": False, "error": str(e)}
            return False
    
    def stage_6_save_artifacts(self):
        """Stage 6: Save models, results, and reports"""
        print(f"\n{'='*60}")
        print(f"STAGE 6: SAVE ARTIFACTS")
        print(f"{'='*60}")
        
        try:
            model = self.training_results["model"]
            
            # Save trained model
            model_path = self.output_dir / "models" / "shared_backbone_lstm.keras"
            model.save(str(model_path))
            print(f"Saved model: {model_path}")
            
            # Save configuration used for training
            config_path = self.output_dir / "models" / "training_config.json"
            config_data = {
                "timestamp": self.timestamp,
                "model_config": {
                    "lookback_window": self.config.model.lookback_window,
                    "sequence_stride": self.config.model.sequence_stride,
                    "prediction_horizon": self.config.model.prediction_horizon,
                    "model_params": self.config.model.model_params,
                    "training_params": self.config.model.training_params
                },
                "tickers": self.config.model.mag7_tickers
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            print(f"Saved config: {config_path}")
            
            # Save complete pipeline results
            results_path = self.output_dir / "evaluation" / "pipeline_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.pipeline_results, f, indent=2, default=str)
            print(f"Saved results: {results_path}")
            
            # Generate and save training report
            report = self.trainer.generate_training_report()
            report_path = self.output_dir / "evaluation" / "training_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Saved report: {report_path}")
            
            return True
            
        except Exception as e:
            print(f"STAGE 6 FAILED: {e}")
            return False
    
    def run_full_pipeline(self, epochs: int = None):
        """Run the complete LSTM training pipeline"""
        print("LSTM SHARED BACKBONE TRAINING PIPELINE")
        print("=" * 60)
        print("Training LSTM for pattern detection on MAG7 stocks")
        
        stages = [
            self.stage_1_initialize_trainer,
            self.stage_2_load_data,
            self.stage_3_prepare_training_data,
            lambda: self.stage_4_train_model(epochs),
            self.stage_5_validate_performance,
            self.stage_6_save_artifacts,
        ]
        
        for i, stage in enumerate(stages, 1):
            if not stage():
                print(f"\nPIPELINE FAILED AT STAGE {i}")
                return False
        
        # Final summary
        final_metrics = self.pipeline_results["model_training"]["final_metrics"]
        overall_stats = self.pipeline_results["performance_validation"]["overall_stats"]
        
        print(f"\n{'='*60}")
        print("LSTM TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        print(f"\nKEY RESULTS:")
        print(f"  - Pattern Detection Accuracy: {final_metrics['pattern_detection_accuracy']:.1%}")
        print(f"  - Cross-Ticker Generalization: {overall_stats['pattern_generalization_score']:.1%}")
        print(f"  - Model Size: {self.pipeline_results['model_training']['model_params']:,} parameters")
        print(f"  - Training Sequences: {self.pipeline_results['training_data_preparation']['stats']['total_sequences']:,}")
        
        print(f"\nOUTPUT DIRECTORY: {self.output_dir}")
        print(f"  - Model: models/shared_backbone_lstm.keras")
        print(f"  - Results: evaluation/pipeline_results.json")
        print(f"  - Report: evaluation/training_report.txt")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Run LSTM Shared Backbone Training Pipeline")
    parser.add_argument("--config", help="Configuration file path (YAML)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (overrides config)")
    parser.add_argument("--quick-test", action="store_true", help="Quick test with fewer epochs")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = LSTMTrainingPipeline(args.config)
        
        # Set epochs based on arguments
        epochs = None
        if args.quick_test:
            epochs = 5
            print("Running QUICK TEST pipeline (5 epochs)")
        elif args.epochs:
            epochs = args.epochs
            print(f"Using custom epochs: {epochs}")
        
        # Run pipeline
        success = pipeline.run_full_pipeline(epochs)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\nPIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)