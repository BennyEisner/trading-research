#!/usr/bin/env python3

"""
Comprehensive Training Test for Updated Pipeline
Tests complete training process with new infrastructure:
- 1-day stride, 95% overlap sequences
- Medium-large model (80-40-20, ~52k params) 
- Enhanced validation monitoring
- High-overlap training considerations
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.data_loader import load_test_data, validate_data_format
from src.training.shared_backbone_trainer import SharedBackboneTrainer
from config.config import get_config


class ComprehensiveTrainingTest:
    """
    Comprehensive test for optimized training pipeline
    Tests all stages from data loading to model validation
    """
    
    def __init__(self, test_mode: str = "fast"):
        """
        Initialize comprehensive training test
        
        Args:
            test_mode: "fast" (4 tickers, 3 years, 20 epochs) or "full" (expanded universe, 10 years, full epochs)
        """
        self.test_mode = test_mode
        self.config = get_config()
        
        # Always use the complete expanded universe - that's the point of comprehensive testing
        expanded_universe = [t for t in self.config.model.expanded_universe if t != 'VIX']
        self.test_tickers = expanded_universe  # All 33 tickers for both modes
        self.use_expanded_universe = True
        
        if test_mode == "fast":
            self.test_days = 1825  # 5 years for faster testing
            self.test_epochs = 25
        else:  # full
            self.test_days = 3650  # 10 years
            self.test_epochs = 50
        
        self.results = {
            "test_config": {
                "mode": test_mode,
                "tickers": len(self.test_tickers),
                "days": self.test_days,
                "epochs": self.test_epochs,
                "infrastructure": "optimized_1day_stride"
            },
            "stage_results": {},
            "performance_metrics": {},
            "validation_results": {},
            "errors": []
        }
        
        print(f"=== COMPREHENSIVE TRAINING TEST ({test_mode.upper()} MODE) ===")
        print(f"Configuration:")
        print(f"  - Tickers: {len(self.test_tickers)} ({self.test_tickers[:5]}{'...' if len(self.test_tickers) > 5 else ''})")
        print(f"  - Data period: {self.test_days} days")
        print(f"  - Training epochs: {self.test_epochs}")
        print(f"  - Model: {self.config.model.model_size} ({self.config.model.model_params['lstm_units_1']}-{self.config.model.model_params['lstm_units_2']}-{self.config.model.model_params['dense_units']})")
        print(f"  - Expected infrastructure: 1-day stride, 95% overlap, batch size {self.config.model.training_params['batch_size']}")
        print()

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run complete training test pipeline"""
        
        try:
            print("STAGE 1: Infrastructure Validation")
            print("-" * 50)
            success = self._test_infrastructure_configuration()
            self.results["stage_results"]["infrastructure"] = success
            if not success:
                return self.results
            
            print("\nSTAGE 2: Data Loading & Validation")
            print("-" * 50)
            success = self._test_data_loading()
            self.results["stage_results"]["data_loading"] = success
            if not success:
                return self.results
            
            print("\nSTAGE 3: Training Data Preparation")
            print("-" * 50)
            success = self._test_training_data_preparation()
            self.results["stage_results"]["training_data_prep"] = success
            if not success:
                return self.results
            
            print("\nSTAGE 4: Model Training with High-Overlap Monitoring")
            print("-" * 50)
            success = self._test_model_training()
            self.results["stage_results"]["model_training"] = success
            if not success:
                return self.results
            
            print("\nSTAGE 5: Training Validation & Performance Analysis")
            print("-" * 50)
            success = self._test_training_validation()
            self.results["stage_results"]["training_validation"] = success
            
            # Overall success
            self.results["overall_success"] = all(self.results["stage_results"].values())
            
            print("\n" + "=" * 80)
            if self.results["overall_success"]:
                print("🎉 ALL STAGES PASSED - READY FOR PHASE 2 INTEGRATION")
            else:
                print("❌ SOME STAGES FAILED - REVIEW ISSUES BEFORE PROCEEDING")
            print("=" * 80)
            
        except Exception as e:
            error_msg = f"Unexpected error in comprehensive test: {e}"
            self.results["errors"].append(error_msg)
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            self.results["overall_success"] = False
        
        return self.results

    def _test_infrastructure_configuration(self) -> bool:
        """Test that infrastructure is configured correctly for optimized training"""
        try:
            print("Validating optimized training infrastructure...")
            
            # Check model configuration
            model_params = self.config.model.model_params
            training_params = self.config.model.training_params
            
            # Validate medium-large configuration
            expected_architecture = (80, 40, 20) if self.config.model.model_size == "medium-large" else None
            if expected_architecture:
                actual_architecture = (model_params['lstm_units_1'], model_params['lstm_units_2'], model_params['dense_units'])
                if actual_architecture != expected_architecture:
                    print(f"  ❌ Model architecture mismatch: expected {expected_architecture}, got {actual_architecture}")
                    return False
                print(f"  ✅ Model architecture: {actual_architecture}")
            
            # Validate high-overlap training configuration
            checks = [
                ("Batch size >= 256", training_params.get('batch_size', 0) >= 256),
                ("Dropout rate >= 0.5", model_params.get('dropout_rate', 0) >= 0.5),
                ("L2 regularization >= 0.008", model_params.get('l2_regularization', 0) >= 0.008),
                ("Monitor val_loss", training_params.get('monitor_metric') == 'val_loss'),
                ("High patience", training_params.get('patience', 0) >= 12)
            ]
            
            for check_name, check_result in checks:
                status = "✅" if check_result else "❌"
                print(f"  {status} {check_name}")
                if not check_result:
                    return False
            
            print("✅ Infrastructure configuration validated")
            return True
            
        except Exception as e:
            print(f"❌ Infrastructure validation failed: {e}")
            return False

    def _test_data_loading(self) -> bool:
        """Test data loading with updated data loader"""
        try:
            print(f"Loading {self.test_days} days for {len(self.test_tickers)} tickers...")
            
            self.ticker_data = load_test_data(self.test_tickers, days=self.test_days)
            
            if not self.ticker_data:
                print("❌ No data loaded")
                return False
            
            # Validate data format
            if not validate_data_format(self.ticker_data):
                print("❌ Data format validation failed")
                return False
            
            # Calculate expected sequences with 1-day stride
            total_expected = 0
            for ticker, data in self.ticker_data.items():
                expected = len(data) - 20 - 5 + 1  # seq_len + pred_horizon - 1
                total_expected += expected
                print(f"  {ticker}: {len(data)} days → ~{expected} sequences")
            
            self.expected_total_sequences = total_expected
            print(f"✅ Data loading successful: {len(self.ticker_data)} tickers, ~{total_expected:,} expected sequences")
            return True
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            self.results["errors"].append(f"Data loading error: {e}")
            return False

    def _test_training_data_preparation(self) -> bool:
        """Test training data preparation with 1-day stride"""
        try:
            print("Preparing training data with 1-day stride...")
            
            # Initialize trainer
            self.trainer = SharedBackboneTrainer(
                tickers=list(self.ticker_data.keys()),
                use_expanded_universe=self.use_expanded_universe
            )
            
            # Prepare training data (this uses 1-day stride internally)
            self.training_data = self.trainer.prepare_training_data(self.ticker_data)
            
            if not self.training_data:
                print("❌ No training data prepared")
                return False
            
            # Analyze prepared data
            total_sequences = sum(len(X) for X, y in self.training_data.values())
            
            # Validate 1-day stride implementation
            overlap_check = True
            for ticker, (X, y) in self.training_data.items():
                ticker_days = len(self.ticker_data[ticker])
                expected_sequences = ticker_days - 20 - 5 + 1
                actual_sequences = len(X)
                
                # Should be very close for 1-day stride (allow reasonable tolerance for data differences)
                tolerance = max(10, expected_sequences * 0.02)  # 2% tolerance or 10 sequences, whichever is larger
                if abs(actual_sequences - expected_sequences) > tolerance:
                    print(f"  ❌ {ticker}: Expected ~{expected_sequences}, got {actual_sequences} (significant difference)")
                    overlap_check = False
                else:
                    overlap_pct = ((20 - 1) / 20) * 100  # 95% for 1-day stride
                    if abs(actual_sequences - expected_sequences) > 5:
                        print(f"  ⚠️  {ticker}: {actual_sequences} sequences ({overlap_pct:.0f}% overlap) - minor data difference")
                    else:
                        print(f"  ✅ {ticker}: {actual_sequences} sequences ({overlap_pct:.0f}% overlap)")
            
            if not overlap_check:
                print("❌ 1-day stride validation failed")
                return False
            
            # Data quality checks
            for ticker, (X, y) in self.training_data.items():
                if X.shape[2] != 17:  # Should have 17 features
                    print(f"❌ {ticker}: Expected 17 features, got {X.shape[2]}")
                    return False
                
                # Check target distribution (should be continuous 0-1)
                y_min, y_max = np.min(y), np.max(y)
                if y_min < 0 or y_max > 1:
                    print(f"❌ {ticker}: Target range [{y_min:.3f}, {y_max:.3f}] outside [0,1]")
                    return False
                
                # Check for reasonable target diversity
                unique_targets = len(np.unique(np.round(y, 3)))
                if unique_targets < 10:  # Should have diverse continuous targets
                    print(f"❌ {ticker}: Only {unique_targets} unique target values (too few)")
                    return False
            
            # Calculate data/parameter ratio (this should now be adequate with full universe)
            approx_model_params = 52000  # Medium-large model ~52k params
            data_param_ratio = total_sequences / approx_model_params
            
            print(f"✅ Training data preparation successful:")
            print(f"    - Total sequences: {total_sequences:,}")
            print(f"    - Successful tickers: {len(self.training_data)}/{len(self.test_tickers)}")
            print(f"    - Model parameters: ~{approx_model_params:,}")
            print(f"    - Data/parameter ratio: {data_param_ratio:.1f}x")
            print(f"    - 1-day stride validated: 95% overlap confirmed")
            
            if data_param_ratio < 1.5:
                print(f"    ⚠️  WARNING: Data/param ratio {data_param_ratio:.1f}x below recommended 1.5x")
            
            self.results["performance_metrics"]["total_sequences"] = total_sequences
            self.results["performance_metrics"]["successful_tickers"] = len(self.training_data)
            self.results["performance_metrics"]["data_param_ratio"] = data_param_ratio
            
            return True
            
        except Exception as e:
            print(f"❌ Training data preparation failed: {e}")
            self.results["errors"].append(f"Training data prep error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _test_model_training(self) -> bool:
        """Test model training with high-overlap sequence monitoring"""
        try:
            print("Training model with high-overlap monitoring...")
            print("⚠️  WARNING: Training loss will drop deceptively fast due to 95% overlap")
            print("⚠️  FOCUS: Validation loss is the true performance indicator")
            
            # Train model
            self.training_results = self.trainer.train_shared_backbone(
                training_data=self.training_data,
                validation_split=0.2,
                epochs=self.test_epochs
            )
            
            if "model" not in self.training_results:
                print("❌ No model in training results")
                return False
            
            model = self.training_results["model"]
            history = self.training_results.get("history", {})
            
            # Analyze training behavior
            if history and "loss" in history and "val_loss" in history:
                train_loss = history["loss"]
                val_loss = history["val_loss"]
                
                print(f"✅ Training completed: {len(train_loss)} epochs")
                print(f"    - Model parameters: {model.count_params():,}")
                print(f"    - Final train loss: {train_loss[-1]:.4f}")
                print(f"    - Final val loss: {val_loss[-1]:.4f}")
                
                # High-overlap validation checks
                train_improvement = (train_loss[0] - train_loss[-1]) / train_loss[0]
                val_improvement = (val_loss[0] - val_loss[-1]) / val_loss[0]
                
                print(f"    - Train loss improvement: {train_improvement:.1%}")
                print(f"    - Val loss improvement: {val_improvement:.1%}")
                
                # Warning if training loss drops too fast compared to validation
                if train_improvement > 0.8 and val_improvement < 0.3:
                    print("⚠️  HIGH-OVERLAP WARNING: Training loss dropped much faster than validation")
                    print("⚠️  This is expected with 95% overlap - focus on validation metrics")
                
                # Check for training stability
                if np.isnan(train_loss[-1]) or np.isnan(val_loss[-1]):
                    print("❌ Training resulted in NaN losses")
                    return False
                
                self.results["performance_metrics"]["final_train_loss"] = float(train_loss[-1])
                self.results["performance_metrics"]["final_val_loss"] = float(val_loss[-1])
                self.results["performance_metrics"]["train_improvement"] = float(train_improvement)
                self.results["performance_metrics"]["val_improvement"] = float(val_improvement)
                
            else:
                print("⚠️  No training history available for analysis")
            
            return True
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            self.results["errors"].append(f"Model training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _test_training_validation(self) -> bool:
        """Test training validation and performance analysis"""
        try:
            print("Validating training results...")
            
            model = self.training_results["model"]
            
            # Cross-ticker validation
            validation_results = self.trainer.validate_cross_ticker_performance(
                self.training_data, model
            )
            
            if not validation_results:
                print("❌ Cross-ticker validation failed")
                return False
            
            # Extract performance metrics
            ticker_accuracies = []
            for ticker, results in validation_results["ticker_performance"].items():
                acc = results.get("mean_pattern_detection_accuracy", 0)
                ticker_accuracies.append(acc)
                print(f"    - {ticker}: {acc:.3f} pattern detection accuracy")
            
            avg_accuracy = np.mean(ticker_accuracies)
            generalization_score = validation_results["overall_stats"]["pattern_generalization_score"]
            
            print(f"✅ Training validation results:")
            print(f"    - Average accuracy: {avg_accuracy:.3f}")
            print(f"    - Generalization score: {generalization_score:.3f}")
            print(f"    - Cross-ticker consistency: {np.std(ticker_accuracies):.3f} std")
            
            # Success criteria for optimized pipeline
            success_criteria = [
                ("Accuracy > 0.50", avg_accuracy > 0.50),
                ("Generalization > 0.50", generalization_score > 0.50),
                ("Consistency < 0.15", np.std(ticker_accuracies) < 0.15),
            ]
            
            all_criteria_met = True
            for criterion, met in success_criteria:
                status = "✅" if met else "❌"
                print(f"    {status} {criterion}")
                if not met:
                    all_criteria_met = False
            
            self.results["performance_metrics"]["avg_accuracy"] = float(avg_accuracy)
            self.results["performance_metrics"]["generalization_score"] = float(generalization_score)
            self.results["performance_metrics"]["accuracy_std"] = float(np.std(ticker_accuracies))
            self.results["validation_results"] = validation_results
            
            if all_criteria_met:
                print("✅ All validation criteria met - ready for Phase 2")
                return True
            else:
                print("❌ Some validation criteria not met - review performance")
                return False
            
        except Exception as e:
            print(f"❌ Training validation failed: {e}")
            self.results["errors"].append(f"Training validation error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_results(self, output_dir: Path = None) -> Path:
        """Save comprehensive test results"""
        if output_dir is None:
            output_dir = Path(f"test_results/comprehensive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / "comprehensive_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary report
        report_file = output_dir / "test_summary.txt"
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE TRAINING TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Mode: {self.test_mode}\n")
            f.write(f"Overall Success: {self.results.get('overall_success', False)}\n\n")
            
            f.write("STAGE RESULTS:\n")
            for stage, success in self.results["stage_results"].items():
                f.write(f"  {stage}: {'PASS' if success else 'FAIL'}\n")
            
            if self.results["performance_metrics"]:
                f.write("\nPERFORMANCE METRICS:\n")
                for metric, value in self.results["performance_metrics"].items():
                    f.write(f"  {metric}: {value}\n")
            
            if self.results["errors"]:
                f.write("\nERRORS:\n")
                for error in self.results["errors"]:
                    f.write(f"  - {error}\n")
        
        print(f"\n📊 Results saved to: {output_dir}")
        return output_dir


def run_fast_test() -> bool:
    """Run fast test (4 tickers, 3 years, 20 epochs)"""
    test = ComprehensiveTrainingTest(test_mode="fast")
    results = test.run_comprehensive_test()
    test.save_results()
    return results.get("overall_success", False)


def run_full_test() -> bool:
    """Run full test (expanded universe, 10 years, 50 epochs)"""
    test = ComprehensiveTrainingTest(test_mode="full")
    results = test.run_comprehensive_test()
    test.save_results()
    return results.get("overall_success", False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Training Test")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast",
                        help="Test mode: fast (4 tickers, 20 epochs) or full (expanded universe, 50 epochs)")
    
    args = parser.parse_args()
    
    if args.mode == "fast":
        success = run_fast_test()
    else:
        success = run_full_test()
    
    print(f"\n{'SUCCESS' if success else 'FAILED'}: Comprehensive training test ({args.mode} mode)")
    sys.exit(0 if success else 1)