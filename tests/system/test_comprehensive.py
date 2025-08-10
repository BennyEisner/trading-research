#!/usr/bin/env python3
"""
Comprehensive System Test
all major workflow testing into a singletest suite
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.utilities.data_fixtures import TestConfigurationFixtures, TestDataGenerator
from tests.utilities.test_helpers import TestAssertions, TestEnvironment, TestMetrics, TestTimer


@dataclass
class ComprehensiveTestConfig:
    """Configuration for comprehensive testing"""

    mode: str = "fast"  # "fast" or "full"
    tickers: List[str] = None
    days: int = 100
    epochs: int = 3
    correlation_threshold: float = 0.1  # Minimum acceptable correlation

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["AAPL", "MSFT", "NVDA"] if self.mode == "fast" else ["AAPL", "MSFT", "NVDA", "GOOG", "TSLA"]


class ComprehensiveSystemTest:
    """
    Single comprehensive test class that validates the entire system
    No duplicate functionality - each test method has a specific, unique purpose
    """

    def __init__(self, config: ComprehensiveTestConfig = None):
        self.config = config or ComprehensiveTestConfig()
        self.results = {}
        self.timer = TestTimer()

    def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run the complete system test suite"""

        print(f"🔬 COMPREHENSIVE SYSTEM TEST SUITE")
        print(f"{'='*60}")
        print(f"Mode: {self.config.mode}")
        print(f"Tickers: {self.config.tickers}")
        print(f"Timeline: {self.config.days} days, {self.config.epochs} epochs")
        print(f"{'='*60}")

        with self.timer.measure():
            # Stage 1: Infrastructure validation
            print(f"\n📋 STAGE 1: Infrastructure Validation")
            self.results["infrastructure"] = self._test_infrastructure()

            # Stage 2: Data pipeline validation
            print(f"\n📊 STAGE 2: Data Pipeline Validation")
            self.results["data_pipeline"] = self._test_data_pipeline()

            # Stage 3: Model training validation
            print(f"\n🧠 STAGE 3: Model Training Validation")
            self.results["model_training"] = self._test_model_training()

            # Stage 4: System integration validation
            print(f"\n🔗 STAGE 4: System Integration Validation")
            self.results["system_integration"] = self._test_system_integration()

            # Stage 5: Performance validation
            print(f"\n⚡ STAGE 5: Performance Validation")
            self.results["performance"] = self._test_performance()

        self._generate_final_report()
        return self.results

    def _test_infrastructure(self) -> Dict[str, Any]:
        """Test system infrastructure and dependencies"""

        results = {"passed": True, "issues": []}

        try:
            # Test imports
            print("  ✓ Testing imports...")
            from config.config import get_config
            from src.models.shared_backbone_lstm import SharedBackboneLSTMBuilder
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            # Test configuration
            print("  ✓ Testing configuration...")
            config = get_config()
            assert hasattr(config.model, "lookback_window")
            assert hasattr(config.model, "model_params")

            # Test TensorFlow availability
            print("  ✓ Testing TensorFlow...")
            import tensorflow as tf

            assert len(tf.config.list_physical_devices()) > 0

            print("  ✅ Infrastructure validation PASSED")

        except Exception as e:
            results["passed"] = False
            results["issues"].append(f"Infrastructure error: {e}")
            print(f"  ❌ Infrastructure validation FAILED: {e}")

        return results

    def _test_data_pipeline(self) -> Dict[str, Any]:
        """Test complete data pipeline from loading to preprocessing"""

        results = {"passed": True, "issues": [], "metrics": {}}

        try:
            print("  ✓ Loading test data...")
            # Use test data generator for consistent, fast testing
            ticker_data = TestDataGenerator.generate_multi_ticker_data(
                tickers=self.config.tickers, days=self.config.days
            )

            # Validate data quality
            for ticker, df in ticker_data.items():
                TestAssertions.assert_dataframe_properties(
                    df,
                    expected_columns=["Open", "High", "Low", "Close", "Volume"],
                    min_rows=self.config.days - 10,  # Allow for slight variation
                )

            print("  ✓ Testing feature generation...")
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            trainer = SharedBackboneTrainer(tickers=self.config.tickers, use_expanded_universe=False)

            training_data = trainer.prepare_training_data(ticker_data)

            # Validate training data
            assert len(training_data) > 0, "No training data generated"

            total_sequences = sum(len(X) for X, y in training_data.values())
            results["metrics"]["total_sequences"] = total_sequences
            results["metrics"]["tickers_processed"] = len(training_data)

            print(f"  ✓ Generated {total_sequences} sequences from {len(training_data)} tickers")
            print("  ✅ Data pipeline validation PASSED")

            # Store data for next stages
            self._ticker_data = ticker_data
            self._training_data = training_data

        except Exception as e:
            results["passed"] = False
            results["issues"].append(f"Data pipeline error: {e}")
            print(f"  ❌ Data pipeline validation FAILED: {e}")

        return results

    def _test_model_training(self) -> Dict[str, Any]:
        """Test model training with correlation optimization"""

        results = {"passed": True, "issues": [], "metrics": {}}

        try:
            if not hasattr(self, "_training_data"):
                raise Exception("Training data not available - data pipeline must pass first")

            print("  ✓ Initializing model training...")
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            trainer = SharedBackboneTrainer(tickers=self.config.tickers, use_expanded_universe=False)

            print(f"  ✓ Training model for {self.config.epochs} epochs...")
            with TestEnvironment.suppress_tensorflow_warnings():
                training_results = trainer.train_shared_backbone(
                    training_data=self._training_data, validation_split=0.2, epochs=self.config.epochs
                )

            # Validate training results
            assert "model" in training_results, "No model in training results"
            assert "final_metrics" in training_results, "No final metrics in training results"

            final_metrics = training_results["final_metrics"]

            # Extract key metrics
            correlation = final_metrics.get("correlation", 0)
            best_val_correlation = final_metrics.get("best_validation_correlation", 0)
            pattern_accuracy = final_metrics.get("pattern_detection_accuracy", 0)

            results["metrics"].update(
                {
                    "correlation": correlation,
                    "best_validation_correlation": best_val_correlation,
                    "pattern_accuracy": pattern_accuracy,
                    "train_loss": final_metrics.get("train_loss", 0),
                    "val_loss": final_metrics.get("val_loss", 0),
                }
            )

            # Validate correlation achievement
            if abs(best_val_correlation) >= self.config.correlation_threshold:
                print(f"  ✅ Correlation optimization SUCCEEDED: {best_val_correlation:.4f}")
            else:
                print(
                    f"  ⚠️  Correlation below threshold: {best_val_correlation:.4f} < {self.config.correlation_threshold}"
                )
                results["issues"].append(f"Low correlation: {best_val_correlation:.4f}")

            # Validate pattern detection
            if pattern_accuracy > 0.5:
                print(f"  ✅ Pattern detection WORKING: {pattern_accuracy:.3f} accuracy")
            else:
                print(f"  ⚠️  Pattern detection poor: {pattern_accuracy:.3f} accuracy")
                results["issues"].append(f"Poor pattern accuracy: {pattern_accuracy:.3f}")

            print("  ✅ Model training validation PASSED")

            # Store model for next stages
            self._trained_model = training_results["model"]
            self._training_results = training_results

        except Exception as e:
            results["passed"] = False
            results["issues"].append(f"Model training error: {e}")
            print(f"  ❌ Model training validation FAILED: {e}")

        return results

    def _test_system_integration(self) -> Dict[str, Any]:
        """Test complete system integration and cross-ticker performance"""

        results = {"passed": True, "issues": [], "metrics": {}}

        try:
            if not hasattr(self, "_trained_model"):
                raise Exception("Trained model not available - training must pass first")

            print("  ✓ Testing cross-ticker validation...")
            from src.training.shared_backbone_trainer import SharedBackboneTrainer

            trainer = SharedBackboneTrainer(tickers=self.config.tickers, use_expanded_universe=False)

            cross_ticker_results = trainer.validate_cross_ticker_performance(
                training_data=self._training_data, trained_model=self._trained_model
            )

            # Extract overall statistics
            overall_stats = cross_ticker_results["overall_stats"]

            results["metrics"].update(
                {
                    "mean_correlation": overall_stats.get("mean_correlation", 0),
                    "mean_pattern_accuracy": overall_stats.get("mean_pattern_detection_accuracy", 0),
                    "successful_tickers": overall_stats.get("successful_tickers", 0),
                    "generalization_score": overall_stats.get("pattern_generalization_score", 0),
                }
            )

            # Validate generalization
            if results["metrics"]["mean_correlation"] >= self.config.correlation_threshold:
                print(f"  ✅ Cross-ticker generalization GOOD: {results['metrics']['mean_correlation']:.4f}")
            else:
                print(f"  ⚠️  Cross-ticker generalization weak: {results['metrics']['mean_correlation']:.4f}")
                results["issues"].append("Weak cross-ticker generalization")

            print("  ✅ System integration validation PASSED")

        except Exception as e:
            results["passed"] = False
            results["issues"].append(f"System integration error: {e}")
            print(f"  ❌ System integration validation FAILED: {e}")

        return results

    def _test_performance(self) -> Dict[str, Any]:
        """Test system performance and benchmarks"""

        results = {"passed": True, "issues": [], "metrics": {}}

        try:
            print("  ✓ Testing training performance...")

            # Measure prediction performance
            if hasattr(self, "_trained_model") and hasattr(self, "_training_data"):

                # Get sample data for prediction timing
                sample_ticker = list(self._training_data.keys())[0]
                sample_X, sample_y = self._training_data[sample_ticker]

                # Time predictions
                prediction_timer = TestTimer()
                with prediction_timer.measure():
                    predictions = self._trained_model.predict(sample_X[:100], verbose=0)

                results["metrics"].update(
                    {
                        "training_time_seconds": self.timer.elapsed(),
                        "prediction_time_ms": prediction_timer.elapsed() * 1000,
                        "sequences_per_second": 100 / max(prediction_timer.elapsed(), 0.001),
                    }
                )

                print(f"  ✓ Training completed in {self.timer.elapsed():.1f}s")
                print(f"  ✓ Prediction speed: {results['metrics']['sequences_per_second']:.0f} sequences/sec")

                # Performance benchmarks
                if self.timer.elapsed() < 300:  # 5 minutes for fast mode
                    print("  ✅ Training performance ACCEPTABLE")
                else:
                    print(f"  ⚠️  Training slow: {self.timer.elapsed():.1f}s")
                    results["issues"].append("Slow training performance")

            print("  ✅ Performance validation PASSED")

        except Exception as e:
            results["passed"] = False
            results["issues"].append(f"Performance error: {e}")
            print(f"  ❌ Performance validation FAILED: {e}")

        return results

    def _generate_final_report(self):
        """Generate comprehensive final report"""

        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE TEST RESULTS")
        print(f"{'='*60}")

        overall_success = all(stage_result.get("passed", False) for stage_result in self.results.values())

        print(f"Overall Success: {'✅ PASS' if overall_success else '❌ FAIL'}")
        print(f"Total Runtime: {self.timer.elapsed():.1f}s")

        # Stage results
        print(f"\nStage Results:")
        for stage, result in self.results.items():
            status = "PASS" if result.get("passed", False) else "FAIL"
            issues = len(result.get("issues", []))
            print(f"  {stage:20}: {status:4} ({issues} issues)")

        # Key metrics summary
        if "model_training" in self.results:
            metrics = self.results["model_training"]["metrics"]
            print(f"\nKey Performance Metrics:")
            print(f"  Correlation: {metrics.get('best_validation_correlation', 0):8.4f}")
            print(f"  Pattern Acc: {metrics.get('pattern_accuracy', 0):8.4f}")
            print(f"  Training Loss: {metrics.get('train_loss', 0):6.4f}")
            print(f"  Val Loss: {metrics.get('val_loss', 0):10.4f}")

        if "system_integration" in self.results:
            metrics = self.results["system_integration"]["metrics"]
            print(f"\nIntegration Metrics:")
            print(f"  Cross-ticker Correlation: {metrics.get('mean_correlation', 0):8.4f}")
            print(f"  Generalization Score: {metrics.get('generalization_score', 0):12.4f}")
            print(f"  Successful Tickers: {metrics.get('successful_tickers', 0):8}")

        # Issues summary
        all_issues = []
        for stage_result in self.results.values():
            all_issues.extend(stage_result.get("issues", []))

        if all_issues:
            print(f"\n⚠️  Issues Found ({len(all_issues)}):")
            for issue in all_issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ No issues found!")

        # Success criteria
        print(f"\n📊 Success Criteria:")
        criteria = [
            ("Infrastructure", self.results.get("infrastructure", {}).get("passed", False)),
            ("Data Pipeline", self.results.get("data_pipeline", {}).get("passed", False)),
            ("Model Training", self.results.get("model_training", {}).get("passed", False)),
            ("System Integration", self.results.get("system_integration", {}).get("passed", False)),
            ("Performance", self.results.get("performance", {}).get("passed", False)),
        ]

        for criterion, passed in criteria:
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion}")

        self.results["overall_success"] = overall_success
        self.results["total_runtime"] = self.timer.elapsed()
        self.results["issues_count"] = len(all_issues)


def run_fast_comprehensive_test():
    """Run fast comprehensive test (for CI/development)"""
    config = ComprehensiveTestConfig(
        mode="fast", tickers=["AAPL", "MSFT"], days=50, epochs=2, correlation_threshold=0.05  # Relaxed for fast test
    )

    test = ComprehensiveSystemTest(config)
    return test.run_complete_test_suite()


def run_full_comprehensive_test():
    """Run full comprehensive test (for release validation)"""
    config = ComprehensiveTestConfig(
        mode="full", tickers=["AAPL", "MSFT", "NVDA", "GOOG", "TSLA"], days=200, epochs=10, correlation_threshold=0.1
    )

    test = ComprehensiveSystemTest(config)
    return test.run_complete_test_suite()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive system tests")
    parser.add_argument(
        "--mode", choices=["fast", "full"], default="fast", help="Test mode: fast (CI/dev) or full (release)"
    )

    args = parser.parse_args()

    if args.mode == "fast":
        results = run_fast_comprehensive_test()
    else:
        results = run_full_comprehensive_test()

    # Exit with appropriate code
    exit_code = 0 if results.get("overall_success", False) else 1
    sys.exit(exit_code)

