#!/usr/bin/env python3

"""
Model Predictions Analysis Script
Extract and analyze actual model predictions to confirm constant prediction hypothesis
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Skip TensorFlow warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from src.features.pattern_feature_calculator import FeatureCalculator
from src.training.shared_backbone_trainer import SharedBackboneTrainer
from tests.utilities.data_loader import load_test_data


class ModelPredictionAnalyzer:
    """Analyze model predictions to diagnose constant prediction issue"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("prediction_analysis_results")
        self.output_dir.mkdir(exist_ok=True)

        self.results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "prediction_statistics": {},
            "correlation_analysis": {},
            "diagnostic_findings": [],
            "model_behavior": {},
        }

    def analyze_fresh_model_behavior(self, tickers: List[str] = None, train_epochs: int = 10) -> Dict[str, Any]:
        """Train a fresh model and analyze its prediction behavior"""

        if tickers is None:
            tickers = ["AAPL", "MSFT", "GOOG", "NVDA"]  # Fast subset for analysis

        print("=== MODEL PREDICTION BEHAVIOR ANALYSIS ===")
        print(f"Training fresh model on {len(tickers)} tickers for {train_epochs} epochs")
        print()

        # Load and prepare training data
        print("Loading and preparing training data...")
        raw_data = load_test_data(tickers, days=500)  # Moderate dataset for analysis

        if not raw_data:
            print("❌ Failed to load training data")
            return self.results

        # Prepare feature data
        feature_calculator = FeatureCalculator()
        training_data = {}

        for ticker, data in raw_data.items():
            try:
                features_df = feature_calculator.calculate_all_features(data)
                if features_df is not None and len(features_df) > 100:
                    training_data[ticker] = features_df
                    print(f"  ✅ {ticker}: {len(features_df)} samples prepared")
                else:
                    print(f"  ❌ {ticker}: Insufficient feature data")
            except Exception as e:
                print(f"  ❌ {ticker}: Feature error - {e}")

        if not training_data:
            print("❌ No training data prepared")
            return self.results

        # Initialize trainer
        print(f"\\nInitializing trainer for {len(training_data)} tickers...")
        trainer = SharedBackboneTrainer(
            tickers=list(training_data.keys()), use_expanded_universe=False  # Simple training for analysis
        )

        # Prepare training sequences
        print("Preparing training sequences...")
        prepared_data = trainer.prepare_training_data(training_data)

        if not prepared_data:
            print("❌ Failed to prepare training sequences")
            return self.results

        # Analyze initial model predictions (random initialization)
        print("\\nAnalyzing random initialization predictions...")
        self._analyze_model_predictions(trainer, prepared_data, stage="random_init")

        # Train for few epochs and analyze progression
        print(f"\\nTraining model for {train_epochs} epochs...")
        training_results = trainer.train_shared_backbone(
            training_data=prepared_data, validation_split=0.2, epochs=train_epochs
        )

        if "model" not in training_results:
            print("❌ Training failed")
            return self.results

        model = training_results["model"]
        history = training_results.get("history", {})

        # Analyze trained model predictions
        print("\\nAnalyzing trained model predictions...")
        self._analyze_model_predictions_direct(model, prepared_data, stage="trained")

        # Analyze training progression
        self._analyze_training_progression(history)

        # Generate diagnostic findings
        self._generate_prediction_diagnostics()

        # Save results
        self._save_prediction_results()

        return self.results

    def _analyze_model_predictions(self, trainer: SharedBackboneTrainer, prepared_data: Dict, stage: str):
        """Analyze model predictions at different training stages"""

        # Build a fresh model for analysis
        sample_ticker = list(prepared_data.keys())[0]
        sample_X, sample_y = prepared_data[sample_ticker]
        input_shape = (sample_X.shape[1], sample_X.shape[2])

        # Create model with same architecture as trainer
        from config.config import get_config
        from src.models.shared_backbone_lstm import SharedBackboneLSTMBuilder

        config = get_config()
        builder = SharedBackboneLSTMBuilder(config.dict())
        model = builder.build_model(input_shape)

        self._analyze_model_predictions_direct(model, prepared_data, stage)

    def _analyze_model_predictions_direct(self, model: tf.keras.Model, prepared_data: Dict, stage: str):
        """Analyze predictions from a specific model"""

        print(f"  Analyzing {stage} model predictions...")

        all_predictions = []
        all_targets = []
        ticker_stats = {}

        for ticker, (X, y) in prepared_data.items():
            # Get model predictions
            predictions = model.predict(X, verbose=0).flatten()
            targets = y.flatten()

            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # Calculate ticker-specific statistics
            pred_stats = {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
                "median": float(np.median(predictions)),
                "unique_values": len(np.unique(np.round(predictions, 4))),
                "variance": float(np.var(predictions)),
            }

            target_stats = {
                "mean": float(np.mean(targets)),
                "std": float(np.std(targets)),
                "correlation": float(np.corrcoef(predictions, targets)[0, 1]) if np.std(predictions) > 1e-8 else 0.0,
            }

            ticker_stats[ticker] = {"predictions": pred_stats, "targets": target_stats, "samples": len(predictions)}

            print(
                f"    {ticker}: pred_mean={pred_stats['mean']:.4f}, pred_std={pred_stats['std']:.6f}, corr={target_stats['correlation']:.6f}"
            )

        # Overall statistics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        overall_stats = {
            "total_samples": len(all_predictions),
            "prediction_mean": float(np.mean(all_predictions)),
            "prediction_std": float(np.std(all_predictions)),
            "prediction_range": [float(np.min(all_predictions)), float(np.max(all_predictions))],
            "target_mean": float(np.mean(all_targets)),
            "target_std": float(np.std(all_targets)),
            "overall_correlation": (
                float(np.corrcoef(all_predictions, all_targets)[0, 1]) if np.std(all_predictions) > 1e-8 else 0.0
            ),
            "prediction_uniqueness": len(np.unique(np.round(all_predictions, 4))),
            "constant_prediction_test": float(np.std(all_predictions)) < 1e-6,
        }

        self.results["prediction_statistics"][stage] = {"overall": overall_stats, "by_ticker": ticker_stats}

        print(
            f"  Overall: pred_std={overall_stats['prediction_std']:.6f}, corr={overall_stats['overall_correlation']:.6f}"
        )

        # Check for constant predictions
        if overall_stats["constant_prediction_test"]:
            print(f"  🚨 CONSTANT PREDICTIONS DETECTED: std = {overall_stats['prediction_std']:.8f}")
        elif overall_stats["prediction_std"] < 0.001:
            print(f"  ⚠️  NEAR-CONSTANT PREDICTIONS: std = {overall_stats['prediction_std']:.6f}")

    def _analyze_training_progression(self, history: Dict):
        """Analyze how predictions change during training"""

        if not history:
            print("  ❌ No training history available")
            return

        progression_analysis = {}

        # Extract key metrics
        metrics_to_analyze = [
            "loss",
            "val_loss",
            "_correlation_metric",
            "val__correlation_metric",
            "_pattern_detection_accuracy",
            "val__pattern_detection_accuracy",
        ]

        for metric in metrics_to_analyze:
            if metric in history:
                values = history[metric]
                progression_analysis[metric] = {
                    "initial": float(values[0]) if len(values) > 0 else 0.0,
                    "final": float(values[-1]) if len(values) > 0 else 0.0,
                    "min_value": float(np.min(values)),
                    "max_value": float(np.max(values)),
                    "trend": "decreasing" if values[-1] < values[0] else "increasing",
                    "improvement": float((values[0] - values[-1]) / values[0]) if values[0] != 0 else 0.0,
                }

        self.results["model_behavior"]["training_progression"] = progression_analysis

        # Print key findings
        print(f"  Training Progression Analysis:")

        if "_correlation_metric" in progression_analysis:
            corr_data = progression_analysis["_correlation_metric"]
            print(f"    Correlation: {corr_data['initial']:.6f} → {corr_data['final']:.6f}")

        if "_pattern_detection_accuracy" in progression_analysis:
            acc_data = progression_analysis["_pattern_detection_accuracy"]
            print(f"    Accuracy: {acc_data['initial']:.4f} → {acc_data['final']:.4f}")

        if "loss" in progression_analysis:
            loss_data = progression_analysis["loss"]
            print(f"    Loss: {loss_data['initial']:.4f} → {loss_data['final']:.4f}")

    def _generate_prediction_diagnostics(self):
        """Generate diagnostic findings from prediction analysis"""

        findings = []

        # Check both random and trained stages
        for stage in ["random_init", "trained"]:
            if stage not in self.results["prediction_statistics"]:
                continue

            overall = self.results["prediction_statistics"][stage]["overall"]

            # Constant prediction check
            if overall.get("constant_prediction_test", False):
                findings.append(f"CONSTANT_PREDICTIONS_{stage.upper()}")
            elif overall.get("prediction_std", 1.0) < 0.001:
                findings.append(f"NEAR_CONSTANT_PREDICTIONS_{stage.upper()}")

            # Correlation check
            if abs(overall.get("overall_correlation", 0)) < 0.01:
                findings.append(f"ZERO_CORRELATION_{stage.upper()}")

            # Range check
            pred_range = overall.get("prediction_range", [0, 1])
            if pred_range[1] - pred_range[0] < 0.1:
                findings.append(f"LIMITED_PREDICTION_RANGE_{stage.upper()}")

        # Training progression check
        if "training_progression" in self.results["model_behavior"]:
            progression = self.results["model_behavior"]["training_progression"]

            # Check if correlation improved
            if "_correlation_metric" in progression:
                corr_improvement = abs(progression["_correlation_metric"]["final"]) - abs(
                    progression["_correlation_metric"]["initial"]
                )
                if corr_improvement < 0.01:
                    findings.append("NO_CORRELATION_IMPROVEMENT")

            # Check if model is learning patterns vs. bias
            if "loss" in progression and "_pattern_detection_accuracy" in progression:
                loss_improvement = progression["loss"]["improvement"]
                acc_improvement = progression["_pattern_detection_accuracy"]["improvement"]

                if loss_improvement > 0.5 and abs(acc_improvement) < 0.05:
                    findings.append("LEARNING_BIAS_NOT_PATTERNS")

        self.results["diagnostic_findings"] = findings

        print(f"  Diagnostic Findings:")
        for finding in findings:
            print(f"    🚨 {finding}")

    def _save_prediction_results(self):
        """Save prediction analysis results"""

        # Save complete JSON results
        results_file = self.output_dir / "prediction_analysis.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save summary report
        summary_file = self.output_dir / "PREDICTION_ANALYSIS.md"
        with open(summary_file, "w") as f:
            f.write("# Model Prediction Analysis Summary\\n\\n")
            f.write(f"Analysis Date: {self.results['analysis_timestamp']}\\n\\n")

            # Prediction statistics
            for stage in ["random_init", "trained"]:
                if stage in self.results["prediction_statistics"]:
                    stats = self.results["prediction_statistics"][stage]["overall"]
                    f.write(f"## {stage.replace('_', ' ').title()} Predictions\\n\\n")
                    f.write(f"- Mean: {stats.get('prediction_mean', 0):.4f}\\n")
                    f.write(f"- Std: {stats.get('prediction_std', 0):.6f}\\n")
                    f.write(f"- Range: {stats.get('prediction_range', [0,0])}\\n")
                    f.write(f"- Correlation: {stats.get('overall_correlation', 0):.6f}\\n")
                    f.write(f"- Constant predictions: {stats.get('constant_prediction_test', False)}\\n\\n")

            # Findings
            f.write("## Diagnostic Findings\\n\\n")
            for finding in self.results.get("diagnostic_findings", []):
                f.write(f"- 🚨 **{finding}**\\n")

        print(f"\\n📊 Prediction analysis results saved:")
        print(f"  - Complete data: {results_file}")
        print(f"  - Summary report: {summary_file}")


def main():
    """Main execution function"""

    # Create analyzer
    analyzer = ModelPredictionAnalyzer()

    # Run prediction analysis
    results = analyzer.analyze_fresh_model_behavior(train_epochs=15)

    # Print final summary
    print(f"\\n{'='*80}")
    print("MODEL PREDICTION ANALYSIS COMPLETE")
    print(f"{'='*80}")

    findings = results.get("diagnostic_findings", [])
    print(f"Identified {len(findings)} diagnostic findings:")

    for finding in findings:
        print(f"  🚨 {finding}")

    # Check for constant prediction confirmation
    constant_found = any("CONSTANT" in finding for finding in findings)
    if constant_found:
        print("\\n✅ CONFIRMED: Model producing constant/near-constant predictions")
        print("   This explains zero correlation despite high accuracy!")

    return constant_found


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

