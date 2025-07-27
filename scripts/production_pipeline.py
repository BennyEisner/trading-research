#!/usr/bin/env python3

"""
Production ML Pipeline
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Verify Python version
if sys.version_info < (3, 12):
    raise RuntimeError("This application requires Python 3.12 or higher")

sys.path.append(".")

from src.training.directional_trainer import DirectionalEnhancedTrainer
from src.training.hyperparameter_optimizer import run_directional_tuning


class ProductionMLPipeline:
    """
    Production ML Pipeline following industry best practices
    """

    def __init__(self, config_file="production_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.pipeline_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directories
        self.create_output_structure()

    def load_config(self):
        """Load configuration from file or create default"""

        default_config = {
            "pipeline": {
                "stages": {
                    "data_preparation": True,
                    "hyperparameter_tuning": True,
                    "model_training": True,
                    "statistical_validation": True,
                    "model_evaluation": True,
                    "model_artifacts": True,
                },
                "validation": {
                    "block_size": 5,
                    "n_bootstrap": 1000,
                    "significance_threshold": 0.05,
                    "require_significance": False,
                },
                "hyperparameter_tuning": {
                    "method": "random_search",  # random_search, grid_search, both
                    "n_trials": 20,
                    "epochs_per_trial": 12,
                    "enable_grid_search": False,
                },
            },
            "model": {
                "tickers": ["AAPL", "MSFT", "GOOG", "NVDA", "TSLA", "AMZN", "META"],
                "years_of_data": 10,
                "database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db",
                "lookbook_window": 30,
                "target_features": 20,
                "features_per_category": 6,
                "random_seed": 42,
            },
            "training": {"max_epochs": 100, "patience": 15, "validation_split": 0.2, "test_split": 0.2},
        }

        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config = json.load(f)
                print(f"Loaded configuration from {self.config_file}")
        else:
            config = default_config
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
                print(f"Created default configuration: {self.config_file}")

        return config

    def create_output_structure(self):
        """Create output directory structure"""

        self.output_dir = Path(f"production_runs/{self.timestamp}")

        # Create subdirectories
        subdirs = ["data", "hyperparameters", "models", "evaluation", "logs", "artifacts"]

        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        print(f"Created output structure: {self.output_dir}")

    def stage_1_data_preparation(self):
        """Stage 1: Data Preparation and Feature Engineering"""

        print(f"\n{'='*60}")
        print(f"🔧 STAGE 1: DATA PREPARATION & FEATURE ENGINEERING")
        print(f"{'='*60}")

        if not self.config["pipeline"]["stages"]["data_preparation"]:
            print(" Skipping data preparation (disabled in config)")
            return None

        try:
            # Initialize trainer for data preparation
            trainer = DirectionalEnhancedTrainer(self.config["model"])

            # Prepare comprehensive data with enhanced features
            print("Loading and preparing enhanced dataset...")
            data_dict = trainer.prepare_comprehensive_training_data()

            # Extract data splits from dictionary
            X_train, y_train = data_dict["train"]
            X_val, y_val = data_dict["val"]
            X_test, y_test = data_dict["test"]

            # Create feature names
            first_ticker_data = list(data_dict["ticker_data"].values())[0]
            exclude_cols = ["date", "open", "high", "low", "close", "volume", "ticker", "daily_return"]
            feature_names = [col for col in first_ticker_data.columns if col not in exclude_cols]

            # Repack as tuple for downstream compatibility
            data_splits = (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)

            # Log data statistics
            data_stats = {
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
                "feature_count": len(feature_names),
                "sequence_length": X_train.shape[1],
                "feature_names": feature_names[:10],  # First 10 features
                "target_distribution": {
                    "train_up_moves": int(sum(y_train > 0)),
                    "train_down_moves": int(sum(y_train < 0)),
                    "validation_up_moves": int(sum(y_val > 0)),
                    "validation_down_moves": int(sum(y_val < 0)),
                },
            }

            # Save data statistics
            with open(self.output_dir / "data" / "data_stats.json", "w") as f:
                json.dump(data_stats, f, indent=2)

            print(f"   Data preparation complete:")
            print(f"   Training: {len(X_train):,} samples")
            print(f"   Validation: {len(X_val):,} samples")
            print(f"   Test: {len(X_test):,} samples")
            print(f"   Features: {len(feature_names)}")
            print(
                f"   Target distribution: {data_stats['target_distribution']['train_up_moves']} up, {data_stats['target_distribution']['train_down_moves']} down"
            )

            self.pipeline_results["stage_1"] = {
                "status": "success",
                "data_stats": data_stats,
                "data_splits": data_splits,
            }

            return data_splits

        except Exception as e:
            print(f"Stage 1 failed: {e}")
            self.pipeline_results["stage_1"] = {"status": "failed", "error": str(e)}
            raise

    def stage_2_hyperparameter_tuning(self, data_splits):
        """Stage 2: Hyperparameter Tuning"""

        print(f"\n{'='*60}")
        print(f"🔍 STAGE 2: HYPERPARAMETER TUNING")
        print(f"{'='*60}")

        if not self.config["pipeline"]["stages"]["hyperparameter_tuning"]:
            print(" Skipping hyperparameter tuning (disabled in config)")
            # Return default parameters
            return self._get_default_hyperparameters()

        try:
            tuning_config = self.config["pipeline"]["hyperparameter_tuning"]

            print(f"Method: {tuning_config['method']}")
            print(f"Trials: {tuning_config['n_trials']}")
            print(f"Epochs per trial: {tuning_config['epochs_per_trial']}")

            # Run hyperparameter tuning
            best_params, best_score, tuning_results = run_directional_tuning(
                data_splits,
                n_random_trials=tuning_config["n_trials"],
                run_grid_search=tuning_config["enable_grid_search"],
            )

            # Save tuning results
            tuning_output_file = self.output_dir / "hyperparameters" / "tuning_results.json"
            with open(tuning_output_file, "w") as f:
                json.dump(tuning_results, f, indent=2)

            print(f"   Hyperparameter tuning complete:")
            print(f"   Best directional accuracy: {best_score:.4f}")
            print(f"   Best parameters saved to: {tuning_output_file}")

            self.pipeline_results["stage_2"] = {
                "status": "success",
                "best_params": best_params,
                "best_score": best_score,
                "tuning_file": str(tuning_output_file),
            }

            return best_params

        except Exception as e:
            print(f"Stage 2 failed: {e}")
            print("Falling back to default hyperparameters")

            default_params = self._get_default_hyperparameters()
            self.pipeline_results["stage_2"] = {
                "status": "failed_fallback",
                "error": str(e),
                "fallback_params": default_params,
            }

            return default_params

    def stage_2_5_statistical_validation(self, training_results, data_splits):
        """Statistical validation on model performance"""

        print(f"\n{'='*60}")
        print(f"STATISTICAL VALIDATION ")
        print(f"{'='*60}")

        if not self.config["pipeline"]["stages"].get("statistical_validation", True):
            print(" Skipping statistical validation (disabled in config)")
            return None

        try:
            from src.validation.robust_time_series_validator import RobustTimeSeriesValidator

            model = training_results["model"]
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names = data_splits

            # Predictions on test set
            test_predictions = model.predict(X_test, verbose=0).flatten()

            # Initialize validator with parameters
            validator = RobustTimeSeriesValidator(
                block_size=self.config["pipeline"]["validation"]["block_size"],
                n_bootstrap=self.config["pipeline"]["validation"]["n_bootstrap"],
                random_state=42,
            )

            print("Running comprehensive statistical validation...")
            print("- Moving block bootstrap test (preserves temporal dependencies)")
            print("- Conditional permutation test (controls for market regimes)")

            # Run validation tests
            validation_results = validator.validate_model_significance(
                returns=y_test, predictions=test_predictions, test_types=["bootstrap", "permutation"]
            )

            validation_file = self.output_dir / "evaluation" / "statistical_validation.json"
            with open(validation_file, "w") as f:
                serializable_results = self._make_json_serializable(validation_results)
                json.dump(serializable_results, f, indent=2)

            # Print validation summary
            validator.print_validation_summary()

            # Extract key metrics
            overall_assessment = validation_results["overall_assessment"]
            is_statistically_significant = overall_assessment["all_significant"]
            min_p_value = overall_assessment["min_p_value"]

            print(f"\n🎯 STATISTICAL VALIDATION SUMMARY:")
            print(f"   Statistically Significant: {'✅ YES' if is_statistically_significant else '❌ NO'}")
            print(f"   Minimum P-value: {min_p_value:.4f}")
            print(f"   Recommendation: {overall_assessment['recommendation']}")

            # Store results
            self.pipeline_results["stage_2_5"] = {
                "status": "success",
                "validation_file": str(validation_file),
                "statistically_significant": is_statistically_significant,
                "min_p_value": min_p_value,
                "recommendation": overall_assessment["recommendation"],
                "bootstrap_p_value": validation_results.get("moving_block_bootstrap", {}).get("p_value"),
                "permutation_p_value": validation_results.get("conditional_permutation", {}).get("p_value"),
            }

            return validation_results

        except Exception as e:
            print(f"Stage 2.5 failed: {e}")
            self.pipeline_results["stage_2_5"] = {"status": "failed", "error": str(e)}
            return None

    def stage_3_model_training(self, data_splits, best_params):
        """Stage 3: Model Training with Optimal Parameters"""

        print(f"\n{'='*60}")
        print(f"STAGE 3: MODEL TRAINING")
        print(f"{'='*60}")

        if not self.config["pipeline"]["stages"]["model_training"]:
            print("⏭️  Skipping model training (disabled in config)")
            return None

        try:
            # Merge best parameters with base config
            training_config = {**self.config["model"], **best_params}

            # Initialize trainer with optimal parameters
            trainer = DirectionalEnhancedTrainer(training_config)

            print(f"Training with optimal hyperparameters:")
            key_params = ["directional_alpha", "learning_rate", "batch_size", "dropout_rate"]
            for param in key_params:
                if param in best_params:
                    print(f"   {param}: {best_params[param]}")

            # Train the model
            training_results = trainer.train_directional_model(data_splits)

            # Save model
            model_file = self.output_dir / "models" / "best_directional_model.keras"
            training_results["model"].save(str(model_file))

            # Save training configuration
            config_file = self.output_dir / "models" / "training_config.json"
            with open(config_file, "w") as f:
                json.dump(training_config, f, indent=2)

            print(f"Model training complete:")
            print(f"Model saved to: {model_file}")
            # FIXED: Handle missing directional accuracy key
            dir_acc = training_results.get("test_results", {}).get(
                "_directional_accuracy",
                training_results.get("test_results", {}).get("manual_directional_accuracy", 0.5),
            )
            print(f"   Final directional accuracy: {dir_acc:.4f}")

            self.pipeline_results["stage_3"] = {
                "status": "success",
                "model_file": str(model_file),
                "config_file": str(config_file),
                "test_results": training_results["test_results"],
            }

            return training_results

        except Exception as e:
            print(f"Stage 3 failed: {e}")
            self.pipeline_results["stage_3"] = {"status": "failed", "error": str(e)}
            raise

    def stage_4_evaluation(self, training_results, data_splits):
        """Stage 4: Model Evaluation and Analysis"""

        print(f"\n{'='*60}")
        print(f"STAGE 4: MODEL EVALUATION & ANALYSIS")
        print(f"{'='*60}")

        if not self.config["pipeline"]["stages"]["model_evaluation"]:
            print(" Skipping model evaluation (disabled in config)")
            return None

        try:
            model = training_results["model"]
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names = data_splits

            evaluation_results = {
                "model_info": {
                    "total_parameters": model.count_params(),
                    "trainable_parameters": sum([len(w.flatten()) for w in model.trainable_weights]),
                    "feature_count": len(feature_names),
                },
                "performance": {},
            }

            # Evaluate on all datasets
            datasets = [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]

            for name, X, y in datasets:
                results = model.evaluate(X, y, verbose=0)
                metrics = dict(zip(model.metrics_names, results))

                # Additional analysis
                predictions = model.predict(X, verbose=0).flatten()
                directional_accuracy = sum(np.sign(y) == np.sign(predictions)) / len(y)

                evaluation_results["performance"][name] = {
                    **metrics,
                    "manual_directional_accuracy": float(directional_accuracy),
                    "sample_count": len(X),
                }

                print(f"{name.capitalize()} Performance:")
                print(f"   Directional Accuracy: {directional_accuracy:.4f}")
                print(f"   Loss: {metrics['loss']:.6f}")
                print(f"   MAE: {metrics['mae']:.6f}")

            # Save evaluation results
            eval_file = self.output_dir / "evaluation" / "evaluation_results.json"
            with open(eval_file, "w") as f:
                json.dump(evaluation_results, f, indent=2, default=str)

            self.pipeline_results["stage_4"] = {
                "status": "success",
                "evaluation_file": str(eval_file),
                "test_directional_accuracy": evaluation_results["performance"]["test"]["manual_directional_accuracy"],
            }

            return evaluation_results

        except Exception as e:
            print(f"Stage 4 failed: {e}")
            self.pipeline_results["stage_4"] = {"status": "failed", "error": str(e)}
            return None

    def stage_5_artifacts(self):
        """Stage 5: Create Production Artifacts"""

        print(f"\n{'='*60}")
        print(f"STAGE 5: PRODUCTION ARTIFACTS")
        print(f"{'='*60}")

        if not self.config["pipeline"]["stages"]["model_artifacts"]:
            print(" Skipping artifact creation (disabled in config)")
            return None

        try:
            # Create pipeline summary
            pipeline_summary = {
                "timestamp": self.timestamp,
                "config": self.config,
                "pipeline_results": self.pipeline_results,
                "production_ready": all(
                    stage.get("status") in ["success", "failed_fallback"] for stage in self.pipeline_results.values()
                ),
            }

            # Save pipeline summary
            summary_file = self.output_dir / "pipeline_summary.json"
            with open(summary_file, "w") as f:
                json.dump(pipeline_summary, f, indent=2, default=str)

            # Create deployment readme
            readme_content = self._create_deployment_readme()
            readme_file = self.output_dir / "DEPLOYMENT_README.md"
            with open(readme_file, "w") as f:
                f.write(readme_content)

            print(f" Production artifacts created:")
            print(f" Pipeline summary: {summary_file}")
            print(f" Deployment guide: {readme_file}")
            print(f" Output directory: {self.output_dir}")

            self.pipeline_results["stage_5"] = {
                "status": "success",
                "summary_file": str(summary_file),
                "readme_file": str(readme_file),
                "output_directory": str(self.output_dir),
            }

            return pipeline_summary

        except Exception as e:
            print(f"Stage 5 failed: {e}")
            self.pipeline_results["stage_5"] = {"status": "failed", "error": str(e)}
            return None

    def run_full_pipeline(self):
        """Run the complete production pipeline"""

        print(f"STARTING PRODUCTION ML PIPELINE")
        print(f"{'='*60}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Configuration: {self.config_file}")

        try:
            # Stage 1: Data Preparation
            data_splits = self.stage_1_data_preparation()
            if data_splits is None:
                raise Exception("Data preparation failed")

            # Stage 2: Hyperparameter Tuning
            best_params = self.stage_2_hyperparameter_tuning(data_splits)

            # Stage 3: Model Training
            training_results = self.stage_3_model_training(data_splits, best_params)
            if training_results is None and self.config["pipeline"]["stages"]["model_training"]:
                raise Exception("Model training failed")

            # Stage 2.5: Statistical Validation
            if training_results is not None:
                validation_results = self.stage_2_5_statistical_validation(training_results, data_splits)

                # Optional: Fail pipeline if not statistically significant
                if (
                    self.config["pipeline"]["validation"].get("require_significance", False)
                    and validation_results
                    and not validation_results.get("overall_assessment", {}).get("all_significant", False)
                ):
                    raise Exception("Model failed statistical significance tests")

            # Stage 4: Evaluation (skip if no training results)
            if training_results is not None:
                evaluation_results = self.stage_4_evaluation(training_results, data_splits)
            else:
                evaluation_results = None

            # Stage 5: Artifacts
            pipeline_summary = self.stage_5_artifacts()

            print(f"\nPRODUCTION PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")

            if "stage_4" in self.pipeline_results and "test_directional_accuracy" in self.pipeline_results["stage_4"]:
                final_accuracy = self.pipeline_results["stage_4"]["test_directional_accuracy"]
                print(f"Final Test Directional Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.1f}%)")

            print(f"All outputs saved to: {self.output_dir}")

            return self.pipeline_results

        except Exception as e:
            print(f"\nPIPELINE FAILED: {e}")

            # Save partial results
            error_summary = {
                "timestamp": self.timestamp,
                "error": str(e),
                "completed_stages": self.pipeline_results,
                "failed_at": len(self.pipeline_results) + 1,
            }

            error_file = self.output_dir / "pipeline_error.json"
            with open(error_file, "w") as f:
                json.dump(error_summary, f, indent=2, default=str)

            print(f"Error details saved to: {error_file}")
            raise

    def _get_default_hyperparameters(self):
        """Get default hyperparameters for fallback"""
        return {
            "directional_alpha": 0.4,
            "learning_rate": 0.0005,
            "batch_size": 64,
            "dropout_rate": 0.4,
            "l2_regularization": 0.005,
            "lstm_units_1": 512,
            "lstm_units_2": 256,
            "lstm_units_3": 128,
            "dense_layers": [256, 128, 64],
            "use_attention": True,
        }

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _create_deployment_readme(self):
        """Create documentation with validation results"""

        validation_status = "NOT TESTED"
        validation_details = ""

        if "stage_2_5" in self.pipeline_results:
            stage_2_5 = self.pipeline_results["stage_2_5"]
            if stage_2_5.get("status") == "success":
                is_significant = stage_2_5.get("statistically_significant", False)
                validation_status = "STATISTICALLY SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
                validation_details = f"""## Statistical Validation Results

- **Overall Significance**: {'PASS' if is_significant else 'FAIL'}
- **Minimum P-value**: {stage_2_5.get('min_p_value', 'N/A')}
- **Bootstrap P-value**: {stage_2_5.get('bootstrap_p_value', 'N/A')}
- **Permutation P-value**: {stage_2_5.get('permutation_p_value', 'N/A')}
- **Recommendation**: {stage_2_5.get('recommendation', 'N/A')}

### What This Means:
- **Statistically Significant**: Model predictions beat random chance with statistical confidence
- **Bootstrap Test**: Validates performance while preserving temporal dependencies  
- **Permutation Test**: Validates performance while controlling for market regime effects
- **P-value < 0.05**: Required for statistical significance at 95% confidence
"""
        return f"""# Production Model Deployment Guide

## Pipeline Run Information
- **Timestamp**: {self.timestamp}
- **Configuration**: {self.config_file}
- **Output Directory**: {self.output_dir}

## Model Validation Status
**Statistical Validation**: {validation_status}

{validation_details}

## Next Steps for Deployment
{'**APPROVED**: This model has passed statistical validation and can proceed to production.' if validation_status.startswith('STATISTICALLY SIGNIFICANT') else '**NOT APPROVED**: This model has NOT passed statistical validation. Do not deploy to production.'}

Generated by Production ML Pipeline - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""


def main():
    """Run the production pipeline"""
    import numpy as np

    pipeline = ProductionMLPipeline("production_config.json")
    results = pipeline.run_full_pipeline()

    return results


if __name__ == "__main__":
    results = main()
