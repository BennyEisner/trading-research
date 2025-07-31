#!/usr/bin/env python3

"""
Hyperparameter tuning system focused on directional accuracy
"""

import json
import sys
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(".")

from src.models.multi_scale_lstm import MultiScaleLSTMBuilder
from config.config import load_config


def ensure_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy types
        return obj.item()
    else:
        return obj


class DirectionalHyperparameterTuner:
    """
    Hyperparameter tuning focused on maximizing directional accuracy
    """

    def __init__(self, X_train, y_train, X_val, y_val, config_path=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.results = []
        
        # Load centralized configuration
        self.config = load_config(config_path) if config_path else load_config()
        self.base_params = self.config.get_model_params()
        print(f"🔧 Base params loaded: monitor_metric={self.base_params.get('monitor_metric', 'NOT_SET')}")
        print(f"🔧 Patience: {self.base_params.get('patience', 'NOT_SET')}")

    def define_search_space(self):
        """Define hyperparameter search space for directional optimization based on central config"""
        
        # Get base values from central config
        base_alpha = self.base_params['directional_alpha']
        base_lr = self.base_params['learning_rate']
        base_batch = self.base_params['batch_size']
        base_dropout = self.base_params['dropout_rate']
        base_l2 = self.base_params['l2_regularization']
        
        # Create search space with explicit type conversion for JSON serialization
        search_space = {
            # Directional loss parameters - narrow range around config value
            "directional_alpha": [float(base_alpha * 0.5), float(base_alpha), float(base_alpha * 2), float(base_alpha * 3), float(base_alpha * 4)],
            # Learning parameters
            "learning_rate": [float(base_lr * 0.6), float(base_lr), float(base_lr * 1.5), float(base_lr * 2)],
            "batch_size": [int(base_batch // 2), int(base_batch), int(base_batch * 1.5), int(base_batch * 2)],
            # Regularization
            "dropout_rate": [float(base_dropout - 0.1), float(base_dropout), float(base_dropout + 0.1), float(base_dropout + 0.2)],
            "l2_regularization": [float(base_l2 * 0.3), float(base_l2), float(base_l2 * 2), float(base_l2 * 3)],
            # Architecture parameters - from config
            "lstm_units_1": [int(self.base_params['lstm_units_1'] // 2), int(self.base_params['lstm_units_1']), int(self.base_params['lstm_units_1'] * 1.5)],
            "lstm_units_2": [int(self.base_params['lstm_units_2'] // 2), int(self.base_params['lstm_units_2']), int(self.base_params['lstm_units_2'] * 1.5)],
            "lstm_units_3": [int(self.base_params['lstm_units_3'] // 2), int(self.base_params['lstm_units_3']), int(self.base_params['lstm_units_3'] * 1.5)],
            "dense_layers": [self.base_params['dense_layers'], [128, 64], [384, 192, 64]],
        }
        
        # Ensure valid ranges with proper type conversion
        search_space["batch_size"] = [max(16, x) for x in search_space["batch_size"]]
        search_space["lstm_units_1"] = [max(64, x) for x in search_space["lstm_units_1"]]
        search_space["lstm_units_2"] = [max(32, x) for x in search_space["lstm_units_2"]]
        search_space["lstm_units_3"] = [max(16, x) for x in search_space["lstm_units_3"]]
        search_space["dropout_rate"] = [max(0.1, min(0.8, x)) for x in search_space["dropout_rate"]]

        return search_space

    def random_search(self, n_trials=20, epochs=15):
        """
        Random search for optimal hyperparameters
        """
        print(f" DIRECTIONAL HYPERPARAMETER TUNING - RANDOM SEARCH")
        print(f"Trials: {n_trials}, Epochs per trial: {epochs}")
        print("=" * 60)

        search_space = self.define_search_space()

        best_score = 0
        best_params = None

        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")
            print("-" * 30)

            # Sample random hyperparameters
            params = {}
            for key, values in search_space.items():
                if key == 'dense_layers':
                    # Handle list-of-lists specially to avoid numpy array conversion
                    params[key] = values[np.random.randint(len(values))]
                else:
                    params[key] = np.random.choice(values)

            # Ensure parameters are JSON serializable
            params = ensure_json_serializable(params)
            print(f"Testing: {json.dumps(params, indent=2)}")

            try:
                # Train model with these parameters
                score, detailed_metrics = self._evaluate_params(params, epochs)

                # Store results
                result = {
                    "trial": trial + 1,
                    "params": params,
                    "directional_accuracy": score,
                    "metrics": detailed_metrics,
                }
                self.results.append(result)

                print(f" Directional Accuracy: {score:.4f}")

                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f" NEW BEST SCORE: {best_score:.4f}")

            except Exception as e:
                print(f" Trial failed: {e}")
                continue

        return best_params, best_score

    def grid_search_focused(self, epochs=10):
        """
        Focused grid search on most important parameters
        """
        print(f" FOCUSED GRID SEARCH ON KEY DIRECTIONAL PARAMETERS")
        print(f"Epochs per trial: {epochs}")
        print("=" * 60)

        # Focus on most impactful parameters around central config values
        base_alpha = self.base_params['directional_alpha']
        base_lr = self.base_params['learning_rate']
        base_batch = self.base_params['batch_size']
        base_dropout = self.base_params['dropout_rate']
        
        focused_space = {
            "directional_alpha": [float(base_alpha), float(base_alpha * 2), float(base_alpha * 3)],
            "learning_rate": [float(base_lr), float(base_lr * 1.5)],
            "batch_size": [int(base_batch), int(base_batch * 1.5)],
            "dropout_rate": [float(base_dropout), float(base_dropout + 0.1)],
        }

        # Fixed architecture from central config with proper type conversion
        fixed_params = {
            "lstm_units_1": int(self.base_params['lstm_units_1']),
            "lstm_units_2": int(self.base_params['lstm_units_2']),
            "lstm_units_3": int(self.base_params['lstm_units_3']),
            "dense_layers": self.base_params['dense_layers'],
            "l2_regularization": float(self.base_params['l2_regularization']),
            "use_attention": bool(self.base_params['use_attention']),
        }

        # Generate all combinations
        param_names = list(focused_space.keys())
        param_values = list(focused_space.values())
        combinations = list(product(*param_values))

        print(f"Testing {len(combinations)} parameter combinations")

        best_score = 0
        best_params = None

        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            params.update(fixed_params)  # Add fixed parameters
            
            # Ensure parameters are JSON serializable
            params = ensure_json_serializable(params)

            print(f"\nGrid Search {i + 1}/{len(combinations)}")
            print(f"Parameters: {json.dumps({k: v for k, v in params.items() if k in focused_space}, indent=2)}")

            try:
                score, detailed_metrics = self._evaluate_params(params, epochs)

                result = {
                    "grid_search": i + 1,
                    "params": params,
                    "directional_accuracy": score,
                    "metrics": detailed_metrics,
                }
                self.results.append(result)

                print(f" Directional Accuracy: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f" NEW BEST: {best_score:.4f}")

            except Exception as e:
                print(f" Failed: {e}")
                continue

        return best_params, best_score

    def _evaluate_params(self, params, epochs):
        """Evaluate hyperparams using temporal CV with detailed logging"""
        print("🔄 Starting parameter evaluation...")

        from src.validation.gapped_time_series_cv import GappedTimeSeriesCV

        print("✅ Creating GappedTimeSeriesCV...")
        gapped_cv = GappedTimeSeriesCV(n_splits=3, test_size=0.2, gap_size=5, expanding_window=True)

        print("✅ Combining train and validation data...")
        X_combined = np.vstack([self.X_train, self.X_val])
        y_combined = np.hstack([self.y_train, self.y_val])
        print(f"   Combined data shape: X={X_combined.shape}, y={y_combined.shape}")

        cv_scores = []
        print("🔄 Starting cross-validation folds...")

        fold_count = 0
        for train_idx, val_idx in gapped_cv.split(X_combined):
            fold_count += 1
            print(f"   📂 Fold {fold_count}: train_idx={len(train_idx)}, val_idx={len(val_idx)}")
            
            X_cv_train, X_cv_val = X_combined[train_idx], X_combined[val_idx]
            y_cv_train, y_cv_val = y_combined[train_idx], y_combined[val_idx]
            print(f"      Data shapes: X_train={X_cv_train.shape}, X_val={X_cv_val.shape}")

            print("   🏗️  Building model...")
            try:
                builder = MultiScaleLSTMBuilder({})
                model = builder.build_directional_focused_model(X_cv_train.shape[1:], **params)
                print(f"      Model built successfully: {model.count_params()} parameters")
            except Exception as e:
                print(f"      ❌ Model building failed: {e}")
                raise

            print("   🎯 Starting model training...")
            try:
                # Monitor metric name from base params
                monitor_metric = self.base_params.get("monitor_metric", "val__directional_accuracy")
                patience = self.base_params.get("patience", 3)
                print(f"      Early stopping: monitor={monitor_metric}, patience={patience}")
                
                # Train with early stopping
                history = model.fit(
                    X_cv_train,
                    y_cv_train,
                    epochs=epochs,
                    batch_size=params.get("batch_size", 64),
                    validation_data=(X_cv_val, y_cv_val),
                    verbose=1,  # Changed to verbose=1 to see training progress
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor=monitor_metric, 
                            patience=patience, 
                            restore_best_weights=True, 
                            mode=self.base_params.get("early_stopping_mode", "max")
                        )
                    ],
                )
                print(f"      Training completed: {len(history.history['loss'])} epochs")
            except Exception as e:
                print(f"      ❌ Model training failed: {e}")
                raise

            print("   📊 Evaluating model...")
            try:
                cv_results = model.evaluate(X_cv_val, y_cv_val, verbose=0)
                print(f"      Raw results: {cv_results}")
                print(f"      Metrics names: {model.metrics_names}")
                
                # The directional accuracy is in the compiled metrics
                # During evaluation, individual metrics are returned in order
                # Model metrics order: ['loss', 'mae', '_directional_accuracy', '_weighted_directional_accuracy', '_correlation_metric']
                if len(cv_results) >= 3:  # loss, mae, directional_accuracy, ...
                    score = cv_results[2]  # _directional_accuracy is the 3rd metric (index 2)
                    cv_scores.append(score)
                    print(f"      Fold {fold_count} score: {score:.4f}")
                else:
                    print(f"      ❌ Unexpected number of metrics: {len(cv_results)}")
                    cv_scores.append(0.5)  # fallback
                    
            except Exception as e:
                print(f"      ❌ Model evaluation failed: {e}")
                raise

        print(f"✅ Cross-validation completed: {len(cv_scores)} folds")
        
        # Return Mean CV Score
        mean_cv_score = np.mean(cv_scores)
        print(f"📈 Mean CV score: {mean_cv_score:.4f}")

        detailed_metrics = {
            "cv_directional_accuracy_mean": mean_cv_score,
            "cv_directional_accuracy_std": np.std(cv_scores),
            "cv_scores_all_folds": cv_scores,
            "temportal_cv_used": True,
        }

        return mean_cv_score, detailed_metrics

    def analyze_results(self):
        """Analyze tuning results to find patterns"""

        if not self.results:
            print("No results to analyze")
            return

        print(f"\n HYPERPARAMETER TUNING ANALYSIS")
        print("=" * 50)

        # Convert to DataFrame for analysis
        df_results = []
        for result in self.results:
            row = result["params"].copy()
            row["directional_accuracy"] = result["directional_accuracy"]
            row.update(result["metrics"])
            df_results.append(row)

        df = pd.DataFrame(df_results)

        # Top 5 performers
        top_5 = df.nlargest(5, "directional_accuracy")
        print(f"\n TOP 5 CONFIGURATIONS:")
        for i, (_, row) in enumerate(top_5.iterrows()):
            print(f"{i+1}. Directional Accuracy: {row['directional_accuracy']:.4f}")
            print(f"   directional_alpha: {row['directional_alpha']}")
            print(f"   learning_rate: {row['learning_rate']}")
            print(f"   batch_size: {row['batch_size']}")
            print(f"   dropout_rate: {row['dropout_rate']}")
            print()

        # Parameter impact analysis
        print(f" PARAMETER IMPACT ANALYSIS:")

        numeric_params = ["directional_alpha", "learning_rate", "dropout_rate", "l2_regularization"]

        for param in numeric_params:
            if param in df.columns:
                correlation = df[param].corr(df["directional_accuracy"])
                print(f"   {param}: correlation = {correlation:+.3f}")

        # Best parameter ranges
        print(f"\n OPTIMAL PARAMETER RANGES (Top 25%):")
        top_25_percent = df.nlargest(max(1, len(df) // 4), "directional_accuracy")

        for param in numeric_params:
            if param in top_25_percent.columns:
                min_val = top_25_percent[param].min()
                max_val = top_25_percent[param].max()
                mean_val = top_25_percent[param].mean()
                print(f"   {param}: [{min_val:.4f}, {max_val:.4f}] (mean: {mean_val:.4f})")

        return df

    def get_best_config(self):
        """Get the best configuration found"""

        if not self.results:
            return None

        best_result = max(self.results, key=lambda x: x["directional_accuracy"])

        return {
            "best_params": best_result["params"],
            "best_score": best_result["directional_accuracy"],
            "best_metrics": best_result["metrics"],
        }


def run_directional_tuning(data_splits, n_random_trials=15, run_grid_search=True):
    """
    Run comprehensive directional hyperparameter tuning
    """
    print(" STARTING DIRECTIONAL HYPERPARAMETER TUNING")
    print("=" * 60)

    # Handle different data split formats
    try:
        if isinstance(data_splits, tuple) and len(data_splits) == 7:
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names = data_splits
        elif isinstance(data_splits, dict):
            # Handle dictionary format from zero temporal loss trainer
            (X_train, y_train) = data_splits["train"]
            (X_val, y_val) = data_splits["val"]
            (X_test, y_test) = data_splits["test"]
            feature_names = data_splits.get("feature_names", [])
        else:
            raise ValueError(f"Unsupported data_splits format: {type(data_splits)}")

        print(f"Data format: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"Validation: X_val={X_val.shape}, y_val={y_val.shape}")

    except Exception as e:
        print(f" Error parsing data splits: {e}")
        print(f"Data splits type: {type(data_splits)}")
        if hasattr(data_splits, "__len__"):
            print(f"Data splits length: {len(data_splits)}")
        raise

    # Create tuner with optional config path
    config_path = None  # Could be passed as parameter
    tuner = DirectionalHyperparameterTuner(X_train, y_train, X_val, y_val, config_path)

    # Run random search
    best_random, best_random_score = tuner.random_search(n_trials=n_random_trials, epochs=12)

    # Run focused grid search
    if run_grid_search:
        best_grid, best_grid_score = tuner.grid_search_focused(epochs=10)

        # Compare results
        if best_grid_score > best_random_score:
            best_overall = best_grid
            best_overall_score = best_grid_score
            best_method = "Grid Search"
        else:
            best_overall = best_random
            best_overall_score = best_random_score
            best_method = "Random Search"
    else:
        best_overall = best_random
        best_overall_score = best_random_score
        best_method = "Random Search"

    # Analyze results
    results_df = tuner.analyze_results()

    print(f"\n HYPERPARAMETER TUNING COMPLETE!")
    print(f"Best Method: {best_method}")
    print(f"Best Directional Accuracy: {best_overall_score:.4f}")
    print(f"Best Configuration:")
    print(json.dumps(best_overall, indent=2))

    # Save results
    tuning_results = {
        "best_config": tuner.get_best_config(),
        "all_results": tuner.results,
        "summary": {"total_trials": len(tuner.results), "best_method": best_method, "best_score": best_overall_score},
    }

    with open("directional_tuning_results.json", "w") as f:
        json.dump(tuning_results, f, indent=2)

    print("Results saved to: directional_tuning_results.json")

    return best_overall, best_overall_score, tuning_results


if __name__ == "__main__":
    print("Directional hyperparameter tuning module")
    print("Import this module and call run_directional_tuning(data_splits)")
