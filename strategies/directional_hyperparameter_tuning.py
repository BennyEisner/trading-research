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

sys.path.append('.')

from src.models.multi_scale_lstm import MultiScaleLSTMBuilder


class DirectionalHyperparameterTuner:
    """
    Hyperparameter tuning focused on maximizing directional accuracy
    """
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.results = []
    
    def define_search_space(self):
        """Define hyperparameter search space for directional optimization"""
        
        search_space = {
            # Directional loss parameters
            'directional_alpha': [0.2, 0.3, 0.4, 0.5, 0.6],
            
            # Learning parameters
            'learning_rate': [0.0003, 0.0005, 0.001, 0.002],
            'batch_size': [32, 64, 96, 128],
            
            # Regularization
            'dropout_rate': [0.3, 0.4, 0.5],
            'l2_regularization': [0.001, 0.003, 0.005, 0.01],
            
            # Architecture parameters
            'lstm_units_1': [256, 384, 512],
            'lstm_units_2': [128, 192, 256], 
            'lstm_units_3': [64, 96, 128],
            'dense_layers': [[128, 64], [256, 128, 64], [384, 192, 64]]
        }
        
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
            params = {
                key: np.random.choice(values) 
                for key, values in search_space.items()
            }
            
            print(f"Testing: {json.dumps(params, indent=2)}")
            
            try:
                # Train model with these parameters
                score, detailed_metrics = self._evaluate_params(params, epochs)
                
                # Store results
                result = {
                    'trial': trial + 1,
                    'params': params,
                    'directional_accuracy': score,
                    'metrics': detailed_metrics
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
        
        # Focus on most impactful parameters
        focused_space = {
            'directional_alpha': [0.3, 0.4, 0.5],
            'learning_rate': [0.0005, 0.001],
            'batch_size': [64, 96],
            'dropout_rate': [0.4, 0.5],
        }
        
        # Fixed architecture for speed
        fixed_params = {
            'lstm_units_1': 384,
            'lstm_units_2': 192,
            'lstm_units_3': 96,
            'dense_layers': [256, 128, 64],
            'l2_regularization': 0.005,
            'use_attention': True
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
            
            print(f"\nGrid Search {i + 1}/{len(combinations)}")
            print(f"Parameters: {json.dumps({k: v for k, v in params.items() if k in focused_space}, indent=2)}")
            
            try:
                score, detailed_metrics = self._evaluate_params(params, epochs)
                
                result = {
                    'grid_search': i + 1,
                    'params': params,
                    'directional_accuracy': score,
                    'metrics': detailed_metrics
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
        """
        Evaluate a set of hyperparameters
        """
        # Build model with these parameters
        builder = MultiScaleLSTMBuilder({})
        model = builder.build_directional_focused_model(
            self.X_train.shape[1:], 
            **params
        )
        
        # Train with early stopping
        history = model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=params.get('batch_size', 64),
            validation_data=(self.X_val, self.y_val),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val__directional_accuracy',
                    patience=3,
                    restore_best_weights=True,
                    mode='max'
                )
            ]
        )
        
        # Evaluate directional performance
        val_results = model.evaluate(self.X_val, self.y_val, verbose=0)
        
        # Get directional accuracy (should be in metrics)
        metrics_dict = dict(zip(model.metrics_names, val_results))
        directional_accuracy = metrics_dict.get('_directional_accuracy', 0.5)
        
        # Additional metrics for analysis
        predictions = model.predict(self.X_val, verbose=0).flatten()
        
        detailed_metrics = {
            'val_loss': metrics_dict.get('loss', float('inf')),
            'val_mae': metrics_dict.get('mae', float('inf')),
            'val_directional_accuracy': directional_accuracy,
            'val_weighted_directional_accuracy': metrics_dict.get('_weighted_directional_accuracy', 0.5),
            'val_up_down_accuracy': metrics_dict.get('_up_down_accuracy', 0.5),
            'final_epoch': len(history.history['loss']),
            'training_stability': np.std(history.history['val__directional_accuracy'][-5:]) if len(history.history.get('val__directional_accuracy', [])) >= 5 else 1.0
        }
        
        return directional_accuracy, detailed_metrics
    
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
            row = result['params'].copy()
            row['directional_accuracy'] = result['directional_accuracy']
            row.update(result['metrics'])
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Top 5 performers
        top_5 = df.nlargest(5, 'directional_accuracy')
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
        
        numeric_params = ['directional_alpha', 'learning_rate', 'dropout_rate', 'l2_regularization']
        
        for param in numeric_params:
            if param in df.columns:
                correlation = df[param].corr(df['directional_accuracy'])
                print(f"   {param}: correlation = {correlation:+.3f}")
        
        # Best parameter ranges
        print(f"\n OPTIMAL PARAMETER RANGES (Top 25%):")
        top_25_percent = df.nlargest(max(1, len(df) // 4), 'directional_accuracy')
        
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
        
        best_result = max(self.results, key=lambda x: x['directional_accuracy'])
        
        return {
            'best_params': best_result['params'],
            'best_score': best_result['directional_accuracy'],
            'best_metrics': best_result['metrics']
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
            (X_train, y_train) = data_splits['train']
            (X_val, y_val) = data_splits['val']
            (X_test, y_test) = data_splits['test']
            feature_names = data_splits.get('feature_names', [])
        else:
            raise ValueError(f"Unsupported data_splits format: {type(data_splits)}")
        
        print(f"Data format: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"Validation: X_val={X_val.shape}, y_val={y_val.shape}")
        
    except Exception as e:
        print(f" Error parsing data splits: {e}")
        print(f"Data splits type: {type(data_splits)}")
        if hasattr(data_splits, '__len__'):
            print(f"Data splits length: {len(data_splits)}")
        raise
    
    # Create tuner
    tuner = DirectionalHyperparameterTuner(X_train, y_train, X_val, y_val)
    
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
        'best_config': tuner.get_best_config(),
        'all_results': tuner.results,
        'summary': {
            'total_trials': len(tuner.results),
            'best_method': best_method,
            'best_score': best_overall_score
        }
    }
    
    with open('directional_tuning_results.json', 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    print("Results saved to: directional_tuning_results.json")
    
    return best_overall, best_overall_score, tuning_results

if __name__ == "__main__":
    print("Directional hyperparameter tuning module")
    print("Import this module and call run_directional_tuning(data_splits)")
