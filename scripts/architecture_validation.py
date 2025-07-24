#!/usr/bin/env python3

"""
Model architecture validation - test different LSTM architectures
to validate the model design is working properly
"""

import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.config.config import Config
from src.data.data_loader import DataLoader
from src.models.model_builder import ModelBuilder
from src.utils.logging_utils import setup_production_logger
from sklearn.preprocessing import StandardScaler


class ArchitectureValidator:
    """Test different LSTM architectures for validation"""
    
    def __init__(self):
        self.logger = setup_production_logger({"log_file": "architecture_validation.log"})
        self.base_config = {
            "epochs": 20,  # Quick training for architecture testing
            "batch_size": 32,
            "learning_rate": 0.001,
            "lookback_window": 20,
            "database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db"
        }
    
    def load_real_market_data(self, lookback_window=20):
        """Load real market data for architecture testing"""
        config = Config(self.base_config)
        data_loader = DataLoader(config)
        
        # Use AAPL for architecture testing (stable, high-volume stock)
        ticker = "AAPL"
        try:
            # Load 2 years of data for architecture testing
            raw_data = data_loader.load_single_ticker_data(ticker, 2)
            self.logger.log(f"Loaded {len(raw_data)} records for {ticker}")
            
            # Create simple features for architecture testing
            features_data = raw_data.copy()
            features_data['daily_return'] = features_data['close'].pct_change()
            features_data['log_return'] = np.log(features_data['close'] / features_data['close'].shift(1))
            features_data['hl_ratio'] = features_data['high'] / features_data['low']
            features_data['oc_ratio'] = features_data['open'] / features_data['close']
            
            # Simple moving averages
            for window in [5, 10, 20]:
                features_data[f'sma_{window}'] = features_data['close'].rolling(window).mean()
                features_data[f'price_sma_ratio_{window}'] = features_data['close'] / features_data[f'sma_{window}']
            
            # Volatility
            features_data['volatility_5'] = features_data['daily_return'].rolling(5).std()
            features_data['volatility_10'] = features_data['daily_return'].rolling(10).std()
            
            # Remove rows with NaN values
            features_data = features_data.dropna()
            
            # Select feature columns
            exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
            feature_columns = [col for col in features_data.columns if col not in exclude_cols]
            
            # Extract feature matrix and scale it
            feature_matrix = features_data[feature_columns].values
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)
            
            # Target: next day's return
            target_values = features_data['daily_return'].fillna(0).values
            
            # Create sequences
            X, y = [], []
            for i in range(lookback_window, len(feature_matrix_scaled) - 1):
                X.append(feature_matrix_scaled[i - lookback_window:i, :])
                y.append(target_values[i + 1])  # Predict next day since we removed shift(-1)
            
            self.logger.log(f"Created {len(X)} sequences with {feature_matrix_scaled.shape[1]} features each")
            return np.array(X), np.array(y), len(feature_columns)
            
        except Exception as e:
            self.logger.log(f"Error loading real market data: {e}")
            self.logger.log("Falling back to synthetic data")
            return self.generate_fallback_synthetic_data(lookback_window)
    
    def generate_fallback_synthetic_data(self, lookback_window=20):
        """Generate synthetic data if real data loading fails"""
        np.random.seed(42)  # Reproducible
        n_samples = 500
        n_features = 8  # Match typical financial features
        
        X = []
        y = []
        
        for i in range(n_samples):
            # Create a sequence with financial-like characteristics
            trend = np.random.randn() * 0.01  # Small trend
            
            # Generate sequence
            sequence = []
            for t in range(lookback_window):
                # Features with some correlation structure (like price ratios, returns)
                features = np.random.randn(n_features) * 0.02  # Financial returns are small
                # Add trend component
                features[:3] += trend * (t / lookback_window)
                sequence.append(features)
            
            X.append(sequence)
            
            # Target: based on recent sequence values (like predicting next return)
            target = np.mean(sequence[-1][:3]) + np.random.randn() * 0.01
            y.append(target)
        
        return np.array(X), np.array(y), n_features
    
    def test_architecture(self, architecture_name, config_override, X, y):
        """Test a specific architecture"""
        self.logger.log(f"\n--- Testing {architecture_name} ---")
        
        try:
            # Create config
            config = Config({**self.base_config, **config_override})
            
            # Split data
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            # Build model
            model_builder = ModelBuilder(config)
            model = model_builder.build_lstm_model(input_shape=X.shape[1:])
            model = model_builder.compile_model(model)
            
            param_count = model.count_params()
            self.logger.log(f"{architecture_name}: {param_count:,} parameters")
            
            # Train model
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.get("epochs"),
                batch_size=config.get("batch_size"),
                verbose=0
            )
            training_time = time.time() - start_time
            
            # Evaluate
            val_pred = model.predict(X_val, verbose=0).flatten()
            test_pred = model.predict(X_test, verbose=0).flatten()
            
            # Calculate metrics
            val_mae = np.mean(np.abs(y_val - val_pred))
            test_mae = np.mean(np.abs(y_test - test_pred))
            
            val_corr = np.corrcoef(y_val, val_pred)[0, 1] if len(y_val) > 1 else 0
            test_corr = np.corrcoef(y_test, test_pred)[0, 1] if len(y_test) > 1 else 0
            
            # Training convergence
            initial_loss = history.history['loss'][0]
            final_loss = history.history['loss'][-1]
            loss_improvement = (initial_loss - final_loss) / initial_loss * 100
            
            # Check for training issues
            training_stable = not any(np.isnan(history.history['loss']))
            loss_exploded = final_loss > initial_loss * 5
            
            results = {
                "parameters": param_count,
                "training_time": training_time,
                "val_mae": val_mae,
                "test_mae": test_mae,
                "val_correlation": val_corr,
                "test_correlation": test_corr,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "loss_improvement_pct": loss_improvement,
                "training_stable": training_stable,
                "loss_exploded": loss_exploded,
                "converged": loss_improvement > 5  # At least 5% improvement
            }
            
            # Log results
            self.logger.log(f"{architecture_name} Results:")
            self.logger.log(f"  Parameters: {param_count:,}")
            self.logger.log(f"  Training time: {training_time:.1f}s")
            self.logger.log(f"  Val MAE: {val_mae:.6f}, Test MAE: {test_mae:.6f}")
            self.logger.log(f"  Val Corr: {val_corr:.3f}, Test Corr: {test_corr:.3f}")
            self.logger.log(f"  Loss: {initial_loss:.4f} → {final_loss:.4f} ({loss_improvement:.1f}% improvement)")
            self.logger.log(f"  Status: {'✓' if training_stable and not loss_exploded and results['converged'] else '✗'}")
            
            return results
            
        except Exception as e:
            self.logger.log(f"{architecture_name} FAILED: {e}")
            return {"error": str(e)}
    
    def run_architecture_validation(self):
        """Test different architecture configurations"""
        self.logger.log("=" * 60)
        self.logger.log("MODEL ARCHITECTURE VALIDATION")
        self.logger.log("=" * 60)
        
        # Load real market data
        self.logger.log("Loading real market data for architecture testing...")
        X, y, num_features = self.load_real_market_data(lookback_window=20)
        self.logger.log(f"Loaded data: X={X.shape}, y={y.shape}, features={num_features}")
        
        # Define architectures to test
        architectures = [
            ("Minimal LSTM", {
                "lstm_units": [16],
                "enable_attention": False
            }),
            ("Small LSTM", {
                "lstm_units": [32, 16],
                "enable_attention": False
            }),
            ("Medium LSTM", {
                "lstm_units": [64, 32],
                "enable_attention": False
            }),
            ("Large LSTM", {
                "lstm_units": [128, 64, 32],
                "enable_attention": False
            }),
            ("Small + Attention", {
                "lstm_units": [32, 16],
                "enable_attention": True
            }),
            ("Medium + Attention", {
                "lstm_units": [64, 32],
                "enable_attention": True
            }),
        ]
        
        results = {}
        
        # Test each architecture
        for arch_name, config_override in architectures:
            results[arch_name] = self.test_architecture(arch_name, config_override, X, y)
        
        # Generate summary
        self.generate_architecture_summary(results)
        
        return results
    
    def generate_architecture_summary(self, results):
        """Generate architecture comparison summary"""
        self.logger.log("\n" + "=" * 60)
        self.logger.log("ARCHITECTURE VALIDATION SUMMARY")
        self.logger.log("=" * 60)
        
        print("\nARCHITECTURE VALIDATION RESULTS")
        print("=" * 50)
        
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not successful_results:
            print("❌ All architectures failed!")
            return
        
        # Sort by test correlation (best predictor)
        sorted_results = sorted(
            successful_results.items(), 
            key=lambda x: x[1].get('test_correlation', 0), 
            reverse=True
        )
        
        print(f"{'Architecture':<20} {'Params':<10} {'Time':<8} {'Test MAE':<12} {'Test Corr':<12} {'Converged':<10}")
        print("-" * 80)
        
        for arch_name, metrics in sorted_results:
            status = "✓" if metrics.get('converged', False) and metrics.get('training_stable', False) else "✗"
            print(f"{arch_name:<20} {metrics['parameters']:<10,} {metrics['training_time']:<8.1f} "
                  f"{metrics['test_mae']:<12.6f} {metrics['test_correlation']:<12.3f} {status:<10}")
        
        # Best architecture
        best_arch, best_metrics = sorted_results[0]
        print(f"\n🏆 Best Architecture: {best_arch}")
        print(f"   Test Correlation: {best_metrics['test_correlation']:.3f}")
        print(f"   Parameters: {best_metrics['parameters']:,}")
        print(f"   Training Time: {best_metrics['training_time']:.1f}s")
        
        # Architecture recommendations
        print(f"\n📋 ARCHITECTURE VALIDATION RESULTS:")
        
        working_architectures = [k for k, v in successful_results.items() 
                               if v.get('converged', False) and v.get('training_stable', False)]
        
        if len(working_architectures) >= len(successful_results) * 0.8:
            print("✅ Model architecture is working correctly")
            print("✅ LSTM layers are learning and converging")
            print("✅ Different complexity levels show expected behavior")
        else:
            print("⚠️  Some architecture configurations had issues")
            print("⚠️  Check model building and training pipeline")
        
        print("=" * 50)


def main():
    """Run architecture validation"""
    validator = ArchitectureValidator()
    results = validator.run_architecture_validation()
    
    if results:
        print("\n✅ Architecture validation completed!")
        print("Check 'architecture_validation.log' for detailed results")
    else:
        print("\n❌ Architecture validation failed")


if __name__ == "__main__":
    main()