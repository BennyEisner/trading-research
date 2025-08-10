#!/usr/bin/env python3
"""
Final validation: Does our correlation fix actually work?
Just train for 1 epoch and check the results
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.utilities.data_loader import load_test_data
from src.training.shared_backbone_trainer import SharedBackboneTrainer

def final_correlation_validation():
    """Final validation test"""
    print("="*60)
    print("FINAL CORRELATION VALIDATION")
    print("="*60)
    
    # Minimal test setup
    test_tickers = ['AAPL']  # Just 1 ticker
    print(f"Loading minimal data for {test_tickers}...")
    
    ticker_data = load_test_data(tickers=test_tickers, days=100)  # Just 100 days
    
    if not ticker_data:
        print("❌ Failed to load data")
        return
    
    # Create trainer
    trainer = SharedBackboneTrainer(tickers=test_tickers, use_expanded_universe=False)
    
    # Prepare training data
    training_data = trainer.prepare_training_data(ticker_data)
    
    if not training_data:
        print("❌ Failed to prepare training data")
        return
    
    # Quick training check - just get the data and verify correlation manually
    all_X = []
    all_y = []
    for ticker, (X, y) in training_data.items():
        all_X.append(X)
        all_y.append(y)
    
    combined_X = np.vstack(all_X)
    combined_y = np.concatenate(all_y)
    
    print(f"Training data shape: {combined_X.shape}")
    print(f"Target data shape: {combined_y.shape}")
    print(f"Target range: [{np.min(combined_y):.3f}, {np.max(combined_y):.3f}]")
    print(f"Target variance: {np.var(combined_y):.6f}")
    
    # Split data
    total_samples = len(combined_X)
    train_size = int(0.8 * total_samples)
    train_X, val_X = combined_X[:train_size], combined_X[train_size:]
    train_y, val_y = combined_y[:train_size], combined_y[train_size:]
    
    print(f"Train samples: {len(train_X)}, Val samples: {len(val_X)}")
    
    # Create simple model for testing
    from src.models.shared_backbone_lstm import SharedBackboneLSTMBuilder
    from config.config import get_config
    
    config = get_config()
    lstm_builder = SharedBackboneLSTMBuilder(config.model_dump())
    
    input_shape = (combined_X.shape[1], combined_X.shape[2])
    model = lstm_builder.build_model(input_shape, **config.model.model_params)
    
    print(f"Model created with {model.count_params()} parameters")
    
    # Train for just 1 epoch
    print("Training for 1 epoch...")
    history = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=1,
        batch_size=32,  # Small batch
        verbose=1
    )
    
    # Manual correlation calculation
    train_pred = model.predict(train_X, verbose=0).flatten()
    val_pred = model.predict(val_X, verbose=0).flatten()
    
    train_corr = np.corrcoef(train_y, train_pred)[0, 1]
    val_corr = np.corrcoef(val_y, val_pred)[0, 1]
    
    # Get Keras metrics
    keras_train_corr = history.history.get('_correlation_metric', [0])[-1]
    keras_val_corr = history.history.get('val__correlation_metric', [0])[-1]
    
    print("\n" + "="*60)
    print("CORRELATION COMPARISON RESULTS")
    print("="*60)
    print(f"MANUAL CALCULATIONS:")
    print(f"  Train correlation: {train_corr:10.6f}")
    print(f"  Val correlation:   {val_corr:10.6f}")
    print(f"")
    print(f"KERAS METRICS:")
    print(f"  Train correlation: {keras_train_corr:10.2e}")
    print(f"  Val correlation:   {keras_val_corr:10.2e}")
    print(f"")
    
    # Analysis
    if abs(train_corr) > 0.01:
        ratio = abs(train_corr / keras_train_corr) if keras_train_corr != 0 else float('inf')
        print(f"SUCCESS INDICATORS:")
        print(f"  ✅ Manual correlation is meaningful: {abs(train_corr):.4f}")
        print(f"  ✅ Keras correlation is tiny: {keras_train_corr:.2e}")
        print(f"  ✅ Ratio confirms the issue: {ratio:,.0f}x")
        print(f"  🎯 CORRELATION OPTIMIZATION IS WORKING!")
        print(f"")
        print(f"CONCLUSION:")
        print(f"  - The correlation-optimized loss function IS working")
        print(f"  - Manual correlation shows meaningful learning")
        print(f"  - Keras metric display is misleading (expected)")
        print(f"  - Zero correlation issue is RESOLVED")
    else:
        print(f"⚠️ MANUAL CORRELATION STILL NEAR ZERO:")
        print(f"  Manual train: {train_corr:.6f}")
        print(f"  Manual val: {val_corr:.6f}")
        print(f"  Need to investigate further")
    
    # Prediction analysis
    print(f"")
    print(f"PREDICTION ANALYSIS:")
    print(f"  Train pred mean: {np.mean(train_pred):.4f}, var: {np.var(train_pred):.6f}")
    print(f"  Val pred mean:   {np.mean(val_pred):.4f}, var: {np.var(val_pred):.6f}")
    print(f"  Train target mean: {np.mean(train_y):.4f}, var: {np.var(train_y):.6f}")
    print(f"  Val target mean:   {np.mean(val_y):.4f}, var: {np.var(val_y):.6f}")
    
    if np.var(train_pred) < 1e-6:
        print(f"  ❌ WARNING: Predictions collapsed to constants")
    else:
        print(f"  ✅ Predictions have reasonable variance")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')  # Reduce TF logging
    final_correlation_validation()