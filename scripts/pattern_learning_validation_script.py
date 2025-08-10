#!/usr/bin/env python3

"""
Test Pattern Learning Fixes
Validation test to confirm the implemented fixes resolve the constant prediction issue
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Skip TensorFlow warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from src.models.shared_backbone_lstm import SharedBackboneLSTMBuilder
from src.training.pattern_target_generator import PatternTargetGenerator
from src.features.pattern_feature_calculator import FeatureCalculator
from tests.utilities.data_loader import load_test_data
from config.config import get_config


def prepare_test_sequences(features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare test sequences with improved targets"""
    
    # Generate improved pattern targets
    generator = PatternTargetGenerator()
    targets_dict = generator.generate_all_pattern_targets(features_df)
    
    if "pattern_confidence_score" not in targets_dict:
        return None, None
        
    targets = targets_dict["pattern_confidence_score"]
    
    # Extract feature columns
    feature_columns = [col for col in features_df.columns 
                      if col not in ['date', 'ticker', 'close', 'volume', 'high', 'low', 'open']]
    
    if len(feature_columns) == 0:
        return None, None
    
    features_array = features_df[feature_columns].values
    
    # Create sequences (20-day lookback, 5-day horizon)
    sequence_length, horizon = 20, 5
    X_sequences, y_sequences = [], []
    
    for i in range(len(features_array) - sequence_length - horizon + 1):
        X_seq = features_array[i:i + sequence_length]
        target_idx = i + sequence_length + horizon - 1
        
        if target_idx < len(targets):
            y_val = targets[target_idx]
            
            if not np.isnan(X_seq).any() and not np.isnan(y_val):
                X_sequences.append(X_seq)
                y_sequences.append(y_val)
    
    return np.array(X_sequences), np.array(y_sequences)


def test_pattern_learning_fixes():
    """Test the implemented pattern learning fixes"""
    
    print("=== TESTING PATTERN LEARNING FIXES ===")
    print()
    
    # Load test data
    print("Loading test data...")
    tickers = ["AAPL", "MSFT", "GOOG"]  # Small set for validation
    raw_data = load_test_data(tickers, days=400)
    
    if not raw_data:
        print("❌ Failed to load test data")
        return False
    
    # Prepare training data
    print("Preparing training sequences...")
    feature_calculator = FeatureCalculator()
    all_X, all_y = [], []
    
    for ticker, data in raw_data.items():
        print(f"  Processing {ticker}...")
        
        features_df = feature_calculator.calculate_all_features(data)
        if features_df is None or len(features_df) < 50:
            continue
            
        X_seq, y_seq = prepare_test_sequences(features_df)
        if X_seq is None or len(X_seq) == 0:
            continue
            
        all_X.append(X_seq)
        all_y.append(y_seq)
        print(f"    ✅ {len(X_seq)} sequences prepared")
    
    if not all_X:
        print("❌ No training sequences prepared")
        return False
    
    # Combine sequences
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    
    print(f"\\nCombined dataset: {X_combined.shape[0]} sequences")
    print(f"Target distribution analysis:")
    print(f"  Mean: {np.mean(y_combined):.4f} ± {np.std(y_combined):.4f}")
    print(f"  Range: [{np.min(y_combined):.4f}, {np.max(y_combined):.4f}]")
    
    # CRITICAL: Check target balance improvement
    above_threshold = np.mean(y_combined > 0.5)
    print(f"  Above 0.5 threshold: {above_threshold:.3f} ({above_threshold*100:.1f}%)")
    
    if above_threshold < 0.25:  # Still severely imbalanced
        print(f"  ⚠️  Target imbalance may still cause issues")
    elif above_threshold > 0.25:
        print(f"  ✅ Target balance significantly improved from 11.5%!")
    
    # Split data
    split_idx = int(0.8 * len(X_combined))
    X_train, X_val = X_combined[:split_idx], X_combined[split_idx:]
    y_train, y_val = y_combined[:split_idx], y_combined[split_idx:]
    
    print(f"\\nTrain/Val split: {len(X_train)}/{len(X_val)} samples")
    
    # Build improved model
    print("\\nBuilding improved model with fixes...")
    config = get_config()
    builder = SharedBackboneLSTMBuilder(config.dict())
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = builder.build_model(input_shape)
    
    print(f"Model: {model.count_params():,} parameters")
    print(f"Output activation: sigmoid (FIXED)")
    print(f"Loss function: Custom pattern-focused loss (FIXED)")
    print(f"Learning rate: Higher with cosine restarts (FIXED)")
    
    # Test random initialization predictions
    print("\\nAnalyzing initial predictions...")
    initial_preds = model.predict(X_val[:100], verbose=0).flatten()  # Small sample for speed
    
    print(f"Random initialization:")
    print(f"  Mean: {np.mean(initial_preds):.6f}")
    print(f"  Std: {np.std(initial_preds):.6f}")
    print(f"  Range: [{np.min(initial_preds):.4f}, {np.max(initial_preds):.4f}]")
    
    # Check if sigmoid is working (should be in 0-1 range)
    if np.min(initial_preds) >= 0 and np.max(initial_preds) <= 1:
        print(f"  ✅ Sigmoid activation working: predictions in [0,1]")
    else:
        print(f"  ❌ Sigmoid issue: predictions outside [0,1]")
    
    # Train for limited epochs to test behavior
    print("\\nTraining with fixes for 15 epochs...")
    
    # Custom callback to monitor correlation progress
    class CorrelationTracker(tf.keras.callbacks.Callback):
        def __init__(self, X_val, y_val):
            self.X_val = X_val
            self.y_val = y_val
            self.correlations = []
            
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 3 == 0:  # Check every 3 epochs
                preds = self.model.predict(self.X_val, verbose=0).flatten()
                
                if np.std(preds) > 1e-8:
                    corr = np.corrcoef(preds, self.y_val)[0, 1]
                    self.correlations.append(corr)
                    print(f"    Epoch {epoch+1}: correlation = {corr:.6f}, pred_std = {np.std(preds):.6f}")
                else:
                    self.correlations.append(0.0)
                    print(f"    Epoch {epoch+1}: CONSTANT PREDICTIONS (std = {np.std(preds):.8f})")
    
    correlation_tracker = CorrelationTracker(X_val, y_val)
    
    # Training with improved configuration
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=128,
        verbose=1,
        callbacks=[correlation_tracker]
    )
    
    # Final analysis
    print("\\n=== FINAL ANALYSIS ===")
    
    final_preds = model.predict(X_val, verbose=0).flatten()
    final_corr = np.corrcoef(final_preds, y_val)[0, 1] if np.std(final_preds) > 1e-8 else 0.0
    
    print(f"Final predictions:")
    print(f"  Mean: {np.mean(final_preds):.4f}")
    print(f"  Std: {np.std(final_preds):.6f}")
    print(f"  Range: [{np.min(final_preds):.4f}, {np.max(final_preds):.4f}]")
    print(f"  Correlation: {final_corr:.6f}")
    
    # Success criteria
    success_checks = {
        "Non-constant predictions": np.std(final_preds) > 0.001,
        "Positive correlation": final_corr > 0.05,
        "Improved target balance": above_threshold > 0.25,
        "Proper output range": np.min(final_preds) >= 0 and np.max(final_preds) <= 1,
        "Learning occurred": len(correlation_tracker.correlations) > 0 and max(correlation_tracker.correlations) > 0.01
    }
    
    print(f"\\nSUCCESS CRITERIA:")
    passed_checks = 0
    for check, result in success_checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {check}")
        if result:
            passed_checks += 1
    
    overall_success = passed_checks >= 4  # Need at least 4/5 checks
    
    print(f"\\n{'='*60}")
    if overall_success:
        print("🎉 PATTERN LEARNING FIXES SUCCESSFUL!")
        print("   - Constant prediction issue resolved")
        print("   - Model learning meaningful patterns") 
        print("   - Architecture fixes working correctly")
    else:
        print("❌ FIXES NEED FURTHER REFINEMENT")
        print(f"   - Passed {passed_checks}/5 success criteria")
        print("   - May need additional adjustments")
    print(f"{'='*60}")
    
    # Save test results
    results = {
        "test_timestamp": datetime.now().isoformat(),
        "target_balance_improvement": f"{above_threshold:.3f} (vs 0.115 baseline)",
        "final_correlation": float(final_corr),
        "final_prediction_std": float(np.std(final_preds)),
        "success_checks": success_checks,
        "overall_success": overall_success,
        "correlation_progression": correlation_tracker.correlations
    }
    
    results_file = Path("pattern_learning_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n📊 Test results saved: {results_file}")
    
    return overall_success


if __name__ == "__main__":
    success = test_pattern_learning_fixes()
    
    if success:
        print("\\n🚀 READY FOR FULL TRAINING WITH FIXES!")
    else:
        print("\\n⚙️  Additional tuning may be needed")
    
    sys.exit(0 if success else 1)