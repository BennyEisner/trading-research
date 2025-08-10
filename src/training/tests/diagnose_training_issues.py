#!/usr/bin/env python3

"""
Comprehensive LSTM Training Diagnostics
Identify root causes of zero correlation and learning failures
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from config.config import get_config
from tests.data_loader import load_test_data
from src.training.shared_backbone_trainer import create_shared_backbone_trainer

def diagnose_training_issues():
    """Run comprehensive diagnostics on training pipeline"""
    
    print("=" * 80)
    print("LSTM TRAINING DIAGNOSTICS")
    print("Identifying root causes of zero correlation and learning failures")
    print("=" * 80)
    print()
    
    # Initialize trainer (use MAG7 for faster diagnosis)
    trainer = create_shared_backbone_trainer(use_expanded_universe=False)
    print(f"Diagnostic universe: {len(trainer.tickers)} tickers")
    
    # Load small dataset for analysis
    ticker_data = load_test_data(trainer.tickers, days=200)
    print(f"Loaded data for {len(ticker_data)} tickers")
    
    # Prepare training data
    print("\n" + "="*60)
    print("STEP 1: ANALYZE RAW FEATURES")
    print("="*60)
    
    training_data = trainer.prepare_training_data(ticker_data)
    
    # Get first ticker data for detailed analysis
    first_ticker = list(training_data.keys())[0]
    X_sample, y_sample = training_data[first_ticker]
    
    print(f"\nAnalyzing {first_ticker} data:")
    print(f"  Sequences: {X_sample.shape}")
    print(f"  Features per timestep: {X_sample.shape[2]}")
    print(f"  Target range: [{y_sample.min():.3f}, {y_sample.max():.3f}]")
    print(f"  Target mean: {y_sample.mean():.3f}")
    print(f"  Target std: {y_sample.std():.3f}")
    
    # DIAGNOSTIC 1: Feature Quality Analysis
    print("\n" + "-"*40)
    print("DIAGNOSTIC 1: Feature Quality")
    print("-"*40)
    
    # Analyze feature variance and distributions
    feature_names = trainer.pattern_engine.feature_calculator.get_feature_names()[:17]  # First 17 features
    
    print("Feature variance analysis:")
    for i, feature_name in enumerate(feature_names):
        feature_data = X_sample[:, :, i].flatten()  # Flatten across sequences and time
        
        # Calculate statistics
        variance = np.var(feature_data)
        mean = np.mean(feature_data)
        zero_pct = np.mean(feature_data == 0) * 100
        nan_pct = np.mean(np.isnan(feature_data)) * 100
        inf_pct = np.mean(np.isinf(feature_data)) * 100
        
        status = "OK"
        if variance < 1e-10:
            status = "ZERO_VAR"
        elif zero_pct > 90:
            status = "MOSTLY_ZERO"
        elif nan_pct > 5:
            status = "HIGH_NAN"
        elif inf_pct > 0:
            status = "HAS_INF"
        
        print(f"  {i:2d}. {feature_name:25s}: var={variance:.2e}, mean={mean:.3f}, zero%={zero_pct:.1f}, [{status}]")
    
    # DIAGNOSTIC 2: Target-Feature Correlation Analysis
    print("\n" + "-"*40)
    print("DIAGNOSTIC 2: Target-Feature Correlations")
    print("-"*40)
    
    # Calculate correlation between each feature and targets
    correlations = []
    for i, feature_name in enumerate(feature_names):
        # Use the last timestep of each sequence (most recent)
        feature_values = X_sample[:, -1, i]  # Last timestep of each sequence
        
        # Calculate correlation with targets
        correlation = np.corrcoef(feature_values, y_sample)[0, 1] if len(np.unique(feature_values)) > 1 else 0.0
        correlations.append((feature_name, correlation))
        
        print(f"  {feature_name:25s}: corr={correlation:.4f}")
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\nTop correlations:")
    for feature_name, corr in correlations[:5]:
        print(f"  {feature_name:25s}: {corr:.4f}")
    
    max_correlation = max(abs(corr) for _, corr in correlations)
    if max_correlation < 0.05:
        print(f"\n❌ CRITICAL ISSUE: Highest feature correlation = {max_correlation:.4f}")
        print("   This explains zero model correlation - no signal in features!")
    
    # DIAGNOSTIC 3: Target Distribution Analysis
    print("\n" + "-"*40)
    print("DIAGNOSTIC 3: Target Distribution")
    print("-"*40)
    
    print("Target distribution analysis:")
    print(f"  Range: [{y_sample.min():.3f}, {y_sample.max():.3f}]")
    print(f"  Mean: {y_sample.mean():.3f}")
    print(f"  Std: {y_sample.std():.3f}")
    
    # Check for target clustering
    unique_values = np.unique(y_sample)
    print(f"  Unique values: {len(unique_values)} out of {len(y_sample)} samples")
    
    if len(unique_values) < 10:
        print("  Target values:", unique_values[:10])
        print("❌ ISSUE: Very few unique target values - targets may be too quantized")
    
    # Check target autocorrelation (time series pattern)
    if len(y_sample) > 1:
        target_autocorr = np.corrcoef(y_sample[:-1], y_sample[1:])[0, 1]
        print(f"  Target autocorrelation: {target_autocorr:.4f}")
        
        if abs(target_autocorr) > 0.5:
            print("❌ ISSUE: High target autocorrelation - may indicate data leakage")
    
    # DIAGNOSTIC 4: Sequence Overlap Analysis
    print("\n" + "-"*40)
    print("DIAGNOSTIC 4: Sequence Overlap")
    print("-"*40)
    
    # Analyze temporal correlation between overlapping sequences
    stride = trainer.config.model.sequence_stride  # Should be 5
    lookback = trainer.config.model.lookback_window  # Should be 20
    
    overlap_ratio = (lookback - stride) / lookback
    print(f"  Sequence stride: {stride}")
    print(f"  Lookback window: {lookback}")
    print(f"  Overlap ratio: {overlap_ratio:.1%}")
    
    if overlap_ratio > 0.8:
        print("⚠️  WARNING: Very high overlap - may cause data leakage")
    
    # DIAGNOSTIC 5: External Data Validation
    print("\n" + "-"*40)
    print("DIAGNOSTIC 5: External Data Integration")
    print("-"*40)
    
    # Check if external data features are actually populated
    external_features = ['sector_relative_strength', 'market_beta_instability', 'vix_term_structure']
    
    for feature_name in external_features:
        if feature_name in feature_names:
            idx = feature_names.index(feature_name)
            feature_data = X_sample[:, -1, idx]  # Last timestep
            
            zero_pct = np.mean(feature_data == 0) * 100
            variance = np.var(feature_data)
            
            status = "WORKING" if variance > 1e-6 and zero_pct < 90 else "NOT_WORKING"
            print(f"  {feature_name:25s}: zero%={zero_pct:.1f}, var={variance:.2e} [{status}]")
    
    # DIAGNOSTIC 6: Data Leakage Check
    print("\n" + "-"*40)
    print("DIAGNOSTIC 6: Data Leakage Detection")
    print("-"*40)
    
    # Check if future information is leaking into features
    # Compare first timestep vs last timestep of sequences
    first_timestep_features = X_sample[:, 0, :].mean(axis=0)  # Average across sequences
    last_timestep_features = X_sample[:, -1, :].mean(axis=0)
    
    print("Feature evolution from first to last timestep:")
    large_changes = 0
    for i, feature_name in enumerate(feature_names):
        change = abs(last_timestep_features[i] - first_timestep_features[i])
        if change > 0.1:  # Significant change
            large_changes += 1
        if i < 5:  # Show first 5 features
            print(f"  {feature_name:25s}: {first_timestep_features[i]:.3f} → {last_timestep_features[i]:.3f}")
    
    if large_changes < 5:
        print("❌ ISSUE: Features barely change over time - may indicate static/leaked data")
    
    print("\n" + "="*60)
    print("SUMMARY OF ISSUES FOUND")
    print("="*60)
    
    issues = []
    if max_correlation < 0.05:
        issues.append("No predictive features (max correlation <0.05)")
    if len(unique_values) < 10:
        issues.append("Over-quantized targets")
    if overlap_ratio > 0.8:
        issues.append("Excessive sequence overlap")
    if large_changes < 5:
        issues.append("Features lack temporal variation")
    
    if issues:
        print("CRITICAL ISSUES IDENTIFIED:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nRECOMMENDATION: Fix these issues before training")
    else:
        print("No obvious issues found - problem may be in model architecture")
    
    return issues

if __name__ == "__main__":
    issues = diagnose_training_issues()
    exit(1 if issues else 0)