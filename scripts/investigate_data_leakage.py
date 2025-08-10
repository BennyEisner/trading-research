#!/usr/bin/env python3

"""
Data Leakage Investigation Script
Critical analysis of the suspicious 69% correlation
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.pattern_feature_calculator import FeatureCalculator
from src.training.pattern_target_generator import PatternTargetGenerator
from src.training.shared_backbone_trainer import SharedBackboneTrainer


class DataLeakageInvestigator:
    """Systematic investigation of potential data leakage in LSTM training"""
    
    def __init__(self):
        self.results = {}
        
    def investigate_target_generation(self, ticker_data):
        """Phase 1.1: Analyze temporal dependencies in target generation"""
        
        print("="*80)
        print("PHASE 1.1: TARGET GENERATION TEMPORAL DEPENDENCY AUDIT")
        print("="*80)
        
        # Test with a single ticker first
        ticker = "AAPL"
        if ticker not in ticker_data:
            ticker = list(ticker_data.keys())[0]
        
        data = ticker_data[ticker]
        
        # Calculate features
        feature_calc = FeatureCalculator(symbol=ticker)
        features_df = feature_calc.calculate_all_features(data)
        
        # Generate targets
        target_gen = PatternTargetGenerator(lookback_window=20, validation_horizons=[3, 5, 10])
        targets = target_gen.generate_all_pattern_targets(features_df, primary_horizon=5)
        
        # Audit each target type for temporal dependencies
        self._audit_momentum_persistence_target(features_df, targets)
        self._audit_volatility_regime_target(features_df, targets)
        self._audit_trend_exhaustion_target(features_df, targets) 
        self._audit_volume_divergence_target(features_df, targets)
        
        # Test target-feature correlations at same timepoint (should be near zero)
        self._test_simultaneous_correlations(features_df, targets)
        
        return self.results.get('target_audit', {})
    
    def _audit_momentum_persistence_target(self, features_df, targets):
        """Audit momentum persistence target for data leakage"""
        
        print("\n--- MOMENTUM PERSISTENCE TARGET AUDIT ---")
        
        momentum_feature = features_df["momentum_persistence_7d"].values
        returns_1d = features_df["returns_1d"].values
        target_values = targets["momentum_persistence_binary"]
        
        # Check for circular dependencies
        issues = []
        
        # Test: correlation between target and features at same timepoint
        same_time_corr = np.corrcoef(momentum_feature[~np.isnan(momentum_feature)], 
                                   target_values[~np.isnan(momentum_feature)])[0,1]
        
        print(f"Target vs momentum feature correlation (same timepoint): {same_time_corr:.4f}")
        if abs(same_time_corr) > 0.3:
            issues.append(f"HIGH CORRELATION ({same_time_corr:.4f}) between target and input feature at same time")
        
        # Test: check if targets use future information
        future_dependency_score = self._test_future_dependency(
            features_df, target_values, "momentum_persistence_7d"
        )
        print(f"Future dependency score: {future_dependency_score:.4f}")
        if future_dependency_score > 0.2:
            issues.append(f"Target appears to use future information (score: {future_dependency_score:.4f})")
        
        # Test: validate that "historical" periods don't include current timepoint
        lookback_window = 20
        circular_dependency_count = 0
        
        for i in range(lookback_window, len(features_df) - 5):
            # Check if target generation logic includes current timepoint
            if target_values[i] > 0.5:  # Only check positive targets
                # Simulate the target generation logic 
                current_momentum = momentum_feature[i]
                historical_period = slice(max(0, i - lookback_window), i)  # Should NOT include i
                
                # Check if current timepoint data influences target
                historical_returns = returns_1d[historical_period]
                
                # This would be the problematic pattern
                if not np.isnan(current_momentum) and len(historical_returns) > 0:
                    # If historical calculation period overlaps with current timepoint, that's leakage
                    pass  # Logic inspection needed
        
        if issues:
            print("🚨 MOMENTUM PERSISTENCE ISSUES DETECTED:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ No obvious momentum persistence leakage detected")
            
        self.results.setdefault('target_audit', {})['momentum_persistence'] = {
            'same_time_correlation': same_time_corr,
            'future_dependency_score': future_dependency_score,
            'issues': issues
        }
    
    def _audit_volatility_regime_target(self, features_df, targets):
        """Audit volatility regime target for data leakage"""
        
        print("\n--- VOLATILITY REGIME TARGET AUDIT ---")
        
        vol_feature = features_df["volatility_regime_change"].values
        target_values = targets["volatility_regime_binary"]
        
        # Same timepoint correlation test
        valid_mask = ~(np.isnan(vol_feature) | np.isnan(target_values))
        if np.sum(valid_mask) > 1:
            same_time_corr = np.corrcoef(vol_feature[valid_mask], target_values[valid_mask])[0,1]
            print(f"Target vs volatility feature correlation (same timepoint): {same_time_corr:.4f}")
            
            if abs(same_time_corr) > 0.3:
                print(f"🚨 HIGH CORRELATION ({same_time_corr:.4f}) - possible data leakage")
            else:
                print("✅ Volatility regime correlation within acceptable range")
                
            self.results.setdefault('target_audit', {})['volatility_regime'] = {
                'same_time_correlation': same_time_corr,
                'valid_samples': np.sum(valid_mask)
            }
    
    def _audit_trend_exhaustion_target(self, features_df, targets):
        """Audit trend exhaustion target for data leakage"""
        
        print("\n--- TREND EXHAUSTION TARGET AUDIT ---")
        
        trend_feature = features_df["trend_exhaustion"].values
        target_values = targets["trend_exhaustion_binary"]
        
        # Same timepoint correlation test
        valid_mask = ~(np.isnan(trend_feature) | np.isnan(target_values))
        if np.sum(valid_mask) > 1:
            same_time_corr = np.corrcoef(trend_feature[valid_mask], target_values[valid_mask])[0,1]
            print(f"Target vs trend exhaustion correlation (same timepoint): {same_time_corr:.4f}")
            
            if abs(same_time_corr) > 0.3:
                print(f"🚨 HIGH CORRELATION ({same_time_corr:.4f}) - possible data leakage")
            else:
                print("✅ Trend exhaustion correlation within acceptable range")
                
            self.results.setdefault('target_audit', {})['trend_exhaustion'] = {
                'same_time_correlation': same_time_corr,
                'valid_samples': np.sum(valid_mask)
            }
    
    def _audit_volume_divergence_target(self, features_df, targets):
        """Audit volume divergence target for data leakage"""
        
        print("\n--- VOLUME DIVERGENCE TARGET AUDIT ---")
        
        vol_div_feature = features_df["volume_price_divergence"].values
        target_values = targets["volume_divergence_binary"]
        
        # Same timepoint correlation test
        valid_mask = ~(np.isnan(vol_div_feature) | np.isnan(target_values))
        if np.sum(valid_mask) > 1:
            same_time_corr = np.corrcoef(vol_div_feature[valid_mask], target_values[valid_mask])[0,1]
            print(f"Target vs volume divergence correlation (same timepoint): {same_time_corr:.4f}")
            
            if abs(same_time_corr) > 0.3:
                print(f"🚨 HIGH CORRELATION ({same_time_corr:.4f}) - possible data leakage")
            else:
                print("✅ Volume divergence correlation within acceptable range")
                
            self.results.setdefault('target_audit', {})['volume_divergence'] = {
                'same_time_correlation': same_time_corr,
                'valid_samples': np.sum(valid_mask)
            }
    
    def _test_simultaneous_correlations(self, features_df, targets):
        """Test correlations between targets and all input features at same timepoint"""
        
        print("\n--- TARGET vs INPUT FEATURE CORRELATIONS (SAME TIMEPOINT) ---")
        print("These should be near zero if targets are properly constructed")
        
        combined_target = targets["pattern_confidence_score"]
        high_correlations = []
        
        feature_names = [
            "price_acceleration", "volume_price_divergence", "volatility_regime_change",
            "return_skewness_7d", "momentum_persistence_7d", "volatility_clustering",
            "trend_exhaustion", "garch_volatility_forecast", "intraday_range_expansion",
            "overnight_gap_behavior", "end_of_day_momentum", "sector_relative_strength",
            "market_beta_instability", "vix_term_structure", "returns_1d", "returns_3d", 
            "returns_7d", "volume_normalized"
        ]
        
        for feature_name in feature_names:
            if feature_name in features_df.columns:
                feature_values = features_df[feature_name].values
                
                valid_mask = ~(np.isnan(feature_values) | np.isnan(combined_target))
                if np.sum(valid_mask) > 10:
                    correlation = np.corrcoef(feature_values[valid_mask], combined_target[valid_mask])[0,1]
                    
                    print(f"  {feature_name:25}: {correlation:6.3f}")
                    
                    if abs(correlation) > 0.4:
                        high_correlations.append((feature_name, correlation))
        
        if high_correlations:
            print("\n🚨 HIGH CORRELATIONS DETECTED (possible data leakage):")
            for feature_name, correlation in high_correlations:
                print(f"   - {feature_name}: {correlation:.3f}")
        else:
            print("\n✅ All feature-target correlations within acceptable range")
            
        self.results.setdefault('target_audit', {})['simultaneous_correlations'] = high_correlations
    
    def _test_future_dependency(self, features_df, target_values, feature_name):
        """Test if targets depend on future feature values"""
        
        feature_values = features_df[feature_name].values
        future_dependency_scores = []
        
        # Test correlations with future feature values (1, 3, 5 days ahead)
        for lag in [1, 3, 5]:
            if len(feature_values) > lag:
                future_features = feature_values[lag:]
                current_targets = target_values[:-lag]
                
                valid_mask = ~(np.isnan(future_features) | np.isnan(current_targets))
                if np.sum(valid_mask) > 10:
                    future_corr = abs(np.corrcoef(future_features[valid_mask], 
                                                 current_targets[valid_mask])[0,1])
                    future_dependency_scores.append(future_corr)
        
        return np.mean(future_dependency_scores) if future_dependency_scores else 0.0
    
    def run_random_target_test(self, ticker_data):
        """Phase 2.1: Test with random targets to confirm feature leakage"""
        
        print("\n" + "="*80)
        print("PHASE 2.1: RANDOM TARGET TEST")
        print("="*80)
        print("If correlation remains high with random targets, confirms feature leakage")
        
        # Use small subset for quick test
        test_ticker = "AAPL" if "AAPL" in ticker_data else list(ticker_data.keys())[0]
        data = ticker_data[test_ticker]
        
        # Prepare data with proper features
        trainer = SharedBackboneTrainer(tickers=[test_ticker], use_expanded_universe=False)
        training_data = trainer.prepare_training_data({test_ticker: data})
        
        if test_ticker not in training_data:
            print(f"❌ Could not prepare training data for {test_ticker}")
            return
        
        X, y_original = training_data[test_ticker]
        
        # Replace targets with random values
        np.random.seed(42)  # Reproducible results
        y_random = np.random.uniform(0, 1, len(y_original))
        
        # Quick correlation test (without full training)
        # Test if input features at current timepoint correlate with random targets
        
        if len(X.shape) == 3:  # (samples, sequence_length, features)
            # Use last timepoint of each sequence (most recent data)
            last_timepoint_features = X[:, -1, :]  # Shape: (samples, features)
            
            feature_correlations = []
            for feature_idx in range(last_timepoint_features.shape[1]):
                feature_vals = last_timepoint_features[:, feature_idx]
                
                valid_mask = ~(np.isnan(feature_vals) | np.isnan(y_random))
                if np.sum(valid_mask) > 10:
                    corr = abs(np.corrcoef(feature_vals[valid_mask], y_random[valid_mask])[0,1])
                    feature_correlations.append(corr)
            
            max_random_correlation = np.max(feature_correlations) if feature_correlations else 0.0
            mean_random_correlation = np.mean(feature_correlations) if feature_correlations else 0.0
            
            print(f"Random target correlation with features:")
            print(f"  Maximum correlation: {max_random_correlation:.4f}")
            print(f"  Mean correlation: {mean_random_correlation:.4f}")
            print(f"  Expected: ~0.05 for random data")
            
            if max_random_correlation > 0.15:
                print("🚨 HIGH CORRELATION WITH RANDOM TARGETS - confirms feature construction issues")
            else:
                print("✅ Random target correlations normal - features appear properly constructed")
                
            self.results['random_target_test'] = {
                'max_correlation': max_random_correlation,
                'mean_correlation': mean_random_correlation,
                'samples_tested': len(y_random),
                'features_tested': len(feature_correlations)
            }
    
    def run_gap_sensitivity_test(self, ticker_data):
        """Phase 2.2: Test correlation sensitivity to temporal gaps"""
        
        print("\n" + "="*80) 
        print("PHASE 2.2: GAP SENSITIVITY TEST")
        print("="*80)
        print("True patterns degrade gradually, data leakage drops sharply with gaps")
        
        # Test with different gap sizes
        gap_sizes = [0, 1, 5, 10, 20]
        gap_correlations = {}
        
        test_ticker = "AAPL" if "AAPL" in ticker_data else list(ticker_data.keys())[0]
        data = ticker_data[test_ticker]
        
        print(f"Testing gap sensitivity with {test_ticker}...")
        
        for gap_size in gap_sizes:
            print(f"\nTesting gap size: {gap_size} days")
            
            try:
                # Prepare data with specific gap
                correlation = self._test_specific_gap_size(data, test_ticker, gap_size)
                gap_correlations[gap_size] = correlation
                print(f"  Correlation with {gap_size}-day gap: {correlation:.4f}")
                
            except Exception as e:
                print(f"  Error testing gap {gap_size}: {e}")
                gap_correlations[gap_size] = None
        
        # Analyze gap sensitivity pattern
        valid_correlations = [(gap, corr) for gap, corr in gap_correlations.items() 
                             if corr is not None]
        
        if len(valid_correlations) >= 3:
            print("\n--- GAP SENSITIVITY ANALYSIS ---")
            
            # Check for sharp drop (indicates leakage) vs gradual decline (real patterns)
            correlations = [corr for gap, corr in valid_correlations]
            max_corr = max(correlations)
            min_corr = min(correlations)
            
            correlation_drop = max_corr - min_corr
            
            print(f"Correlation drop from gap 0 to max gap: {correlation_drop:.4f}")
            
            if correlation_drop > 0.4:
                print("🚨 SHARP CORRELATION DROP - likely indicates data leakage")
            elif correlation_drop > 0.2:
                print("⚠️  MODERATE CORRELATION DROP - possible leakage or very strong short-term patterns")  
            else:
                print("✅ GRADUAL CORRELATION CHANGE - consistent with real patterns")
        
        self.results['gap_sensitivity_test'] = gap_correlations
        
        return gap_correlations
    
    def _test_specific_gap_size(self, data, ticker, gap_size):
        """Test correlation with specific temporal gap"""
        
        # This is a simplified test - we'd need to modify the actual training pipeline
        # for full testing, but this gives us a quick correlation check
        
        feature_calc = FeatureCalculator(symbol=ticker)
        features_df = feature_calc.calculate_all_features(data)
        
        target_gen = PatternTargetGenerator(lookback_window=20, validation_horizons=[3, 5, 10])
        targets = target_gen.generate_all_pattern_targets(features_df, primary_horizon=5)
        
        combined_target = targets["pattern_confidence_score"]
        
        # Simple feature-target correlation with gap
        if len(features_df) > gap_size + 20:
            # Use feature values at t, target values at t+gap_size
            feature_subset = features_df["momentum_persistence_7d"].values[:-gap_size-1]
            target_subset = combined_target[gap_size+1:]
            
            valid_mask = ~(np.isnan(feature_subset) | np.isnan(target_subset))
            
            if np.sum(valid_mask) > 10:
                correlation = abs(np.corrcoef(feature_subset[valid_mask], 
                                            target_subset[valid_mask])[0,1])
                return correlation
        
        return 0.0
    
    def generate_investigation_report(self):
        """Generate comprehensive investigation report"""
        
        print("\n" + "="*80)
        print("DATA LEAKAGE INVESTIGATION REPORT") 
        print("="*80)
        
        if not self.results:
            print("❌ No investigation results available")
            return
        
        print(f"Investigation completed at: {datetime.now()}")
        
        # Target audit summary
        if 'target_audit' in self.results:
            print(f"\n--- TARGET GENERATION AUDIT ---")
            target_audit = self.results['target_audit']
            
            for target_type, audit_result in target_audit.items():
                if target_type != 'simultaneous_correlations':
                    print(f"{target_type}:")
                    if 'same_time_correlation' in audit_result:
                        corr = audit_result['same_time_correlation'] 
                        status = "🚨 SUSPICIOUS" if abs(corr) > 0.3 else "✅ OK"
                        print(f"  Same-time correlation: {corr:.4f} {status}")
                    
                    if 'issues' in audit_result:
                        for issue in audit_result['issues']:
                            print(f"  Issue: {issue}")
            
            if 'simultaneous_correlations' in target_audit:
                high_corrs = target_audit['simultaneous_correlations']
                if high_corrs:
                    print(f"\nHigh feature-target correlations detected:")
                    for feature, corr in high_corrs:
                        print(f"  {feature}: {corr:.3f}")
        
        # Random target test summary  
        if 'random_target_test' in self.results:
            print(f"\n--- RANDOM TARGET TEST ---")
            random_test = self.results['random_target_test']
            max_corr = random_test.get('max_correlation', 0)
            
            status = "🚨 FAILED" if max_corr > 0.15 else "✅ PASSED"
            print(f"Max correlation with random targets: {max_corr:.4f} {status}")
        
        # Gap sensitivity summary
        if 'gap_sensitivity_test' in self.results:
            print(f"\n--- GAP SENSITIVITY TEST ---")
            gap_results = self.results['gap_sensitivity_test']
            
            valid_gaps = [(gap, corr) for gap, corr in gap_results.items() 
                         if corr is not None]
            
            if len(valid_gaps) >= 2:
                correlations = [corr for gap, corr in sorted(valid_gaps)]
                drop = max(correlations) - min(correlations) 
                
                status = "🚨 SHARP DROP" if drop > 0.4 else "✅ GRADUAL"
                print(f"Correlation drop across gaps: {drop:.4f} {status}")
        
        print(f"\n--- RECOMMENDATIONS ---")
        
        # Generate recommendations based on findings
        high_risk_issues = []
        
        if 'target_audit' in self.results:
            for target_type, results in self.results['target_audit'].items():
                if isinstance(results, dict) and 'same_time_correlation' in results:
                    if abs(results['same_time_correlation']) > 0.3:
                        high_risk_issues.append(f"High {target_type} correlation")
        
        if 'random_target_test' in self.results:
            if self.results['random_target_test'].get('max_correlation', 0) > 0.15:
                high_risk_issues.append("High random target correlation")
        
        if high_risk_issues:
            print("🚨 HIGH RISK - Data leakage likely present:")
            for issue in high_risk_issues:
                print(f"  - {issue}")
            print("\nImmediate action required:")
            print("  1. Fix target generation logic to use only historical data")
            print("  2. Add temporal gaps between features and targets") 
            print("  3. Re-audit all feature calculations for forward-looking logic")
        else:
            print("✅ LOW RISK - No obvious data leakage detected")
            print("Continue with systematic validation and model improvement")


def main():
    """Run complete data leakage investigation"""
    
    # Initialize investigator
    investigator = DataLeakageInvestigator()
    
    # Load sample data for testing 
    print("Loading sample data for investigation...")
    
    # For testing, create minimal synthetic data
    dates = pd.date_range(start='2022-01-01', end='2024-08-01', freq='D')
    n_days = len(dates)
    
    # Simple synthetic OHLCV data for AAPL
    np.random.seed(42)
    base_price = 150
    price_changes = np.cumsum(np.random.normal(0, 0.02, n_days))
    
    synthetic_data = {
        'AAPL': pd.DataFrame({
            'open': base_price + price_changes + np.random.normal(0, 0.5, n_days),
            'high': base_price + price_changes + np.abs(np.random.normal(0, 1, n_days)),
            'low': base_price + price_changes - np.abs(np.random.normal(0, 1, n_days)), 
            'close': base_price + price_changes,
            'volume': np.random.lognormal(15, 0.5, n_days).astype(int)
        }, index=dates)
    }
    
    print(f"Created synthetic data: {n_days} days for investigation")
    
    try:
        # Run investigation phases
        print("\nStarting data leakage investigation...")
        
        # Phase 1: Target generation audit
        investigator.investigate_target_generation(synthetic_data)
        
        # Phase 2: Controlled experiments  
        investigator.run_random_target_test(synthetic_data)
        investigator.run_gap_sensitivity_test(synthetic_data)
        
        # Generate final report
        investigator.generate_investigation_report()
        
    except Exception as e:
        print(f"\n❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nInvestigation completed. Check results above for data leakage indicators.")


if __name__ == "__main__":
    main()