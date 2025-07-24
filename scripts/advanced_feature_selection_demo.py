#!/usr/bin/env python3

"""
Advanced Feature Selection Demonstration - showcasing all enhanced capabilities
based on financial advisor recommendations
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.config.config import Config
from src.data.data_loader import DataLoader
from src.features.feature_selector import FinancialFeatureSelector, FinancialFeatureEngineer
from src.features.lstm_feature_analysis import LSTMFeatureAnalyzer
from src.utils.logging_utils import setup_production_logger


def demonstrate_advanced_feature_selection():
    """Comprehensive demonstration of all advanced feature selection methods"""
    
    logger = setup_production_logger({"log_file": "advanced_feature_selection.log"})
    
    print("🚀 ADVANCED FINANCIAL FEATURE SELECTION")
    print("Implementing Expert Advisor Recommendations")
    print("=" * 80)
    
    # Configuration
    config = Config({
        "database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db"
    })
    
    # Load data from multiple tickers for robust analysis
    data_loader = DataLoader(config)
    tickers = ["AAPL", "MSFT", "GOOG"]
    
    all_data = []
    for ticker in tickers:
        try:
            # Load 8 years of data for comprehensive analysis
            ticker_data = data_loader.load_single_ticker_data(ticker, 8)
            ticker_data['ticker'] = ticker
            all_data.append(ticker_data)
            print(f"📊 Loaded {len(ticker_data)} records for {ticker}")
        except Exception as e:
            print(f"❌ Error loading {ticker}: {e}")
            continue
    
    if not all_data:
        print("❌ No data could be loaded")
        return None
    
    # Combine data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"📈 Combined dataset: {len(combined_data)} records from {len(all_data)} tickers")
    
    # ========================================================================
    # PHASE 1: COMPREHENSIVE FEATURE ENGINEERING
    # ========================================================================
    print(f"\n🏗️  PHASE 1: COMPREHENSIVE FEATURE ENGINEERING")
    print("-" * 60)
    
    engineer = FinancialFeatureEngineer()
    all_features = engineer.create_comprehensive_features(combined_data)
    
    print(f"✅ Created {all_features.shape[1]} candidate features")
    print(f"   • Feature categories: Price ratios, Technical indicators, Volume features")
    print(f"   • Moving averages, Volatility measures, Momentum indicators")
    print(f"   • Regime indicators, Interaction features")
    print(f"   • Data shape: {all_features.shape}")
    
    # Prepare target
    target = all_features['daily_return'].shift(-1).fillna(0).values
    feature_data = all_features.drop(columns=['daily_return'])
    
    print(f"🎯 Target variable: Next day's return")
    print(f"   • Target range: {target.min():.4f} to {target.max():.4f}")
    print(f"   • Target volatility: {target.std():.4f}")
    
    # ========================================================================
    # PHASE 2: ADVANCED FEATURE SELECTION PIPELINE
    # ========================================================================
    print(f"\n🔍 PHASE 2: ADVANCED FEATURE SELECTION PIPELINE")
    print("Implementing Multi-Stage Framework as per Expert Recommendations")
    print("-" * 60)
    
    # Initialize advanced selector
    selector = FinancialFeatureSelector(target_features=15, min_variance_threshold=0.001)
    
    # Run comprehensive selection
    selected_features = selector.fit_transform(feature_data, target)
    
    print(f"\n✅ Advanced selection completed!")
    print(f"   Selected {selected_features.shape[1]} features from {feature_data.shape[1]} candidates")
    
    # ========================================================================
    # PHASE 3: DETAILED ANALYSIS RESULTS
    # ========================================================================
    print(f"\n📋 PHASE 3: DETAILED SELECTION ANALYSIS")
    print("-" * 60)
    
    # Get comprehensive report
    report = selector.get_feature_importance_report()
    
    print("🔬 Selection Method Results:")
    print(f"   • Stationarity Filter: {len([f for f, r in selector.feature_scores_.get('stationarity', {}).items() if r.get('stationary', False)])} stationary features")
    print(f"   • VIF Filter: {len([f for f, v in selector.feature_scores_.get('vif', {}).items() if v < 10])} low-multicollinearity features")
    print(f"   • Time Lag Analysis: {len(selector.feature_scores_.get('time_lags', {}))} features tested for optimal lags")
    print(f"   • Regime Stability: {len([f for f, s in selector.feature_scores_.get('regime_stability', {}).items() if s > 0.3])} regime-stable features")
    print(f"   • Walk-Forward: {len(selector.feature_scores_.get('walk_forward', {}).get('feature_counts', {}))} features tested across time windows")
    
    print(f"\n🏆 TOP 15 SELECTED FEATURES:")
    print(f"{'Rank':<5} {'Feature Name':<30} {'Ensemble Votes':<15} {'Regime Stability':<17} {'Selected':<10}")
    print("-" * 85)
    
    selected_features_list = selected_features.columns.tolist()
    top_features = report.head(15)
    
    for idx, row in top_features.iterrows():
        feature_name = row['feature']
        votes = int(row.get('ensemble_votes', 0))
        stability = selector.feature_scores_.get('regime_stability', {}).get(feature_name, 0)
        is_selected = "✅ YES" if feature_name in selected_features_list else "❌ No"
        
        print(f"{idx+1:<5} {feature_name:<30} {votes:<15} {stability:<17.3f} {is_selected:<10}")
    
    # ========================================================================
    # PHASE 4: FEATURE QUALITY ASSESSMENT  
    # ========================================================================
    print(f"\n📊 PHASE 4: FEATURE QUALITY ASSESSMENT")
    print("-" * 60)
    
    # Correlation analysis
    print("🔗 Feature-Target Correlations:")
    correlations = {}
    for col in selected_features.columns:
        corr = np.corrcoef(selected_features[col].fillna(0), target[:len(selected_features)])[0, 1]
        correlations[col] = abs(corr)
    
    top_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:8]
    for feature, corr in top_correlations:
        print(f"   {feature:<30}: {corr:.4f}")
    
    # Time lag analysis results
    if 'time_lags' in selector.feature_scores_:
        print(f"\n⏰ Optimal Time Lags Discovered:")
        lag_results = selector.feature_scores_['time_lags']
        features_with_lags = [(f, r['best_lag']) for f, r in lag_results.items() if r['best_lag'] > 0]
        features_with_lags.sort(key=lambda x: x[1], reverse=True)
        
        for feature, lag in features_with_lags[:5]:
            print(f"   {feature:<30}: {lag} day(s) optimal lag")
    
    # Market regime analysis
    if 'regime_stability' in selector.feature_scores_:
        print(f"\n🌊 Market Regime Stability (Top Features):")
        regime_scores = selector.feature_scores_['regime_stability']
        stable_features = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for feature, stability in stable_features:
            status = "Stable" if stability > 0.5 else "Moderate" if stability > 0.3 else "Unstable"
            print(f"   {feature:<30}: {stability:.3f} ({status})")
    
    # ========================================================================
    # PHASE 5: CROSS-VALIDATION AND ROBUSTNESS TESTING
    # ========================================================================
    print(f"\n🧪 PHASE 5: ROBUSTNESS TESTING")
    print("-" * 60)
    
    # Time series CV scores
    if 'cv_scores' in selector.feature_scores_:
        cv_scores = selector.feature_scores_['cv_scores']
        selected_cv_scores = {f: cv_scores.get(f, 0) for f in selected_features_list}
        avg_cv_score = np.mean(list(selected_cv_scores.values()))
        
        print(f"📈 Time Series Cross-Validation:")
        print(f"   Average CV Score (selected features): {avg_cv_score:.3f}")
        print(f"   CV Score Range: {min(selected_cv_scores.values()):.3f} - {max(selected_cv_scores.values()):.3f}")
    
    # Walk-forward consistency
    if 'walk_forward' in selector.feature_scores_:
        wf_results = selector.feature_scores_['walk_forward']
        if wf_results.get('num_windows', 0) > 0:
            print(f"\n🚶 Walk-Forward Analysis:")
            print(f"   Analysis windows: {wf_results['num_windows']}")
            
            # Calculate consistency of selected features
            feature_counts = wf_results.get('feature_counts', {})
            selected_consistency = {f: feature_counts.get(f, 0) / wf_results['num_windows'] 
                                  for f in selected_features_list if f in feature_counts}
            
            if selected_consistency:
                avg_consistency = np.mean(list(selected_consistency.values()))
                print(f"   Average consistency (selected features): {avg_consistency:.1%}")
    
    # ========================================================================
    # PHASE 6: PERFORMANCE COMPARISON
    # ========================================================================
    print(f"\n⚖️  PHASE 6: PERFORMANCE COMPARISON")
    print("-" * 60)
    
    # Compare different selection approaches
    comparison_results = {}
    
    # Test different feature sets
    feature_sets = {
        'All Features': feature_data.fillna(0),
        'Selected Features': selected_features.fillna(0),
        'Top Correlation': feature_data[list(dict(top_correlations[:10]).keys())].fillna(0),
    }
    
    for set_name, feature_set in feature_sets.items():
        try:
            # Simple linear model for comparison
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import mean_absolute_error
            
            # Time series CV
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(feature_set):
                X_train, X_val = feature_set.iloc[train_idx], feature_set.iloc[val_idx]
                y_train, y_val = target[train_idx], target[val_idx]
                
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                
                # Directional accuracy
                dir_acc = np.mean(np.sign(y_val) == np.sign(pred))
                cv_scores.append(dir_acc)
            
            comparison_results[set_name] = {
                'features': feature_set.shape[1],
                'avg_directional_accuracy': np.mean(cv_scores),
                'std_directional_accuracy': np.std(cv_scores)
            }
            
        except Exception as e:
            print(f"   Comparison failed for {set_name}: {e}")
    
    print("📊 Feature Set Performance Comparison:")
    for set_name, results in comparison_results.items():
        print(f"   {set_name:<20}: {results['features']:>3} features, "
              f"Dir Acc: {results['avg_directional_accuracy']:.1%} "
              f"(±{results['std_directional_accuracy']:.1%})")
    
    # ========================================================================
    # PHASE 7: EXPORT RESULTS
    # ========================================================================
    print(f"\n💾 PHASE 7: EXPORTING RESULTS")
    print("-" * 60)
    
    # Create comprehensive export
    export_data = {
        'selected_features': selected_features_list,
        'feature_report': report.to_dict('records'),
        'selection_summary': {
            'total_candidates': feature_data.shape[1],
            'final_selected': len(selected_features_list),
            'selection_methods_used': list(selector.feature_scores_.keys()),
            'data_size': len(combined_data),
            'tickers_analyzed': tickers
        }
    }
    
    # Save files
    selected_features.to_csv('advanced_selected_features.csv', index=False)
    report.to_csv('comprehensive_feature_analysis_report.csv', index=False)
    
    # Save detailed results
    import json
    with open('feature_selection_metadata.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_safe_export = {}
        for key, value in export_data.items():
            if key == 'feature_report':
                json_safe_export[key] = value  # Already converted by pandas
            elif isinstance(value, dict):
                json_safe_export[key] = {k: (v if isinstance(v, (str, int, float, bool, list)) 
                                            else str(v)) for k, v in value.items()}
            else:
                json_safe_export[key] = value
        
        json.dump(json_safe_export, f, indent=2)
    
    print("✅ Exported files:")
    print("   • advanced_selected_features.csv - Final selected features")
    print("   • comprehensive_feature_analysis_report.csv - Detailed analysis")  
    print("   • feature_selection_metadata.json - Selection metadata")
    
    # ========================================================================
    # SUMMARY AND RECOMMENDATIONS
    # ========================================================================
    print(f"\n🎯 SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    
    improvement_estimate = 0
    if comparison_results:
        selected_performance = comparison_results.get('Selected Features', {}).get('avg_directional_accuracy', 0)
        all_features_performance = comparison_results.get('All Features', {}).get('avg_directional_accuracy', 0)
        
        if all_features_performance > 0:
            improvement_estimate = ((selected_performance - all_features_performance) / all_features_performance) * 100
    
    print(f"📈 PERFORMANCE IMPROVEMENTS:")
    print(f"   • Feature reduction: {feature_data.shape[1]} → {len(selected_features_list)} features ({(1 - len(selected_features_list)/feature_data.shape[1])*100:.0f}% reduction)")
    if improvement_estimate != 0:
        print(f"   • Performance improvement: {improvement_estimate:+.1f}% directional accuracy")
    print(f"   • Multicollinearity reduced via VIF analysis")
    print(f"   • Non-stationary features removed")
    print(f"   • Optimal time lags identified")
    print(f"   • Market regime stability validated")
    
    print(f"\n🔬 ADVANCED METHODS IMPLEMENTED:")
    print("   ✅ Stationarity testing (ADF test)")
    print("   ✅ VIF analysis for multicollinearity")
    print("   ✅ Time lag optimization") 
    print("   ✅ Market regime stability testing")
    print("   ✅ Time series cross-validation")
    print("   ✅ Walk-forward validation")
    print("   ✅ Ensemble voting across 6 methods")
    
    print(f"\n🚀 NEXT STEPS:")
    print("   1. Use selected features for LSTM model training")
    print("   2. Consider LSTM-specific feature analysis (permutation importance)")
    print("   3. Test on longer time horizons (weekly predictions)")
    print("   4. Validate across different market conditions")
    print("   5. Monitor feature stability in live trading")
    
    print("=" * 80)
    print("🎉 Advanced Feature Selection Complete!")
    
    return selected_features, report, export_data


if __name__ == "__main__":
    try:
        selected_features, report, metadata = demonstrate_advanced_feature_selection()
        
        if selected_features is not None:
            print(f"\n✅ SUCCESS: Advanced feature selection completed successfully")
            print(f"   Selected {selected_features.shape[1]} optimal features for financial prediction")
        else:
            print(f"\n❌ FAILED: Feature selection could not be completed")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()