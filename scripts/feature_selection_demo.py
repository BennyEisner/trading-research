#!/usr/bin/env python3

"""
Demonstration of comprehensive feature selection for financial time series
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.config.config import Config
from src.data.data_loader import DataLoader
from src.features.feature_selector import FinancialFeatureSelector, FinancialFeatureEngineer
from src.utils.logging_utils import setup_production_logger


def demonstrate_feature_selection():
    """Demonstrate comprehensive feature selection on real financial data"""
    
    logger = setup_production_logger({"log_file": "feature_selection_demo.log"})
    
    print("🔬 FINANCIAL FEATURE SELECTION DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    config = Config({
        "database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db"
    })
    
    # Load data
    data_loader = DataLoader(config)
    ticker = "AAPL"
    
    try:
        # Load 5 years of AAPL data
        raw_data = data_loader.load_single_ticker_data(ticker, 5)
        print(f"📊 Loaded {len(raw_data)} records for {ticker}")
        
        # Step 1: Create comprehensive features
        print("\n🏗️  Step 1: Creating Comprehensive Feature Set")
        print("-" * 50)
        
        engineer = FinancialFeatureEngineer()
        all_features = engineer.create_comprehensive_features(raw_data, ticker)
        
        print(f"Created {all_features.shape[1]} candidate features:")
        print(f"   • Data shape: {all_features.shape}")
        print(f"   • Date range: {len(raw_data)} trading days")
        print(f"   • Feature categories: Price ratios, Moving averages, Technical indicators,")
        print(f"     Volatility measures, Momentum indicators, Volume features, Regime indicators")
        
        # Show sample features
        print(f"\nSample features: {list(all_features.columns[:10])}")
        
        # Step 2: Prepare target variable
        print(f"\n🎯 Step 2: Preparing Target Variable")
        print("-" * 50)
        
        target = all_features['daily_return'].shift(-1).fillna(0).values
        feature_data = all_features.drop(columns=['daily_return'])
        
        print(f"Target: Next day's return")
        print(f"Target shape: {target.shape}")
        print(f"Target range: {target.min():.4f} to {target.max():.4f}")
        print(f"Target std: {target.std():.4f}")
        
        # Step 3: Feature selection
        print(f"\n🔍 Step 3: Ensemble Feature Selection")
        print("-" * 50)
        
        selector = FinancialFeatureSelector(target_features=12)
        selected_features = selector.fit_transform(feature_data, target)
        
        print(f"\nSelected {selected_features.shape[1]} optimal features from {feature_data.shape[1]} candidates")
        
        # Step 4: Generate feature importance report
        print(f"\n📋 Step 4: Feature Importance Report")
        print("-" * 50)
        
        report = selector.get_feature_importance_report()
        
        print("Top 15 features by ensemble voting:")
        print(report[['feature', 'ensemble_votes', 'f_test_score', 'random_forest_score', 'selected']].head(15))
        
        # Step 5: Show selected features
        print(f"\n✅ Step 5: Final Selected Features")
        print("-" * 50)
        
        print("Selected features for LSTM training:")
        for i, feature in enumerate(selected_features.columns, 1):
            votes = report[report['feature'] == feature]['ensemble_votes'].iloc[0] if len(report[report['feature'] == feature]) > 0 else 0
            print(f"{i:2d}. {feature:<25} (votes: {votes})")
        
        # Step 6: Data quality assessment
        print(f"\n📈 Step 6: Data Quality Assessment")
        print("-" * 50)
        
        # Check for missing values
        missing_pct = (selected_features.isnull().sum() / len(selected_features) * 100)
        print(f"Missing values: {missing_pct.sum():.2f}% total")
        
        # Feature correlations with target
        correlations = {}
        for col in selected_features.columns:
            corr = selected_features[col].corr(pd.Series(target[:len(selected_features)]))
            correlations[col] = abs(corr)
        
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 5 features by target correlation:")
        for feature, corr in sorted_correlations[:5]:
            print(f"  {feature:<25}: {corr:.4f}")
        
        # Step 7: Sample size calculation
        print(f"\n📊 Step 7: Training Data Assessment")
        print("-" * 50)
        
        print(f"Available sequences (30-day lookback): ~{len(selected_features) - 30:,}")
        print(f"Recommended minimum for LSTM: 10,000 sequences")
        print(f"Current ratio: {(len(selected_features) - 30) / 10000:.2f}x recommended minimum")
        
        if len(selected_features) - 30 < 5000:
            print("⚠️  WARNING: Dataset may be too small for optimal LSTM performance")
            print("   Recommendation: Use 10+ years of data or include more tickers")
        else:
            print("✅ Dataset size is adequate for LSTM training")
        
        # Step 8: Export results
        print(f"\n💾 Step 8: Exporting Results")
        print("-" * 50)
        
        # Save feature selection results
        report.to_csv('feature_selection_report.csv', index=False)
        selected_features.to_csv('selected_features.csv', index=False)
        
        print(f"Exported:")
        print(f"  • feature_selection_report.csv - Complete feature analysis")
        print(f"  • selected_features.csv - Selected features for model training")
        
        print(f"\n🎉 Feature selection demonstration complete!")
        print("=" * 60)
        
        return selected_features, report
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        logger.log(f"Feature selection demo failed: {e}")
        return None, None


def analyze_dataset_size():
    """Analyze potential dataset size with different configurations"""
    
    print("\n📏 DATASET SIZE ANALYSIS")
    print("=" * 60)
    
    configs = [
        {"years": 5, "tickers": 3, "name": "Small (Current)"},
        {"years": 10, "tickers": 5, "name": "Medium"},
        {"years": 15, "tickers": 7, "name": "Large"},
        {"years": 20, "tickers": 10, "name": "Very Large"},
    ]
    
    print(f"{'Configuration':<15} {'Years':<6} {'Tickers':<8} {'Est. Records':<12} {'Est. Sequences':<14} {'Adequate?':<10}")
    print("-" * 80)
    
    for config in configs:
        records = config["years"] * config["tickers"] * 252  # Trading days per year
        sequences = records - 30  # Accounting for lookback window
        adequate = "✅ Yes" if sequences >= 10000 else "⚠️  Small" if sequences >= 5000 else "❌ No"
        
        print(f"{config['name']:<15} {config['years']:<6} {config['tickers']:<8} "
              f"{records:<12,} {sequences:<14,} {adequate:<10}")
    
    print("-" * 80)
    print("Recommendation: Use 'Medium' or larger configuration for robust LSTM training")


if __name__ == "__main__":
    selected_features, report = demonstrate_feature_selection()
    
    if selected_features is not None:
        analyze_dataset_size()
    else:
        print("Feature selection failed - check database connection and data availability")