#!/usr/bin/env python3

"""
Test Data Pipeline Components
Tests data loading and feature generation pipeline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.data_loader import load_test_data, validate_data_format
from src.training.shared_backbone_trainer import create_shared_backbone_trainer


def test_data_loading():
    """Test data loading matches expected format"""
    
    print("Testing Data Loading Pipeline...")
    
    test_tickers = ["AAPL", "MSFT"]
    
    try:
        # Load test data using same method as production
        ticker_data = load_test_data(test_tickers, days=100)
        
        if not ticker_data:
            print("✗ No data loaded")
            return False
        
        # Validate format
        if not validate_data_format(ticker_data):
            print("✗ Data format validation failed")
            return False
        
        print(f"✓ Data loading PASSED - {len(ticker_data)} tickers loaded")
        return True
        
    except Exception as e:
        print(f"✗ Data loading FAILED: {e}")
        return False


def test_feature_generation():
    """Test that pattern features can be generated from loaded data"""
    
    print("\nTesting Feature Generation Pipeline...")
    
    try:
        # Load test data
        test_tickers = ["AAPL", "MSFT"] 
        ticker_data = load_test_data(test_tickers, days=100)
        
        if not ticker_data:
            print("✗ Failed to load test data")
            return False
        
        # Create trainer 
        trainer = create_shared_backbone_trainer(
            tickers=test_tickers,
            use_expanded_universe=False
        )
        
        # Test feature calculation
        print("Calculating portfolio features...")
        portfolio_features = trainer.pattern_engine.calculate_portfolio_features(
            ticker_data, parallel=False  # Sequential for easier debugging
        )
        
        if not portfolio_features:
            print("✗ No features generated")
            return False
        
        # Validate feature output
        for ticker, features_df in portfolio_features.items():
            if features_df is None or features_df.empty:
                print(f"✗ No features for {ticker}")
                return False
            
            print(f"  {ticker}: {features_df.shape} features generated")
        
        print(f"✓ Feature generation PASSED - {len(portfolio_features)} tickers processed")
        return True
        
    except Exception as e:
        print(f"✗ Feature generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_data_preparation():
    """Test full training data preparation pipeline"""
    
    print("\nTesting Training Data Preparation...")
    
    try:
        # Load test data
        test_tickers = ["AAPL"]  # Single ticker for faster test
        ticker_data = load_test_data(test_tickers, days=200)  # More data for sequences
        
        if not ticker_data:
            print("✗ Failed to load test data")
            return False
        
        # Create trainer
        trainer = create_shared_backbone_trainer(
            tickers=test_tickers,
            use_expanded_universe=False
        )
        
        # Prepare training data (features + patterns + sequences)
        print("Preparing training data (features + patterns + sequences)...")
        training_data = trainer.prepare_training_data(ticker_data)
        
        if not training_data:
            print("✗ No training data prepared")
            return False
        
        # Validate training data format
        total_sequences = 0
        for ticker, (X, y) in training_data.items():
            if X is None or y is None:
                print(f"✗ Invalid training data for {ticker}")
                return False
                
            print(f"  {ticker}: X={X.shape}, y={y.shape}")
            total_sequences += len(X)
        
        print(f"✓ Training data preparation PASSED")
        print(f"  - Successful tickers: {len(training_data)}")
        print(f"  - Total sequences: {total_sequences}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training data preparation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("DATA PIPELINE TESTS")
    print("=" * 60)
    
    success = True
    
    # Test data loading
    if not test_data_loading():
        success = False
    
    # Test feature generation  
    if not test_feature_generation():
        success = False
    
    # Test training data preparation
    if not test_training_data_preparation():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("ALL DATA PIPELINE TESTS PASSED ✓")
    else:
        print("SOME DATA PIPELINE TESTS FAILED ✗")
    print("=" * 60)
    
    sys.exit(0 if success else 1)