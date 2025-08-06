#!/usr/bin/env python3

"""
LSTM Pattern Detection Test Runner
Tests the complete LSTM pattern detection pipeline end-to-end with real market data

This test runner validates that the LSTM pattern detection system works end-to-end
before integration with the ensemble strategy system.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import traceback
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the shared backbone trainer
from src.training.shared_backbone_trainer import create_shared_backbone_trainer


class LSTMPatternDetectionTester:
    """
    Comprehensive tester for LSTM pattern detection pipeline
    Tests: Data loading → Feature calculation → Pattern targets → Training → Validation
    """
    
    def __init__(self, test_tickers: list = None, test_period_days: int = 730):
        """
        Initialize LSTM pattern detection tester
        
        Args:
            test_tickers: List of tickers to test (default: subset of MAG7)
            test_period_days: Number of days of historical data to use
        """
        self.test_tickers = test_tickers or ["AAPL", "MSFT", "GOOG"]  # Start small
        self.test_period_days = test_period_days
        self.trainer = None
        self.ticker_data = {}
        self.training_data = {}
        self.training_results = {}
        
        print(f"LSTM Pattern Detection Tester Initialized")
        print(f"   Test Tickers: {self.test_tickers}")
        print(f"   Test Period: {test_period_days} days")
        print(f"   Purpose: Validate LSTM pattern detection pipeline works end-to-end")
        
    def run_full_test(self) -> Dict[str, Any]:
        """
        Run complete LSTM pattern detection test pipeline
        
        Returns:
            Test results with success/failure status and detailed metrics
        """
        results = {
            "overall_success": False,
            "test_stages": {},
            "error_details": {},
            "performance_metrics": {}
        }
        
        try:
            print("\n" + "="*80)
            print("LSTM PATTERN DETECTION PIPELINE TEST")
            print("="*80)
            
            # Stage 1: Initialize trainer
            print("\nSTAGE 1: Initialize LSTM Trainer")
            stage1_success = self._test_trainer_initialization()
            results["test_stages"]["trainer_initialization"] = stage1_success
            
            if not stage1_success:
                results["error_details"]["trainer_initialization"] = "Failed to initialize trainer"
                return results
            
            # Stage 2: Load market data
            print("\nSTAGE 2: Load Market Data")
            stage2_success = self._test_data_loading()
            results["test_stages"]["data_loading"] = stage2_success
            
            if not stage2_success:
                results["error_details"]["data_loading"] = "Failed to load market data"
                return results
            
            # Stage 3: Prepare training data (features + patterns + sequences)
            print("\nSTAGE 3: Prepare Training Data (Features + Patterns + Sequences)")
            stage3_success = self._test_training_data_preparation()
            results["test_stages"]["training_data_preparation"] = stage3_success
            
            if not stage3_success:
                results["error_details"]["training_data_preparation"] = "Failed to prepare training data"
                return results
            
            # Stage 4: Train LSTM model
            print("\nSTAGE 4: Train LSTM Model")
            stage4_success = self._test_lstm_training()
            results["test_stages"]["lstm_training"] = stage4_success
            
            if not stage4_success:
                results["error_details"]["lstm_training"] = "Failed to train LSTM model"
                return results
            
            # Stage 5: Validate model performance
            print("\nSTAGE 5: Validate Model Performance")
            stage5_success, performance_metrics = self._test_model_validation()
            results["test_stages"]["model_validation"] = stage5_success
            results["performance_metrics"] = performance_metrics
            
            if not stage5_success:
                results["error_details"]["model_validation"] = "Failed model validation"
                return results
            
            # All stages passed
            results["overall_success"] = True
            print("\n" + "="*80)
            print("ALL STAGES PASSED - LSTM PATTERN DETECTION PIPELINE WORKS!")
            print("="*80)
            
        except Exception as e:
            results["error_details"]["unexpected_error"] = str(e)
            results["traceback"] = traceback.format_exc()
            print(f"\nUNEXPECTED ERROR: {e}")
            print(traceback.format_exc())
        
        return results
    
    def _test_trainer_initialization(self) -> bool:
        """Test trainer initialization"""
        try:
            self.trainer = create_shared_backbone_trainer(
                tickers=self.test_tickers, 
                use_expanded_universe=False  # Use small test set
            )
            
            print(f"   PASS: Trainer initialized with {len(self.trainer.tickers)} tickers")
            print(f"   - Config loaded: {self.trainer.config is not None}")
            print(f"   - Pattern engine initialized: {self.trainer.pattern_engine is not None}")
            print(f"   - LSTM builder initialized: {self.trainer.lstm_builder is not None}")
            
            return True
            
        except Exception as e:
            print(f"   FAIL: Trainer initialization failed - {e}")
            return False
    
    def _test_data_loading(self) -> bool:
        """Test market data loading"""
        try:
            print(f"   Loading {self.test_period_days} days of data for {len(self.test_tickers)} tickers...")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.test_period_days)
            
            for ticker in self.test_tickers:
                try:
                    # Load data using yfinance (same as trainer expects)
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if data.empty:
                        print(f"   FAIL: No data retrieved for {ticker}")
                        return False
                    
                    # Convert to expected format (lowercase columns)
                    data.columns = data.columns.str.lower()
                    data = data.reset_index()
                    data.columns = [col.lower() for col in data.columns]
                    
                    self.ticker_data[ticker] = data
                    print(f"   - {ticker}: {len(data)} records from {data['date'].min()} to {data['date'].max()}")
                    
                except Exception as e:
                    print(f"   FAIL: Error loading data for {ticker} - {e}")
                    return False
            
            print(f"   PASS: Successfully loaded data for all {len(self.test_tickers)} tickers")
            return True
            
        except Exception as e:
            print(f"   FAIL: Data loading failed - {e}")
            return False
    
    def _test_training_data_preparation(self) -> bool:
        """Test training data preparation (features + patterns + sequences)"""
        try:
            print("   Preparing training data (features + pattern targets + sequences)...")
            
            # This calls the full pipeline:
            # 1. Calculate 17 pattern features via MultiTickerPatternEngine
            # 2. Generate pattern targets via PatternTargetGenerator  
            # 3. Create sequences with proper overlapping
            # 4. Validate sequences
            self.training_data = self.trainer.prepare_training_data(self.ticker_data)
            
            if not self.training_data:
                print("   FAIL: No training data prepared")
                return False
            
            # Analyze prepared data
            total_sequences = sum(len(X) for X, y in self.training_data.values())
            
            print(f"   PASS: Training data prepared successfully")
            print(f"   - Successful tickers: {len(self.training_data)}")
            print(f"   - Total sequences: {total_sequences}")
            
            # Validate data quality
            for ticker, (X, y) in self.training_data.items():
                print(f"   - {ticker}: X shape={X.shape}, y shape={y.shape}, y range=[{y.min():.3f}, {y.max():.3f}]")
                
                # Check for issues
                if np.isnan(X).any():
                    print(f"   WARNING: NaN values found in X for {ticker}")
                if np.isnan(y).any():
                    print(f"   WARNING: NaN values found in y for {ticker}")
                if len(np.unique(y)) < 2:
                    print(f"   WARNING: y values not diverse for {ticker}")
            
            return True
            
        except Exception as e:
            print(f"   FAIL: Training data preparation failed - {e}")
            traceback.print_exc()
            return False
    
    def _test_lstm_training(self) -> bool:
        """Test LSTM model training"""
        try:
            print("   Training LSTM model (this may take a few minutes)...")
            
            # Train with small epoch count for testing
            self.training_results = self.trainer.train_shared_backbone(
                training_data=self.training_data,
                validation_split=0.2,
                epochs=10  # Small for testing
            )
            
            if "model" not in self.training_results:
                print("   FAIL: No model in training results")
                return False
            
            model = self.training_results["model"]
            history = self.training_results.get("history", {})
            
            print(f"   PASS: LSTM training completed")
            print(f"   - Model parameters: ~{model.count_params():,}")
            print(f"   - Training epochs: {len(history.get('loss', []))}")
            
            # Check training stability
            if history.get("loss"):
                final_loss = history["loss"][-1]
                initial_loss = history["loss"][0]
                print(f"   - Training loss: {initial_loss:.4f} → {final_loss:.4f}")
                
                if np.isnan(final_loss):
                    print("   WARNING: Training resulted in NaN loss")
                    return False
                    
                if final_loss > initial_loss * 2:
                    print("   WARNING: Training loss increased significantly")
            
            return True
            
        except Exception as e:
            print(f"   FAIL: LSTM training failed - {e}")
            traceback.print_exc()
            return False
    
    def _test_model_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Test model validation and performance"""
        try:
            print("   Validating model performance...")
            
            model = self.training_results["model"]
            performance_metrics = {}
            
            # Test predictions on training data
            validation_results = self.trainer.validate_cross_ticker_performance(
                self.training_data, model
            )
            
            if not validation_results:
                print("   FAIL: No validation results")
                return False, {}
            
            # Extract key metrics
            for ticker, results in validation_results.items():
                if "pattern_detection_accuracy" in results:
                    accuracy = results["pattern_detection_accuracy"]
                    performance_metrics[f"{ticker}_accuracy"] = accuracy
                    print(f"   - {ticker} pattern detection accuracy: {accuracy:.3f}")
            
            # Overall metrics
            if performance_metrics:
                avg_accuracy = np.mean(list(performance_metrics.values()))
                performance_metrics["average_accuracy"] = avg_accuracy
                print(f"   - Average pattern detection accuracy: {avg_accuracy:.3f}")
                
                # Success criteria
                if avg_accuracy > 0.45:  # Lower threshold for initial test
                    print("   PASS: Model validation successful")
                    return True, performance_metrics
                else:
                    print(f"   FAIL: Average accuracy {avg_accuracy:.3f} below threshold 0.45")
                    return False, performance_metrics
            else:
                print("   FAIL: No performance metrics calculated")
                return False, {}
            
        except Exception as e:
            print(f"   FAIL: Model validation failed - {e}")
            traceback.print_exc()
            return False, {}
    
    def generate_test_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("LSTM PATTERN DETECTION TEST REPORT")
        print("="*80)
        
        print(f"\nOVERALL RESULT: {'PASS' if results['overall_success'] else 'FAIL'}")
        
        print(f"\nTEST STAGES:")
        for stage, success in results["test_stages"].items():
            status = "PASS" if success else "FAIL"
            print(f"   {stage}: {status}")
        
        if results["performance_metrics"]:
            print(f"\nPERFORMANCE METRICS:")
            for metric, value in results["performance_metrics"].items():
                print(f"   {metric}: {value:.4f}")
        
        if results["error_details"]:
            print(f"\nERROR DETAILS:")
            for stage, error in results["error_details"].items():
                print(f"   {stage}: {error}")
        
        if "traceback" in results:
            print(f"\nFULL TRACEBACK:")
            print(results["traceback"])
        
        print("\n" + "="*80)


def main():
    """Run LSTM pattern detection test"""
    
    print("LSTM Pattern Detection Pipeline Test")
    print("This will test the complete LSTM pipeline with real market data")
    print("Expected runtime: 5-10 minutes")
    
    # Create tester with small set for initial validation
    tester = LSTMPatternDetectionTester(
        test_tickers=["AAPL", "MSFT", "GOOG"],  # Start with 3 stocks
        test_period_days=500  # ~2 years of data
    )
    
    # Run full test
    results = tester.run_full_test()
    
    # Generate report
    tester.generate_test_report(results)
    
    # Return status for scripting
    return 0 if results["overall_success"] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)