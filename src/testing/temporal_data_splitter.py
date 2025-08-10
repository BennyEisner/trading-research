#!/usr/bin/env python3

"""
Temporal Data Splitter
Handles proper temporal splitting of financial data with configurable gaps
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))


class TemporalDataSplitter:
    """
    Handles temporal splitting of financial data with proper gap management
    
    This class ensures proper out-of-sample testing by creating temporal splits
    with configurable gaps to prevent data leakage between training and testing periods.
    """
    
    def __init__(self, gap_months: int = 6, train_ratio: float = 0.8):
        """
        Initialize temporal data splitter
        
        Args:
            gap_months: Gap in months between training and testing periods
            train_ratio: Ratio of data to use for training (before gap)
        """
        self.gap_months = gap_months
        self.train_ratio = train_ratio
        self.gap_days = gap_months * 30  # Approximate days per month
    
    def split_ticker_data(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Split single ticker data into train/test with temporal gap
        
        Args:
            data: OHLCV DataFrame for single ticker
            ticker: Ticker symbol for logging
            
        Returns:
            Dictionary with train_data, test_data, and metadata
        """
        data_length = len(data)
        
        # Calculate split indices
        train_end_idx = int(data_length * self.train_ratio)
        test_start_idx = min(train_end_idx + self.gap_days, data_length)
        
        # Validate sufficient data
        if test_start_idx >= data_length - 30:  # Need at least 30 days for testing
            return {
                "success": False,
                "error": f"Insufficient data for {self.gap_days}-day gap",
                "data_length": data_length,
                "required_length": train_end_idx + self.gap_days + 30
            }
        
        # Create temporal splits
        train_data = data.iloc[:train_end_idx].copy()
        test_data = data.iloc[test_start_idx:].copy()
        gap_data = data.iloc[train_end_idx:test_start_idx].copy()
        
        return {
            "success": True,
            "train_data": train_data,
            "test_data": test_data,
            "gap_data": gap_data,
            "metadata": {
                "ticker": ticker,
                "total_samples": data_length,
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "gap_samples": len(gap_data),
                "gap_days": test_start_idx - train_end_idx,
                "train_period": f"{train_data.index[0]} to {train_data.index[-1]}",
                "test_period": f"{test_data.index[0]} to {test_data.index[-1]}",
                "train_ratio": len(train_data) / data_length,
                "test_ratio": len(test_data) / data_length
            }
        }
    
    def split_multi_ticker_data(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Split multiple tickers with temporal gaps
        
        Args:
            ticker_data: Dictionary {ticker: ohlcv_dataframe}
            
        Returns:
            Dictionary with split results for all tickers
        """
        results = {}
        successful_tickers = []
        failed_tickers = []
        
        for ticker, data in ticker_data.items():
            split_result = self.split_ticker_data(data, ticker)
            results[ticker] = split_result
            
            if split_result["success"]:
                successful_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        
        # Generate summary statistics
        if successful_tickers:
            total_train_samples = sum(results[t]["metadata"]["train_samples"] for t in successful_tickers)
            total_test_samples = sum(results[t]["metadata"]["test_samples"] for t in successful_tickers)
            avg_gap_days = np.mean([results[t]["metadata"]["gap_days"] for t in successful_tickers])
        else:
            total_train_samples = total_test_samples = avg_gap_days = 0
        
        summary = {
            "total_tickers": len(ticker_data),
            "successful_tickers": len(successful_tickers),
            "failed_tickers": len(failed_tickers),
            "success_rate": len(successful_tickers) / len(ticker_data) if ticker_data else 0,
            "total_train_samples": total_train_samples,
            "total_test_samples": total_test_samples,
            "average_gap_days": avg_gap_days,
            "configuration": {
                "gap_months": self.gap_months,
                "gap_days": self.gap_days,
                "train_ratio": self.train_ratio
            }
        }
        
        return {
            "splits": results,
            "successful_tickers": successful_tickers,
            "failed_tickers": failed_tickers,
            "summary": summary
        }
    
    def validate_temporal_integrity(self, split_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate temporal integrity of the split
        
        Args:
            split_result: Result from split_ticker_data
            
        Returns:
            Validation report
        """
        if not split_result["success"]:
            return {"valid": False, "error": "Split failed"}
        
        train_data = split_result["train_data"]
        test_data = split_result["test_data"]
        metadata = split_result["metadata"]
        
        # Check temporal ordering
        train_last_date = train_data.index[-1]
        test_first_date = test_data.index[0]
        actual_gap = (test_first_date - train_last_date).days
        
        # Validate gap size
        min_expected_gap = self.gap_days * 0.8  # Allow 20% tolerance for weekends/holidays
        gap_sufficient = actual_gap >= min_expected_gap
        
        # Check for overlapping dates
        train_dates = set(train_data.index)
        test_dates = set(test_data.index)
        overlap = train_dates.intersection(test_dates)
        
        validation_result = {
            "valid": gap_sufficient and len(overlap) == 0,
            "gap_analysis": {
                "expected_gap_days": self.gap_days,
                "actual_gap_days": actual_gap,
                "gap_sufficient": gap_sufficient,
                "gap_ratio": actual_gap / self.gap_days if self.gap_days > 0 else 0
            },
            "overlap_analysis": {
                "overlapping_dates": len(overlap),
                "overlapping_samples": list(overlap)[:10] if overlap else [],  # Show first 10
                "no_overlap": len(overlap) == 0
            },
            "temporal_continuity": {
                "train_last_date": train_last_date,
                "test_first_date": test_first_date,
                "proper_ordering": train_last_date < test_first_date
            }
        }
        
        return validation_result
    
    def generate_split_report(self, multi_split_result: Dict[str, Any]) -> str:
        """
        Generate comprehensive report of temporal splitting results
        
        Args:
            multi_split_result: Result from split_multi_ticker_data
            
        Returns:
            Formatted report string
        """
        summary = multi_split_result["summary"]
        successful_tickers = multi_split_result["successful_tickers"]
        failed_tickers = multi_split_result["failed_tickers"]
        
        report = f"""
Temporal Data Splitting Report
=============================

Configuration:
- Gap between train/test: {summary['configuration']['gap_months']} months ({summary['configuration']['gap_days']} days)
- Training ratio: {summary['configuration']['train_ratio']:.1%}

Summary Results:
- Total tickers processed: {summary['total_tickers']}
- Successful splits: {summary['successful_tickers']} ({summary['success_rate']:.1%})
- Failed splits: {summary['failed_tickers']}
- Total training samples: {summary['total_train_samples']:,}
- Total testing samples: {summary['total_test_samples']:,}
- Average gap: {summary['average_gap_days']:.1f} days

Successful Tickers: {', '.join(successful_tickers[:10])}{'...' if len(successful_tickers) > 10 else ''}

Failed Tickers: {', '.join(failed_tickers) if failed_tickers else 'None'}

Temporal Integrity Validation:
"""
        
        # Add detailed validation for successful tickers
        for ticker in successful_tickers[:5]:  # Show details for first 5
            split_data = multi_split_result["splits"][ticker]
            validation = self.validate_temporal_integrity(split_data)
            
            report += f"""
{ticker}:
  - Training period: {split_data['metadata']['train_period']}
  - Testing period: {split_data['metadata']['test_period']}  
  - Gap: {validation['gap_analysis']['actual_gap_days']} days
  - Integrity: {'✅ VALID' if validation['valid'] else '❌ INVALID'}
"""
        
        return report