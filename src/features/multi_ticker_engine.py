#!/usr/bin/env python3

"""
Multi-Ticker Pattern Feature Engine
Handles batch processing of pattern features for multiple tickers with cross-asset relationships
"""

import warnings
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd

from .pattern_feature_calculator import FeatureCalculator
from .utils.data_validation import DataValidator
from ..validation.pipeline_validator import PipelineValidator

warnings.filterwarnings("ignore")


class MultiTickerPatternEngine:
    """
    Efficient batch processing of pattern features for multiple tickers
    Handles cross-asset relationships and parallel processing
    """
    
    def __init__(self, 
                 tickers: List[str],
                 market_data: Optional[pd.DataFrame] = None,
                 sector_data: Optional[Dict[str, pd.DataFrame]] = None,
                 vix_data: Optional[pd.DataFrame] = None,
                 max_workers: int = 4):
        """
        Initialize multi-ticker pattern engine
        
        Args:
            tickers: List of ticker symbols to process
            market_data: Market data (e.g., SPY) for beta calculations
            sector_data: Dictionary of sector ETF data {ticker: sector_etf_data}
            vix_data: VIX term structure data
            max_workers: Maximum number of parallel workers
        """
        self.tickers = tickers
        self.market_data = market_data
        self.sector_data = sector_data or {}
        self.vix_data = vix_data
        self.max_workers = max_workers
        
        # Validation infrastructure
        self.data_validator = DataValidator()
        self.pipeline_validator = PipelineValidator()
        
        # Thread-safe calculators
        self._calculator_cache = {}
        self._cache_lock = threading.Lock()
        
        self.processing_results = {}
    
    def create_ticker_calculators(self) -> Dict[str, FeatureCalculator]:
        """Create feature calculators for each ticker with proper external data"""
        calculators = {}
        
        for ticker in self.tickers:
            # Get sector data for this ticker if available
            ticker_sector_data = self.sector_data.get(ticker)
            
            # Create calculator with external data dependencies
            calculators[ticker] = FeatureCalculator(
                symbol=ticker,
                market_data=self.market_data,
                sector_data=ticker_sector_data,
                vix_data=self.vix_data
            )
        
        return calculators
    
    def calculate_portfolio_features(self, 
                                   ticker_data: Dict[str, pd.DataFrame],
                                   parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Calculate pattern features for all tickers in portfolio
        
        Args:
            ticker_data: Dictionary {ticker: ohlcv_dataframe}
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary {ticker: features_dataframe}
        """
        
        print(f"Starting feature calculation for {len(self.tickers)} tickers")
        
        # Validate input data
        validation_results = self._validate_portfolio_data(ticker_data)
        
        if parallel and len(self.tickers) > 1:
            return self._calculate_features_parallel(ticker_data)
        else:
            return self._calculate_features_sequential(ticker_data)
    
    def _validate_portfolio_data(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Validate data quality for all tickers"""
        
        validation_results = {}
        
        for ticker in self.tickers:
            if ticker not in ticker_data:
                validation_results[ticker] = {
                    'valid': False,
                    'error': 'Missing data for ticker'
                }
                continue
            
            # Use existing validation infrastructure
            data_validation = self.data_validator.validate_ohlcv_data(ticker_data[ticker], ticker)
            validation_results[ticker] = data_validation
            
            if data_validation['data_quality_score'] < 0.5:
                print(f"Warning: Poor data quality for {ticker}: {data_validation['data_quality_score']:.2f}")
        
        return validation_results
    
    def _calculate_features_parallel(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate features using parallel processing"""
        
        results = {}
        calculators = self.create_ticker_calculators()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit jobs
            future_to_ticker = {}
            for ticker in self.tickers:
                if ticker in ticker_data:
                    future = executor.submit(
                        self._calculate_single_ticker_features,
                        ticker,
                        calculators[ticker],
                        ticker_data[ticker]
                    )
                    future_to_ticker[future] = ticker
            
            # Collect results
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker_features = future.result()
                    results[ticker] = ticker_features
                    print(f"Completed feature calculation for {ticker}")
                except Exception as e:
                    print(f"Error calculating features for {ticker}: {e}")
                    results[ticker] = None
        
        return results
    
    def _calculate_features_sequential(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate features sequentially"""
        
        results = {}
        calculators = self.create_ticker_calculators()
        
        for ticker in self.tickers:
            if ticker in ticker_data:
                try:
                    ticker_features = self._calculate_single_ticker_features(
                        ticker, calculators[ticker], ticker_data[ticker]
                    )
                    results[ticker] = ticker_features
                    print(f"Completed feature calculation for {ticker}")
                except Exception as e:
                    print(f"Error calculating features for {ticker}: {e}")
                    results[ticker] = None
        
        return results
    
    def _calculate_single_ticker_features(self, 
                                        ticker: str,
                                        calculator: FeatureCalculator,
                                        data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for a single ticker"""
        
        # Calculate features using the completed FeatureCalculator
        features = calculator.calculate_all_features(data)
        
        # Validate feature quality
        feature_columns = calculator.get_feature_names()
        is_valid, issues = self.pipeline_validator.validate_feature_data(features, feature_columns)
        
        if not is_valid:
            print(f"Feature validation issues for {ticker}: {issues}")
        
        return features
    
    def create_sequence_features(self, 
                               portfolio_features: Dict[str, pd.DataFrame],
                               sequence_length: int = 20,
                               stride: int = 5) -> Dict[str, np.ndarray]:
        """
        Convert portfolio features to LSTM sequences with swing trading parameters
        
        Args:
            portfolio_features: Dictionary {ticker: features_dataframe}
            sequence_length: Length of sequences for LSTM (default 20 for swing trading)
            stride: Step size between sequences (default 5 for overlapping sequences)
            
        Returns:
            Dictionary {ticker: sequences_array} with shape (n_sequences, sequence_length, n_features)
        """
        
        sequence_results = {}
        
        for ticker, features in portfolio_features.items():
            if features is not None and len(features) > sequence_length:
                sequences = self._create_ticker_sequences(features, sequence_length, stride)
                sequence_results[ticker] = sequences
                print(f"Created {len(sequences)} swing trading sequences for {ticker} (stride={stride})")
            else:
                print(f"Insufficient data for sequences for {ticker}")
                sequence_results[ticker] = None
        
        return sequence_results
    
    def _create_ticker_sequences(self, features: pd.DataFrame, sequence_length: int, stride: int = 5) -> np.ndarray:
        """Create LSTM sequences for a single ticker with overlapping stride"""
        
        # Get numeric features only (exclude date columns if present)
        numeric_features = features.select_dtypes(include=[np.number])
        
        n_samples, n_features = numeric_features.shape
        
        # Calculate number of sequences with stride
        n_sequences = max(0, (n_samples - sequence_length) // stride + 1)
        
        if n_sequences <= 0:
            return np.array([])
        
        sequences = np.zeros((n_sequences, sequence_length, n_features))
        
        # Create overlapping sequences using stride
        for i in range(n_sequences):
            start_idx = i * stride
            end_idx = start_idx + sequence_length
            sequences[i] = numeric_features.iloc[start_idx:end_idx].values
        
        return sequences
    
    def validate_cross_ticker_consistency(self, 
                                        portfolio_features: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Validate consistency across tickers (aligned dates, similar feature distributions)
        """
        
        consistency_results = {
            'date_alignment': {},
            'feature_distributions': {},
            'missing_data_patterns': {},
            'overall_consistency': True
        }
        
        # Check date alignment
        date_ranges = {}
        for ticker, features in portfolio_features.items():
            if features is not None:
                date_ranges[ticker] = {
                    'start': features.index.min(),
                    'end': features.index.max(),
                    'count': len(features)
                }
        
        consistency_results['date_alignment'] = date_ranges
        
        # Check feature distribution consistency
        if len(portfolio_features) > 1:
            feature_stats = {}
            tickers_with_data = [t for t, f in portfolio_features.items() if f is not None]
            
            if len(tickers_with_data) > 1:
                # Get common feature names
                common_features = set(portfolio_features[tickers_with_data[0]].columns)
                for ticker in tickers_with_data[1:]:
                    common_features &= set(portfolio_features[ticker].columns)
                
                # Check distribution consistency for common features
                for feature in common_features:
                    feature_stats[feature] = {}
                    for ticker in tickers_with_data:
                        data = portfolio_features[ticker][feature].dropna()
                        if len(data) > 0:
                            feature_stats[feature][ticker] = {
                                'mean': data.mean(),
                                'std': data.std(),
                                'skew': data.skew(),
                                'count': len(data)
                            }
                
                consistency_results['feature_distributions'] = feature_stats
        
        return consistency_results
    
    def get_portfolio_summary(self, portfolio_features: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Get comprehensive summary of portfolio feature calculation"""
        
        summary = {
            'total_tickers': len(self.tickers),
            'successful_tickers': len([t for t, f in portfolio_features.items() if f is not None]),
            'failed_tickers': [t for t, f in portfolio_features.items() if f is None],
            'feature_counts': {},
            'date_ranges': {},
            'data_quality_summary': {}
        }
        
        for ticker, features in portfolio_features.items():
            if features is not None:
                summary['feature_counts'][ticker] = len(features.columns)
                summary['date_ranges'][ticker] = {
                    'start': str(features.index.min()),
                    'end': str(features.index.max()),
                    'count': len(features)
                }
                
                # Basic data quality metrics
                nan_counts = features.isna().sum()
                summary['data_quality_summary'][ticker] = {
                    'total_nan_values': nan_counts.sum(),
                    'columns_with_nan': (nan_counts > 0).sum(),
                    'max_nan_percentage': (nan_counts / len(features) * 100).max()
                }
        
        return summary


def create_multi_ticker_engine(tickers: List[str], 
                             market_data: Optional[pd.DataFrame] = None,
                             sector_data: Optional[Dict[str, pd.DataFrame]] = None,
                             vix_data: Optional[pd.DataFrame] = None,
                             max_workers: int = 4) -> MultiTickerPatternEngine:
    """Convenience function to create multi-ticker pattern engine"""
    
    return MultiTickerPatternEngine(
        tickers=tickers,
        market_data=market_data,
        sector_data=sector_data,
        vix_data=vix_data,
        max_workers=max_workers
    )


if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Create synthetic test data
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    
    ticker_data = {}
    for ticker in tickers:
        # Synthetic OHLCV data
        base_price = 100 + np.random.randn() * 20
        prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.02)
        
        ticker_data[ticker] = pd.DataFrame({
            'open': prices + np.random.randn(len(dates)) * 0.01,
            'high': prices + np.abs(np.random.randn(len(dates)) * 0.015),
            'low': prices - np.abs(np.random.randn(len(dates)) * 0.015), 
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
    
    # Create engine and calculate features
    engine = create_multi_ticker_engine(tickers)
    portfolio_features = engine.calculate_portfolio_features(ticker_data, parallel=True)
    
    # Get summary
    summary = engine.get_portfolio_summary(portfolio_features)
    print(f"Portfolio summary: {summary}")
    
    # Create sequences
    sequences = engine.create_sequence_features(portfolio_features, sequence_length=30)
    print(f"Created sequences for {len(sequences)} tickers")