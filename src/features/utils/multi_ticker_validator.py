#!/usr/bin/env python3

"""
Multi-Ticker Data Validation
Enhanced validation for multi-ticker pattern feature processing
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from .data_validation import DataValidator, clean_financial_data
from ...validation.pipeline_validator import PipelineValidator

warnings.filterwarnings("ignore")


class MultiTickerValidator:
    """
    Enhanced validator for multi-ticker pattern feature processing
    Extends existing validation infrastructure with cross-ticker consistency checks
    """
    
    def __init__(self, 
                 base_validator: Optional[DataValidator] = None,
                 pipeline_validator: Optional[PipelineValidator] = None,
                 min_overlap_ratio: float = 0.8,
                 max_date_gap_days: int = 7):
        """
        Initialize multi-ticker validator
        
        Args:
            base_validator: Base data validator for individual tickers
            pipeline_validator: Pipeline validator for feature quality
            min_overlap_ratio: Minimum required date overlap between tickers
            max_date_gap_days: Maximum allowed gap in trading days
        """
        
        self.base_validator = base_validator or DataValidator()
        self.pipeline_validator = pipeline_validator or PipelineValidator()
        self.min_overlap_ratio = min_overlap_ratio
        self.max_date_gap_days = max_date_gap_days
        
        self.validation_results = {}
    
    def validate_portfolio_data(self, 
                              ticker_data: Dict[str, pd.DataFrame],
                              required_tickers: Optional[List[str]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive validation of multi-ticker portfolio data
        
        Args:
            ticker_data: Dictionary {ticker: ohlcv_dataframe}
            required_tickers: List of tickers that must be present
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        
        print(f"Validating portfolio data for {len(ticker_data)} tickers")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_tickers': len(ticker_data),
            'individual_validations': {},
            'cross_ticker_validation': {},
            'issues': [],
            'warnings': [],
            'overall_valid': True
        }
        
        # Step 1: Individual ticker validation
        individual_results = self._validate_individual_tickers(ticker_data)
        validation_results['individual_validations'] = individual_results
        
        # Count failed validations
        failed_tickers = [ticker for ticker, result in individual_results.items() 
                         if not result.get('is_valid', False)]
        
        if failed_tickers:
            validation_results['issues'].append(f"Failed individual validation: {failed_tickers}")
        
        # Step 2: Check required tickers
        if required_tickers:
            missing_tickers = set(required_tickers) - set(ticker_data.keys())
            if missing_tickers:
                validation_results['issues'].append(f"Missing required tickers: {list(missing_tickers)}")
                validation_results['overall_valid'] = False
        
        # Step 3: Cross-ticker consistency validation
        if len(ticker_data) > 1:
            cross_ticker_results = self._validate_cross_ticker_consistency(ticker_data)
            validation_results['cross_ticker_validation'] = cross_ticker_results
            
            if not cross_ticker_results.get('consistent', True):
                validation_results['issues'].extend(cross_ticker_results.get('issues', []))
        
        # Step 4: External data validation
        external_data_results = self._validate_external_data_alignment(ticker_data)
        validation_results['external_data_validation'] = external_data_results
        
        # Update overall validity
        validation_results['overall_valid'] = (
            len(validation_results['issues']) == 0 and
            len(failed_tickers) / len(ticker_data) <= 0.2  # Allow up to 20% failures
        )
        
        self.validation_results = validation_results
        
        return validation_results['overall_valid'], validation_results
    
    def _validate_individual_tickers(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Validate each ticker individually using existing infrastructure"""
        
        individual_results = {}
        
        for ticker, data in ticker_data.items():
            try:
                # Use existing data validator
                validation = self.base_validator.validate_ohlcv_data(data, ticker)
                
                # Add additional pattern-specific checks
                pattern_checks = self._validate_pattern_requirements(data, ticker)
                
                # Combine results
                individual_results[ticker] = {
                    **validation,
                    'pattern_validation': pattern_checks,
                    'is_valid': validation.get('data_quality_score', 0) > 0.5 and pattern_checks['pattern_ready']
                }
                
            except Exception as e:
                individual_results[ticker] = {
                    'is_valid': False,
                    'error': str(e),
                    'data_quality_score': 0.0
                }
        
        return individual_results
    
    def _validate_pattern_requirements(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Validate specific requirements for pattern feature calculation"""
        
        pattern_validation = {
            'pattern_ready': True,
            'issues': [],
            'warnings': []
        }
        
        # Check minimum data length for pattern calculations
        min_required_days = 100  # Need sufficient history for rolling calculations
        if len(data) < min_required_days:
            pattern_validation['issues'].append(f"Insufficient data length: {len(data)} < {min_required_days}")
            pattern_validation['pattern_ready'] = False
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            pattern_validation['issues'].append(f"Missing required columns: {missing_cols}")
            pattern_validation['pattern_ready'] = False
        
        # Check data continuity for temporal features
        if 'date' in data.columns or isinstance(data.index, pd.DatetimeIndex):
            dates = pd.to_datetime(data.index if isinstance(data.index, pd.DatetimeIndex) else data['date'])
            date_gaps = dates.to_series().diff()
            
            # Check for excessive gaps that would affect temporal calculations
            if date_gaps.notna().any():
                median_gap = date_gaps.median()
                large_gaps = (date_gaps > median_gap * 5).sum()  # More than 5x normal gap
                
                if large_gaps > len(data) * 0.05:  # More than 5% of data has large gaps
                    pattern_validation['warnings'].append(f"Many large date gaps: {large_gaps} instances")
        
        # Check for extreme values that could affect pattern calculations
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    # Check for extreme outliers (beyond 5 standard deviations)
                    z_scores = np.abs(stats.zscore(col_data))
                    extreme_outliers = (z_scores > 5).sum()
                    
                    if extreme_outliers > len(col_data) * 0.01:  # More than 1% extreme outliers
                        pattern_validation['warnings'].append(f"Many extreme outliers in {col}: {extreme_outliers}")
        
        return pattern_validation
    
    def _validate_cross_ticker_consistency(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate consistency across tickers"""
        
        cross_validation = {
            'consistent': True,
            'issues': [],
            'warnings': [],
            'date_overlap_analysis': {},
            'distribution_analysis': {}
        }
        
        # Get date ranges for each ticker
        ticker_date_ranges = {}
        for ticker, data in ticker_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                dates = data.index
            elif 'date' in data.columns:
                dates = pd.to_datetime(data['date'])
            else:
                continue
            
            ticker_date_ranges[ticker] = {
                'start': dates.min(),
                'end': dates.max(),
                'count': len(dates),
                'dates': set(dates.date)
            }
        
        # Analyze date overlap
        if len(ticker_date_ranges) > 1:
            overlap_analysis = self._analyze_date_overlap(ticker_date_ranges)
            cross_validation['date_overlap_analysis'] = overlap_analysis
            
            # Check minimum overlap requirement
            min_overlap = overlap_analysis.get('min_overlap_ratio', 0)
            if min_overlap < self.min_overlap_ratio:
                cross_validation['issues'].append(
                    f"Insufficient date overlap: {min_overlap:.2%} < {self.min_overlap_ratio:.2%}"
                )
                cross_validation['consistent'] = False
        
        # Analyze distribution consistency for returns
        distribution_analysis = self._analyze_return_distributions(ticker_data)
        cross_validation['distribution_analysis'] = distribution_analysis
        
        # Check for extremely divergent distributions
        if distribution_analysis.get('divergent_tickers'):
            cross_validation['warnings'].append(
                f"Divergent return distributions: {distribution_analysis['divergent_tickers']}"
            )
        
        return cross_validation
    
    def _analyze_date_overlap(self, ticker_date_ranges: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze date overlap between tickers"""
        
        tickers = list(ticker_date_ranges.keys())
        
        if len(tickers) < 2:
            return {'min_overlap_ratio': 1.0, 'pairwise_overlaps': {}}
        
        # Calculate pairwise overlaps
        pairwise_overlaps = {}
        min_overlap_ratio = 1.0
        
        for i, ticker_a in enumerate(tickers):
            for ticker_b in tickers[i+1:]:
                dates_a = ticker_date_ranges[ticker_a]['dates']
                dates_b = ticker_date_ranges[ticker_b]['dates']
                
                overlap = len(dates_a.intersection(dates_b))
                union = len(dates_a.union(dates_b))
                
                overlap_ratio = overlap / union if union > 0 else 0.0
                min_overlap_ratio = min(min_overlap_ratio, overlap_ratio)
                
                pairwise_overlaps[f"{ticker_a}_{ticker_b}"] = {
                    'overlap_count': overlap,
                    'overlap_ratio': overlap_ratio,
                    'ticker_a_count': len(dates_a),
                    'ticker_b_count': len(dates_b)
                }
        
        return {
            'min_overlap_ratio': min_overlap_ratio,
            'pairwise_overlaps': pairwise_overlaps
        }
    
    def _analyze_return_distributions(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze consistency of return distributions across tickers"""
        
        return_stats = {}
        
        for ticker, data in ticker_data.items():
            if 'close' in data.columns and len(data) > 1:
                returns = data['close'].pct_change().dropna()
                
                if len(returns) > 0:
                    return_stats[ticker] = {
                        'mean': returns.mean(),
                        'std': returns.std(),
                        'skew': returns.skew(),
                        'kurtosis': returns.kurtosis(),
                        'count': len(returns)
                    }
        
        # Identify potentially divergent tickers (extreme skew or kurtosis)
        divergent_tickers = []
        if len(return_stats) > 1:
            skew_values = [stats['skew'] for stats in return_stats.values() if not pd.isna(stats['skew'])]
            kurtosis_values = [stats['kurtosis'] for stats in return_stats.values() if not pd.isna(stats['kurtosis'])]
            
            if skew_values:
                skew_threshold = np.percentile(np.abs(skew_values), 90)  # 90th percentile
                for ticker, stats in return_stats.items():
                    if not pd.isna(stats['skew']) and abs(stats['skew']) > skew_threshold * 2:
                        divergent_tickers.append(ticker)
        
        return {
            'ticker_statistics': return_stats,
            'divergent_tickers': divergent_tickers
        }
    
    def _validate_external_data_alignment(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate alignment with external data sources"""
        
        external_validation = {
            'market_data_alignment': 'not_applicable',
            'sector_data_alignment': 'not_applicable',
            'vix_data_alignment': 'not_applicable',
            'issues': []
        }
        
        # This would be enhanced with actual external data validation
        # For now, just check that we have date ranges that could align
        
        if ticker_data:
            # Get overall date range of ticker data
            all_dates = set()
            for data in ticker_data.values():
                if isinstance(data.index, pd.DatetimeIndex):
                    dates = data.index
                elif 'date' in data.columns:
                    dates = pd.to_datetime(data['date'])
                else:
                    continue
                all_dates.update(dates.date)
            
            if all_dates:
                min_date = min(all_dates)
                max_date = max(all_dates)
                
                # Check if date range is reasonable for external data
                if min_date < datetime(2010, 1, 1).date():
                    external_validation['issues'].append("Data range extends before 2010 - external data may be limited")
                
                date_range_days = (max_date - min_date).days
                if date_range_days > 365 * 15:  # More than 15 years
                    external_validation['issues'].append(f"Very long date range ({date_range_days} days) - ensure external data covers full period")
        
        return external_validation
    
    def validate_feature_quality(self, 
                                portfolio_features: Dict[str, pd.DataFrame],
                                feature_names: List[str]) -> Dict[str, Any]:
        """Validate quality of calculated features across portfolio"""
        
        quality_results = {
            'timestamp': datetime.now().isoformat(),
            'ticker_validations': {},
            'cross_ticker_feature_analysis': {},
            'overall_quality': True,
            'issues': []
        }
        
        # Validate features for each ticker using existing pipeline validator
        for ticker, features in portfolio_features.items():
            if features is not None:
                is_valid, issues = self.pipeline_validator.validate_feature_data(features, feature_names)
                
                quality_results['ticker_validations'][ticker] = {
                    'is_valid': is_valid,
                    'issues': issues,
                    'feature_count': len(features.columns),
                    'sample_count': len(features)
                }
                
                if not is_valid:
                    quality_results['overall_quality'] = False
                    quality_results['issues'].extend([f"{ticker}: {issue}" for issue in issues])
        
        # Cross-ticker feature analysis
        cross_analysis = self._analyze_cross_ticker_features(portfolio_features, feature_names)
        quality_results['cross_ticker_feature_analysis'] = cross_analysis
        
        return quality_results
    
    def _analyze_cross_ticker_features(self, 
                                     portfolio_features: Dict[str, pd.DataFrame],
                                     feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature consistency across tickers"""
        
        analysis = {
            'feature_consistency': {},
            'correlation_analysis': {},
            'distribution_similarity': {}
        }
        
        # Analyze each feature across tickers
        for feature_name in feature_names:
            feature_data = {}
            
            # Extract feature data from all tickers
            for ticker, features in portfolio_features.items():
                if features is not None and feature_name in features.columns:
                    feature_values = features[feature_name].dropna()
                    if len(feature_values) > 0:
                        feature_data[ticker] = feature_values
            
            if len(feature_data) > 1:
                # Calculate basic statistics
                stats_summary = {}
                for ticker, values in feature_data.items():
                    stats_summary[ticker] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'count': len(values)
                    }
                
                analysis['feature_consistency'][feature_name] = stats_summary
        
        return analysis
    
    def get_validation_summary(self) -> str:
        """Generate human-readable validation summary"""
        
        if not self.validation_results:
            return "No validation results available. Run validate_portfolio_data() first."
        
        results = self.validation_results
        
        summary = f"""
Multi-Ticker Validation Summary
==============================

Validation Date: {results['timestamp']}
Total Tickers: {results['total_tickers']}

OVERALL STATUS: {'✓ PASS' if results['overall_valid'] else '✗ FAIL'}

Individual Ticker Validation:
"""
        
        # Individual validations
        individual = results.get('individual_validations', {})
        passed = sum(1 for r in individual.values() if r.get('is_valid', False))
        failed = len(individual) - passed
        
        summary += f"- Passed: {passed}/{len(individual)} tickers\n"
        summary += f"- Failed: {failed}/{len(individual)} tickers\n"
        
        if failed > 0:
            failed_tickers = [ticker for ticker, result in individual.items() 
                            if not result.get('is_valid', False)]
            summary += f"- Failed tickers: {failed_tickers}\n"
        
        # Cross-ticker validation
        cross_ticker = results.get('cross_ticker_validation', {})
        if cross_ticker:
            summary += f"\nCross-Ticker Consistency: {'✓ PASS' if cross_ticker.get('consistent', True) else '✗ FAIL'}\n"
            
            date_overlap = cross_ticker.get('date_overlap_analysis', {})
            if date_overlap:
                min_overlap = date_overlap.get('min_overlap_ratio', 0)
                summary += f"- Minimum date overlap: {min_overlap:.2%}\n"
        
        # Issues and warnings
        if results.get('issues'):
            summary += f"\nISSUES:\n"
            for issue in results['issues']:
                summary += f"- {issue}\n"
        
        if results.get('warnings'):
            summary += f"\nWARNINGS:\n"
            for warning in results['warnings']:
                summary += f"- {warning}\n"
        
        summary += "\n" + "="*50
        
        return summary


def create_multi_ticker_validator(min_overlap_ratio: float = 0.8,
                                max_date_gap_days: int = 7) -> MultiTickerValidator:
    """Convenience function to create multi-ticker validator"""
    
    return MultiTickerValidator(
        min_overlap_ratio=min_overlap_ratio,
        max_date_gap_days=max_date_gap_days
    )