#!/usr/bin/env python3

"""
Data validation and cleaning utilities
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional


class DataValidator:
    """
    Comprehensive data validation for financial time series
    """
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_ohlcv_data(self, data: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Comprehensive OHLCV data validation
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Optional symbol name for reporting
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'symbol': symbol,
            'total_rows': len(data),
            'date_range': None,
            'issues': [],
            'warnings': [],
            'data_quality_score': 1.0
        }
        
        # Check required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            results['issues'].append(f"Missing required columns: {missing_cols}")
            return results
        
        # Check date range if index is datetime
        if isinstance(data.index, pd.DatetimeIndex):
            results['date_range'] = f"{data.index.min()} to {data.index.max()}"
        
        # Validate OHLC relationships
        ohlc_issues = self._validate_ohlc_relationships(data)
        results['issues'].extend(ohlc_issues)
        
        # Check for missing values
        missing_stats = self._check_missing_values(data, required_cols)
        results.update(missing_stats)
        
        # Detect outliers
        outlier_stats = self._detect_outliers(data, required_cols)
        results.update(outlier_stats)
        
        # Check data continuity
        continuity_issues = self._check_data_continuity(data)
        results['warnings'].extend(continuity_issues)
        
        # Calculate overall data quality score
        results['data_quality_score'] = self._calculate_quality_score(results)
        
        self.validation_results[symbol] = results
        return results
    
    def _validate_ohlc_relationships(self, data: pd.DataFrame) -> List[str]:
        """Validate OHLC price relationships"""
        issues = []
        
        # High should be >= Open, Close, Low
        if (data["high"] < data["open"]).any():
            issues.append("Invalid OHLC: High < Open detected")
        if (data["high"] < data["close"]).any():
            issues.append("Invalid OHLC: High < Close detected") 
        if (data["high"] < data["low"]).any():
            issues.append("Invalid OHLC: High < Low detected")
            
        # Low should be <= Open, Close, High
        if (data["low"] > data["open"]).any():
            issues.append("Invalid OHLC: Low > Open detected")
        if (data["low"] > data["close"]).any():
            issues.append("Invalid OHLC: Low > Close detected")
            
        # Volume should be non-negative
        if (data["volume"] < 0).any():
            issues.append("Invalid volume: Negative values detected")
            
        # Prices should be positive
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if (data[col] <= 0).any():
                issues.append(f"Invalid {col}: Non-positive values detected")
                
        return issues
    
    def _check_missing_values(self, data: pd.DataFrame, cols: List[str]) -> Dict:
        """Check for missing values in key columns"""
        missing_stats = {}
        total_missing = 0
        
        for col in cols:
            missing_count = data[col].isna().sum()
            missing_pct = missing_count / len(data) * 100
            missing_stats[f'{col}_missing_count'] = missing_count
            missing_stats[f'{col}_missing_pct'] = missing_pct
            total_missing += missing_count
            
        missing_stats['total_missing_values'] = total_missing
        missing_stats['overall_missing_pct'] = total_missing / (len(data) * len(cols)) * 100
        
        return missing_stats
    
    def _detect_outliers(self, data: pd.DataFrame, cols: List[str]) -> Dict:
        """Detect statistical outliers in price/volume data"""
        outlier_stats = {}
        
        for col in cols:
            if col in data.columns and data[col].notna().sum() > 0:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outliers = (z_scores > 5).sum()  # 5 sigma outliers
                outlier_stats[f'{col}_outliers'] = outliers
                outlier_stats[f'{col}_outlier_pct'] = outliers / len(data) * 100
                
        return outlier_stats
    
    def _check_data_continuity(self, data: pd.DataFrame) -> List[str]:
        """Check for data continuity issues"""
        warnings = []
        
        if isinstance(data.index, pd.DatetimeIndex):
            # Check for large gaps in time series
            time_diffs = data.index.to_series().diff()
            median_diff = time_diffs.median()
            
            # Flag gaps more than 3x the median difference
            large_gaps = time_diffs > (median_diff * 3)
            if large_gaps.sum() > 0:
                warnings.append(f"Found {large_gaps.sum()} large time gaps in data")
                
        # Check for consecutive identical values (potential data issues)
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                consecutive_same = (data[col] == data[col].shift(1))
                if consecutive_same.sum() > len(data) * 0.05:  # >5% consecutive same
                    warnings.append(f"High number of consecutive identical {col} values")
                    
        return warnings
    
    def _calculate_quality_score(self, results: Dict) -> float:
        """Calculate overall data quality score (0-1)"""
        score = 1.0
        
        # Penalize for issues
        score -= len(results['issues']) * 0.2
        score -= len(results['warnings']) * 0.05
        
        # Penalize for missing data
        if 'overall_missing_pct' in results:
            score -= results['overall_missing_pct'] / 100 * 0.5
            
        # Penalize for outliers
        outlier_keys = [k for k in results.keys() if k.endswith('_outlier_pct')]
        if outlier_keys:
            avg_outlier_pct = sum(results[k] for k in outlier_keys) / len(outlier_keys)
            score -= avg_outlier_pct / 100 * 0.3
            
        return max(0.0, min(1.0, score))
    
    def get_validation_summary(self) -> pd.DataFrame:
        """Get summary of all validation results"""
        if not self.validation_results:
            return pd.DataFrame()
            
        summary_data = []
        for symbol, results in self.validation_results.items():
            summary_data.append({
                'symbol': symbol,
                'total_rows': results['total_rows'],
                'date_range': results['date_range'],
                'issues_count': len(results['issues']),
                'warnings_count': len(results['warnings']),
                'quality_score': results['data_quality_score'],
                'missing_pct': results.get('overall_missing_pct', 0)
            })
            
        return pd.DataFrame(summary_data)


def clean_financial_data(data: pd.DataFrame, 
                        symbol: str = "",
                        remove_outliers: bool = True,
                        outlier_threshold: float = 5.0,
                        fill_method: str = "ffill") -> Tuple[pd.DataFrame, Dict]:
    """
    Clean financial time series data
    
    Args:
        data: Input DataFrame with OHLCV data
        symbol: Symbol name for reporting
        remove_outliers: Whether to remove statistical outliers
        outlier_threshold: Z-score threshold for outlier detection
        fill_method: Method for filling missing values
        
    Returns:
        Tuple of (cleaned_data, cleaning_stats)
    """
    original_rows = len(data)
    cleaning_stats = {
        'original_rows': original_rows,
        'outliers_removed': 0,
        'values_filled': 0,
        'invalid_rows_removed': 0
    }
    
    # Create copy to avoid modifying original
    cleaned_data = data.copy()
    
    # Remove rows with invalid OHLC relationships
    valid_ohlc = (
        (cleaned_data["high"] >= cleaned_data["low"]) &
        (cleaned_data["high"] >= cleaned_data["open"]) &
        (cleaned_data["high"] >= cleaned_data["close"]) &
        (cleaned_data["low"] <= cleaned_data["open"]) &
        (cleaned_data["low"] <= cleaned_data["close"]) &
        (cleaned_data["volume"] >= 0) &
        (cleaned_data["open"] > 0) &
        (cleaned_data["high"] > 0) &
        (cleaned_data["low"] > 0) &
        (cleaned_data["close"] > 0)
    )
    
    invalid_rows = (~valid_ohlc).sum()
    if invalid_rows > 0:
        cleaned_data = cleaned_data[valid_ohlc]
        cleaning_stats['invalid_rows_removed'] = invalid_rows
        
    # Remove outliers if requested
    if remove_outliers:
        numeric_cols = ["open", "high", "low", "close", "volume"]
        outlier_mask = pd.Series([True] * len(cleaned_data), index=cleaned_data.index)
        
        for col in numeric_cols:
            if col in cleaned_data.columns:
                z_scores = np.abs(stats.zscore(cleaned_data[col].dropna()))
                col_outliers = z_scores > outlier_threshold
                outlier_mask = outlier_mask & ~col_outliers
                cleaning_stats['outliers_removed'] += col_outliers.sum()
                
        cleaned_data = cleaned_data[outlier_mask]
        
    # Fill missing values
    missing_before = cleaned_data.isna().sum().sum()
    
    if fill_method == "ffill":
        cleaned_data = cleaned_data.fillna(method="ffill").fillna(method="bfill")
    elif fill_method == "interpolate":
        cleaned_data = cleaned_data.interpolate()
    elif fill_method == "drop":
        cleaned_data = cleaned_data.dropna()
        
    missing_after = cleaned_data.isna().sum().sum()
    cleaning_stats['values_filled'] = missing_before - missing_after
    
    # Sort by date if datetime index
    if isinstance(cleaned_data.index, pd.DatetimeIndex):
        cleaned_data = cleaned_data.sort_index()
        
    cleaning_stats['final_rows'] = len(cleaned_data)
    cleaning_stats['data_retention'] = len(cleaned_data) / original_rows
    
    return cleaned_data, cleaning_stats