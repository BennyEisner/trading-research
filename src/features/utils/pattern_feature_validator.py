#!/usr/bin/env python3

"""
Pattern Feature Quality Validator
Specialized validation for pattern features with temporal and cross-asset dependencies
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

from ...validation.pipeline_validator import PipelineValidator

warnings.filterwarnings("ignore")


class PatternFeatureValidator:
    """
    Specialized validator for pattern features with focus on:
    - Temporal dependency validation
    - Cross-asset relationship validation  
    - Pattern-specific quality checks
    """
    
    def __init__(self, 
                 base_validator: Optional[PipelineValidator] = None,
                 autocorr_threshold: float = 0.3,
                 pattern_stability_threshold: float = 0.1):
        """
        Initialize pattern feature validator
        
        Args:
            base_validator: Base pipeline validator
            autocorr_threshold: Minimum required autocorrelation for temporal features
            pattern_stability_threshold: Maximum allowed coefficient of variation for stable patterns
        """
        
        self.base_validator = base_validator or PipelineValidator()
        self.autocorr_threshold = autocorr_threshold
        self.pattern_stability_threshold = pattern_stability_threshold
        
        # Define feature categories for specialized validation
        self.feature_categories = {
            'temporal_dependencies': [
                'momentum_persistence_7d',
                'volatility_clustering',
                'trend_exhaustion',
                'garch_volatility_forecast'
            ],
            'nonlinear_patterns': [
                'price_acceleration',
                'volume_price_divergence',
                'volatility_regime_change',
                'return_skewness_7d'
            ],
            'microstructure_patterns': [
                'intraday_range_expansion',
                'overnight_gap_behavior',
                'end_of_day_momentum'
            ],
            'cross_asset_features': [
                'sector_relative_strength',
                'market_beta_instability', 
                'vix_term_structure'
            ],
            'core_context': [
                'returns_1d',
                'returns_3d',
                'returns_7d',
                'volume_normalized',
                'close'
            ]
        }
        
        self.validation_results = {}
    
    def validate_pattern_features(self, 
                                features: pd.DataFrame,
                                ticker: str = "UNKNOWN") -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive validation of pattern features
        
        Args:
            features: DataFrame with calculated pattern features
            ticker: Ticker symbol for reporting
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'total_features': len(features.columns),
            'sample_count': len(features),
            'category_validations': {},
            'temporal_analysis': {},
            'distribution_analysis': {},
            'cross_feature_analysis': {},
            'issues': [],
            'warnings': [],
            'overall_valid': True
        }
        
        # Step 1: Base validation using existing infrastructure
        feature_names = list(features.columns)
        base_valid, base_issues = self.base_validator.validate_feature_data(features, feature_names)
        
        if not base_valid:
            validation_results['issues'].extend([f"Base validation: {issue}" for issue in base_issues])
            validation_results['overall_valid'] = False
        
        # Step 2: Category-specific validation
        category_results = self._validate_feature_categories(features)
        validation_results['category_validations'] = category_results
        
        # Update overall validity based on category results
        for category, result in category_results.items():
            if not result.get('valid', True):
                validation_results['issues'].extend([f"{category}: {issue}" for issue in result.get('issues', [])])
        
        # Step 3: Temporal dependency analysis
        temporal_analysis = self._validate_temporal_dependencies(features)
        validation_results['temporal_analysis'] = temporal_analysis
        
        if not temporal_analysis.get('temporal_valid', True):
            validation_results['warnings'].extend(temporal_analysis.get('warnings', []))
        
        # Step 4: Distribution analysis
        distribution_analysis = self._validate_feature_distributions(features)
        validation_results['distribution_analysis'] = distribution_analysis
        
        # Step 5: Cross-feature relationship analysis
        cross_feature_analysis = self._validate_cross_feature_relationships(features)
        validation_results['cross_feature_analysis'] = cross_feature_analysis
        
        # Final validation decision
        critical_issues = len([issue for issue in validation_results['issues'] if 'critical' in issue.lower()])
        validation_results['overall_valid'] = (
            base_valid and 
            critical_issues == 0 and
            len(validation_results['issues']) <= 3  # Allow some minor issues
        )
        
        self.validation_results[ticker] = validation_results
        
        return validation_results['overall_valid'], validation_results
    
    def _validate_feature_categories(self, features: pd.DataFrame) -> Dict[str, Dict]:
        """Validate features by category with specialized checks"""
        
        category_results = {}
        
        for category, feature_list in self.feature_categories.items():
            category_features = [f for f in feature_list if f in features.columns]
            
            if not category_features:
                category_results[category] = {
                    'valid': False,
                    'issues': ['No features found for this category'],
                    'feature_count': 0
                }
                continue
            
            category_data = features[category_features]
            
            # Category-specific validation
            if category == 'temporal_dependencies':
                result = self._validate_temporal_category(category_data)
            elif category == 'nonlinear_patterns':
                result = self._validate_nonlinear_category(category_data)
            elif category == 'microstructure_patterns':
                result = self._validate_microstructure_category(category_data)
            elif category == 'cross_asset_features':
                result = self._validate_cross_asset_category(category_data)
            else:
                result = self._validate_generic_category(category_data)
            
            result['feature_count'] = len(category_features)
            result['features'] = category_features
            category_results[category] = result
        
        return category_results
    
    def _validate_temporal_category(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal dependency features"""
        
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'temporal_characteristics': {}
        }
        
        for feature in data.columns:
            feature_data = data[feature].dropna()
            
            if len(feature_data) < 30:  # Need sufficient data for temporal analysis
                result['issues'].append(f"{feature}: Insufficient data for temporal validation")
                result['valid'] = False
                continue
            
            # Check for autocorrelation (temporal features should show some persistence)
            try:
                autocorr = acf(feature_data, nlags=5, fft=True)[1]  # 1-lag autocorrelation
                result['temporal_characteristics'][feature] = {
                    'autocorrelation_lag1': autocorr,
                    'has_temporal_structure': abs(autocorr) > 0.1
                }
                
                if feature in ['momentum_persistence_7d', 'volatility_clustering'] and abs(autocorr) < 0.05:
                    result['warnings'].append(f"{feature}: Very low autocorrelation ({autocorr:.3f})")
                    
            except Exception as e:
                result['warnings'].append(f"{feature}: Could not calculate autocorrelation - {str(e)}")
        
        return result
    
    def _validate_nonlinear_category(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate non-linear pattern features"""
        
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'nonlinearity_tests': {}
        }
        
        for feature in data.columns:
            feature_data = data[feature].dropna()
            
            if len(feature_data) == 0:
                result['issues'].append(f"{feature}: No valid data")
                result['valid'] = False
                continue
            
            # Test for non-linear characteristics
            nonlinearity_score = self._calculate_nonlinearity_score(feature_data)
            result['nonlinearity_tests'][feature] = {
                'nonlinearity_score': nonlinearity_score,
                'is_nonlinear': nonlinearity_score > 0.1
            }
            
            # Check for appropriate variability (not too stable, not too chaotic)
            if feature_data.std() > 0:
                coeff_var = feature_data.std() / abs(feature_data.mean() + 1e-8)
                if coeff_var > 10:  # Very high variability
                    result['warnings'].append(f"{feature}: Very high variability (CV={coeff_var:.2f})")
                elif coeff_var < 0.01:  # Very low variability  
                    result['warnings'].append(f"{feature}: Very low variability (CV={coeff_var:.4f})")
        
        return result
    
    def _validate_microstructure_category(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate market microstructure features"""
        
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'microstructure_characteristics': {}
        }
        
        for feature in data.columns:
            feature_data = data[feature].dropna()
            
            if len(feature_data) == 0:
                result['issues'].append(f"{feature}: No valid data")
                result['valid'] = False
                continue
            
            # Check for reasonable ranges
            if feature == 'intraday_range_expansion':
                # Should be positive and typically between 0.1 and 5.0
                if (feature_data < 0).any():
                    result['issues'].append(f"{feature}: Negative values detected")
                    result['valid'] = False
                elif feature_data.max() > 20:
                    result['warnings'].append(f"{feature}: Very high maximum value ({feature_data.max():.2f})")
            
            elif feature == 'overnight_gap_behavior':
                # Should be small percentage changes
                if abs(feature_data).max() > 0.5:  # 50% gap
                    result['warnings'].append(f"{feature}: Very large gaps detected (max: {feature_data.max():.2%})")
            
            elif feature == 'end_of_day_momentum':
                # Should be between 0 and 1
                if (feature_data < 0).any() or (feature_data > 1).any():
                    result['warnings'].append(f"{feature}: Values outside expected range [0,1]")
            
            result['microstructure_characteristics'][feature] = {
                'min': feature_data.min(),
                'max': feature_data.max(),
                'mean': feature_data.mean(),
                'std': feature_data.std()
            }
        
        return result
    
    def _validate_cross_asset_category(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate cross-asset relationship features"""
        
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'cross_asset_analysis': {}
        }
        
        for feature in data.columns:
            feature_data = data[feature].dropna()
            
            # Check if feature is all zeros (indicating missing external data)
            if len(feature_data) > 0:
                zero_ratio = (feature_data == 0).sum() / len(feature_data)
                result['cross_asset_analysis'][feature] = {
                    'zero_ratio': zero_ratio,
                    'external_data_available': zero_ratio < 0.9
                }
                
                if zero_ratio > 0.9:
                    result['warnings'].append(f"{feature}: Mostly zeros - external data may be missing")
                elif zero_ratio > 0.5:
                    result['warnings'].append(f"{feature}: Many zeros ({zero_ratio:.1%}) - check external data quality")
            else:
                result['issues'].append(f"{feature}: No valid data")
        
        return result
    
    def _validate_generic_category(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generic validation for other feature categories"""
        
        result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Basic checks for any feature category
        for feature in data.columns:
            feature_data = data[feature].dropna()
            
            if len(feature_data) == 0:
                result['issues'].append(f"{feature}: No valid data")
                result['valid'] = False
            elif len(feature_data) < 10:
                result['warnings'].append(f"{feature}: Very few valid values ({len(feature_data)})")
        
        return result
    
    def _validate_temporal_dependencies(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal dependencies across features"""
        
        temporal_analysis = {
            'temporal_valid': True,
            'warnings': [],
            'autocorrelation_summary': {},
            'temporal_stability': {}
        }
        
        temporal_features = [f for f in self.feature_categories['temporal_dependencies'] if f in features.columns]
        
        if len(temporal_features) == 0:
            temporal_analysis['warnings'].append("No temporal features found for analysis")
            return temporal_analysis
        
        # Analyze autocorrelation structure
        for feature in temporal_features:
            feature_data = features[feature].dropna()
            
            if len(feature_data) > 30:
                try:
                    # Calculate multiple lags of autocorrelation
                    max_lags = min(20, len(feature_data) // 4)
                    autocorr_values = acf(feature_data, nlags=max_lags, fft=True)
                    
                    temporal_analysis['autocorrelation_summary'][feature] = {
                        'lag1': autocorr_values[1] if len(autocorr_values) > 1 else 0,
                        'lag5': autocorr_values[5] if len(autocorr_values) > 5 else 0,
                        'significant_lags': (np.abs(autocorr_values[1:]) > 0.1).sum()
                    }
                    
                except Exception as e:
                    temporal_analysis['warnings'].append(f"Could not analyze {feature}: {str(e)}")
        
        return temporal_analysis
    
    def _validate_feature_distributions(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature distributions for reasonable characteristics"""
        
        distribution_analysis = {
            'distribution_summary': {},
            'outlier_analysis': {},
            'normality_tests': {}
        }
        
        for feature in features.columns:
            feature_data = features[feature].dropna()
            
            if len(feature_data) > 10:
                # Basic distribution statistics
                distribution_analysis['distribution_summary'][feature] = {
                    'mean': feature_data.mean(),
                    'std': feature_data.std(),
                    'skewness': feature_data.skew(),
                    'kurtosis': feature_data.kurtosis(),
                    'min': feature_data.min(),
                    'max': feature_data.max()
                }
                
                # Outlier analysis
                if feature_data.std() > 0:
                    z_scores = np.abs((feature_data - feature_data.mean()) / feature_data.std())
                    outlier_count = (z_scores > 3).sum()
                    distribution_analysis['outlier_analysis'][feature] = {
                        'outlier_count': outlier_count,
                        'outlier_ratio': outlier_count / len(feature_data)
                    }
        
        return distribution_analysis
    
    def _validate_cross_feature_relationships(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate relationships between features"""
        
        cross_analysis = {
            'correlation_matrix': {},
            'highly_correlated_pairs': [],
            'independence_tests': {}
        }
        
        # Calculate correlation matrix for numeric features
        numeric_features = features.select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr()
            cross_analysis['correlation_matrix'] = corr_matrix.to_dict()
            
            # Find highly correlated pairs
            for i, feature_a in enumerate(corr_matrix.columns):
                for j, feature_b in enumerate(corr_matrix.columns[i+1:], i+1):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.9:  # Very high correlation
                        cross_analysis['highly_correlated_pairs'].append({
                            'feature_a': feature_a,
                            'feature_b': feature_b,
                            'correlation': corr_value
                        })
        
        return cross_analysis
    
    def _calculate_nonlinearity_score(self, data: pd.Series) -> float:
        """Calculate a simple nonlinearity score for a feature"""
        
        if len(data) < 10:
            return 0.0
        
        try:
            # Compare linear vs polynomial fit
            x = np.arange(len(data))
            
            # Linear fit
            linear_coef = np.polyfit(x, data, 1)
            linear_pred = np.polyval(linear_coef, x)
            linear_mse = np.mean((data - linear_pred) ** 2)
            
            # Quadratic fit
            if len(data) > 20:
                quad_coef = np.polyfit(x, data, 2)
                quad_pred = np.polyval(quad_coef, x)
                quad_mse = np.mean((data - quad_pred) ** 2)
                
                # Nonlinearity score based on improvement from quadratic fit
                if linear_mse > 0:
                    nonlinearity_score = max(0, (linear_mse - quad_mse) / linear_mse)
                else:
                    nonlinearity_score = 0.0
            else:
                nonlinearity_score = 0.0
            
            return nonlinearity_score
            
        except:
            return 0.0
    
    def generate_validation_report(self, ticker: str = None) -> str:
        """Generate comprehensive validation report"""
        
        if not self.validation_results:
            return "No validation results available. Run validate_pattern_features() first."
        
        if ticker and ticker in self.validation_results:
            results_to_report = {ticker: self.validation_results[ticker]}
        else:
            results_to_report = self.validation_results
        
        report = f"""
Pattern Feature Validation Report
=================================

Validation Date: {datetime.now().isoformat()}
Tickers Analyzed: {len(results_to_report)}

"""
        
        for ticker, results in results_to_report.items():
            report += f"\nTicker: {ticker}\n"
            report += f"{'='*50}\n"
            
            overall_status = "✓ PASS" if results['overall_valid'] else "✗ FAIL"
            report += f"Overall Status: {overall_status}\n"
            report += f"Total Features: {results['total_features']}\n"
            report += f"Sample Count: {results['sample_count']}\n"
            
            # Category results
            report += f"\nCategory Validation Results:\n"
            for category, cat_result in results.get('category_validations', {}).items():
                cat_status = "✓" if cat_result.get('valid', True) else "✗"
                feature_count = cat_result.get('feature_count', 0)
                report += f"  {cat_status} {category}: {feature_count} features\n"
                
                if cat_result.get('issues'):
                    for issue in cat_result['issues']:
                        report += f"    - Issue: {issue}\n"
                
                if cat_result.get('warnings'):
                    for warning in cat_result['warnings']:
                        report += f"    - Warning: {warning}\n"
            
            # Overall issues
            if results.get('issues'):
                report += f"\nISSUES:\n"
                for issue in results['issues']:
                    report += f"  - {issue}\n"
            
            if results.get('warnings'):
                report += f"\nWARNINGS:\n"
                for warning in results['warnings']:
                    report += f"  - {warning}\n"
        
        return report


def create_pattern_feature_validator(autocorr_threshold: float = 0.3,
                                   pattern_stability_threshold: float = 0.1) -> PatternFeatureValidator:
    """Convenience function to create pattern feature validator"""
    
    return PatternFeatureValidator(
        autocorr_threshold=autocorr_threshold,
        pattern_stability_threshold=pattern_stability_threshold
    )