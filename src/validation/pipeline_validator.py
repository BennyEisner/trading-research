#!/usr/bin/env python3

"""
Comprehensive pipeline validation and data quality checks
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Any

from ..utils.utils import ValidationUtils, MetricsUtils


class PipelineValidator:
    """Comprehensive validation for ML pipeline components"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.validation_results = {}
    
    def log(self, message, level="INFO"):
        """Log message with fallback to print"""
        if self.logger:
            self.logger.log(message)
        else:
            print(f"[{level}] {message}")
    
    def validate_raw_data(self, data: pd.DataFrame, ticker: str) -> Tuple[bool, List[str]]:
        """Validate raw OHLCV data quality"""
        issues = []
        
        # Check required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        if issues:  # Can't continue without basic columns
            return False, issues
        
        # Check data types
        try:
            data['date'] = pd.to_datetime(data['date'])
        except Exception as e:
            issues.append(f"Invalid date format: {e}")
        
        # Check for OHLC consistency
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if data[col].dtype not in ['float64', 'int64']:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    issues.append(f"Cannot convert {col} to numeric: {e}")
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.sum() > 0:
            issues.append(f"Invalid OHLC relationships in {invalid_ohlc.sum()} records")
        
        # Check for negative prices or volumes
        for col in price_cols + ['volume']:
            if (data[col] <= 0).sum() > 0:
                negative_count = (data[col] <= 0).sum()
                issues.append(f"Non-positive values in {col}: {negative_count} records")
        
        # Check for extreme price movements (>50% daily change)
        if len(data) > 1:
            price_changes = data['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()
            if extreme_changes > 0:
                issues.append(f"Extreme price movements (>50%): {extreme_changes} occurrences")
        
        # Check data continuity (gaps in time series)
        data_sorted = data.sort_values('date')
        date_diffs = data_sorted['date'].diff()
        if len(data) > 1:
            median_gap = date_diffs.median()
            large_gaps = (date_diffs > median_gap * 7).sum()  # 7x normal gap
            if large_gaps > 0:
                issues.append(f"Large time gaps detected: {large_gaps} instances")
        
        # Data quality summary
        total_records = len(data)
        date_range = f"{data['date'].min()} to {data['date'].max()}"
        
        self.validation_results[f'{ticker}_raw_data'] = {
            'total_records': total_records,
            'date_range': date_range,
            'issues': issues,
            'is_valid': len(issues) == 0
        }
        
        if issues:
            self.log(f"Data quality issues for {ticker}: {'; '.join(issues)}", "WARNING")
        else:
            self.log(f"Raw data validation passed for {ticker}: {total_records} records")
        
        return len(issues) == 0, issues
    
    def validate_feature_data(self, data: pd.DataFrame, feature_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate engineered features"""
        issues = []
        
        if not feature_columns:
            issues.append("No features provided for validation")
            return False, issues
        
        feature_data = data[feature_columns]
        
        # Check for NaN values
        nan_counts = feature_data.isna().sum()
        high_nan_features = nan_counts[nan_counts > len(data) * 0.1].index.tolist()
        if high_nan_features:
            issues.append(f"High NaN counts (>10%) in features: {high_nan_features}")
        
        # Check for infinite values
        inf_counts = {}
        for col in feature_columns:
            if np.isinf(feature_data[col]).sum() > 0:
                inf_counts[col] = np.isinf(feature_data[col]).sum()
        
        if inf_counts:
            issues.append(f"Infinite values detected: {inf_counts}")
        
        # Check for constant features (zero variance)
        constant_features = []
        for col in feature_columns:
            if feature_data[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            issues.append(f"Constant features (zero variance): {constant_features}")
        
        # Check for extreme outliers (>6 standard deviations)
        outlier_features = {}
        for col in feature_columns:
            if feature_data[col].dtype in ['float64', 'int64']:
                z_scores = np.abs((feature_data[col] - feature_data[col].mean()) / feature_data[col].std())
                extreme_outliers = (z_scores > 6).sum()
                if extreme_outliers > 0:
                    outlier_features[col] = extreme_outliers
        
        if outlier_features:
            issues.append(f"Extreme outliers detected: {outlier_features}")
        
        # Check feature correlation (highly correlated features)
        try:
            correlation_matrix = feature_data.corr()
            high_correlations = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = abs(correlation_matrix.iloc[i, j])
                    if corr_value > 0.95 and not np.isnan(corr_value):
                        high_correlations.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            corr_value
                        ))
            
            if high_correlations:
                issues.append(f"Highly correlated features (>0.95): {len(high_correlations)} pairs")
                
        except Exception as e:
            issues.append(f"Could not compute correlations: {e}")
        
        # Feature distribution analysis
        skewed_features = []
        for col in feature_columns:
            if feature_data[col].dtype in ['float64', 'int64']:
                skewness = feature_data[col].skew()
                if abs(skewness) > 3:  # Highly skewed
                    skewed_features.append(col)
        
        if len(skewed_features) > len(feature_columns) * 0.5:
            issues.append(f"Many highly skewed features: {len(skewed_features)}/{len(feature_columns)}")
        
        self.validation_results['feature_validation'] = {
            'total_features': len(feature_columns),
            'nan_features': len(high_nan_features),
            'infinite_features': len(inf_counts),
            'constant_features': len(constant_features),
            'outlier_features': len(outlier_features),
            'skewed_features': len(skewed_features),
            'issues': issues,
            'is_valid': len(issues) == 0
        }
        
        if issues:
            self.log(f"Feature validation issues: {'; '.join(issues)}", "WARNING")
        else:
            self.log(f"Feature validation passed: {len(feature_columns)} features")
        
        return len(issues) == 0, issues
    
    def validate_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate LSTM sequences"""
        issues = []
        
        # Basic shape validation
        if len(X.shape) != 3:
            issues.append(f"Expected 3D sequences, got shape {X.shape}")
        
        if len(y.shape) != 1:
            issues.append(f"Expected 1D targets, got shape {y.shape}")
        
        if len(X) != len(y):
            issues.append(f"Sequence count mismatch: X={len(X)}, y={len(y)}")
        
        if len(issues) > 0:  # Can't continue with shape issues
            return False, issues
        
        # Check for NaN values
        nan_sequences = np.isnan(X).any(axis=(1, 2)).sum()
        nan_targets = np.isnan(y).sum()
        
        if nan_sequences > 0:
            issues.append(f"NaN values in {nan_sequences} sequences")
        
        if nan_targets > 0:
            issues.append(f"NaN values in {nan_targets} targets")
        
        # Check for infinite values
        inf_sequences = np.isinf(X).any(axis=(1, 2)).sum()
        inf_targets = np.isinf(y).sum()
        
        if inf_sequences > 0:
            issues.append(f"Infinite values in {inf_sequences} sequences")
        
        if inf_targets > 0:
            issues.append(f"Infinite values in {inf_targets} targets")
        
        # Check sequence diversity
        sequence_stds = np.std(X, axis=1)
        zero_variance_sequences = (sequence_stds == 0).any(axis=1).sum()
        
        # Check if ALL features have zero variance (which would be a real problem)
        all_features_zero_var = (sequence_stds == 0).all(axis=1).sum()
        
        if all_features_zero_var > len(X) * 0.1:  # More than 10% of sequences have zero variance in ALL features
            issues.append(f"High number of sequences with zero variance in all features: {all_features_zero_var}")
        elif zero_variance_sequences == len(X):  # Every sequence has at least one zero-variance feature
            if self.logger:
                self.logger.info(f"Note: All sequences have some zero-variance features (likely ticker identity), but this may be expected")
            else:
                print(f"Note: All sequences have some zero-variance features (likely ticker identity), but this may be expected")
        
        # Check target distribution
        target_std = np.std(y)
        if target_std == 0:
            issues.append("All targets are identical (zero variance)")
        elif target_std < 1e-6:
            issues.append(f"Very low target variance: {target_std}")
        
        # Check for extreme target values
        target_percentiles = np.percentile(y, [1, 99])
        extreme_targets = ((y < target_percentiles[0] * 10) | (y > target_percentiles[1] * 10)).sum()
        
        if extreme_targets > 0:
            issues.append(f"Extreme target values detected: {extreme_targets}")
        
        self.validation_results['sequence_validation'] = {
            'sequence_shape': X.shape,
            'target_shape': y.shape,
            'nan_sequences': nan_sequences,
            'nan_targets': nan_targets,
            'zero_variance_sequences': zero_variance_sequences,
            'target_std': target_std,
            'issues': issues,
            'is_valid': len(issues) == 0
        }
        
        if issues:
            self.log(f"Sequence validation issues: {'; '.join(issues)}", "WARNING")
        else:
            self.log(f"Sequence validation passed: {X.shape}")
        
        return len(issues) == 0, issues
    
    def validate_model_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate model predictions"""
        issues = []
        
        # Basic validation
        if len(y_true) != len(y_pred):
            issues.append(f"Prediction count mismatch: true={len(y_true)}, pred={len(y_pred)}")
            return False, issues
        
        # Check for NaN/infinite predictions
        nan_predictions = np.isnan(y_pred).sum()
        inf_predictions = np.isinf(y_pred).sum()
        
        if nan_predictions > 0:
            issues.append(f"NaN predictions: {nan_predictions}")
        
        if inf_predictions > 0:
            issues.append(f"Infinite predictions: {inf_predictions}")
        
        # Check prediction distribution
        pred_std = np.std(y_pred)
        if pred_std == 0:
            issues.append("All predictions are identical")
        elif pred_std < 1e-6:
            issues.append(f"Very low prediction variance: {pred_std}")
        
        # Check for extreme predictions
        true_range = np.max(y_true) - np.min(y_true)
        pred_range = np.max(y_pred) - np.min(y_pred)
        
        if pred_range > true_range * 10:
            issues.append(f"Prediction range much larger than true range: {pred_range:.4f} vs {true_range:.4f}")
        
        # Statistical validation
        try:
            # Directional accuracy
            dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
            
            # Correlation
            correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
            
            # Mean Absolute Error
            mae = np.mean(np.abs(y_true - y_pred))
            
            if dir_acc < 40:  # Worse than random for binary classification
                issues.append(f"Very low directional accuracy: {dir_acc:.1f}%")
            
            if abs(correlation) < 0.1:
                issues.append(f"Very low correlation: {correlation:.3f}")
            
            self.validation_results['prediction_validation'] = {
                'directional_accuracy': dir_acc,
                'correlation': correlation,
                'mae': mae,
                'nan_predictions': nan_predictions,
                'inf_predictions': inf_predictions,
                'prediction_std': pred_std,
                'issues': issues,
                'is_valid': len(issues) == 0
            }
            
        except Exception as e:
            issues.append(f"Could not compute prediction metrics: {e}")
        
        if issues:
            self.log(f"Prediction validation issues: {'; '.join(issues)}", "WARNING")
        else:
            self.log(f"Prediction validation passed")
        
        return len(issues) == 0, issues
    
    def validate_training_stability(self, history: Dict) -> Tuple[bool, List[str]]:
        """Validate training stability from history"""
        issues = []
        
        if 'loss' not in history:
            issues.append("No loss history available")
            return False, issues
        
        losses = history['loss']
        
        # Check for exploding loss
        if len(losses) > 1:
            initial_loss = losses[0]
            final_loss = losses[-1]
            max_loss = max(losses)
            
            if final_loss > initial_loss * 5:
                issues.append(f"Loss explosion: {initial_loss:.4f} → {final_loss:.4f}")
            
            if max_loss > initial_loss * 10:
                issues.append(f"Loss spike detected: max={max_loss:.4f}")
        
        # Check for NaN losses
        nan_losses = sum(1 for loss in losses if np.isnan(loss))
        if nan_losses > 0:
            issues.append(f"NaN losses encountered: {nan_losses} epochs")
        
        # Check convergence
        if len(losses) >= 10:
            recent_losses = losses[-10:]
            loss_std = np.std(recent_losses)
            
            if loss_std > np.mean(recent_losses) * 0.5:
                issues.append("Training appears unstable (high loss variance)")
        
        # Validation loss comparison
        if 'val_loss' in history:
            val_losses = history['val_loss']
            train_val_gap = []
            
            for train_loss, val_loss in zip(losses, val_losses):
                if not (np.isnan(train_loss) or np.isnan(val_loss)):
                    train_val_gap.append(val_loss - train_loss)
            
            if train_val_gap:
                avg_gap = np.mean(train_val_gap)
                if avg_gap > np.mean(losses) * 2:
                    issues.append(f"Large train/validation gap indicates overfitting: {avg_gap:.4f}")
        
        self.validation_results['training_validation'] = {
            'epochs_completed': len(losses),
            'initial_loss': losses[0] if losses else None,
            'final_loss': losses[-1] if losses else None,
            'nan_losses': nan_losses,
            'issues': issues,
            'is_valid': len(issues) == 0
        }
        
        if issues:
            self.log(f"Training stability issues: {'; '.join(issues)}", "WARNING")
        else:
            self.log("Training stability validation passed")
        
        return len(issues) == 0, issues
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'overall_status': 'PASS'
        }
        
        # Check if any validation failed
        for validation_key, validation_data in self.validation_results.items():
            if isinstance(validation_data, dict) and not validation_data.get('is_valid', True):
                summary['overall_status'] = 'FAIL'
                break
        
        # Count issues
        total_issues = 0
        for validation_data in self.validation_results.values():
            if isinstance(validation_data, dict) and 'issues' in validation_data:
                total_issues += len(validation_data['issues'])
        
        summary['total_issues'] = total_issues
        
        return summary
    
    def log_validation_summary(self):
        """Log comprehensive validation summary"""
        summary = self.get_validation_summary()
        
        self.log("=" * 50)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 50)
        
        self.log(f"Overall Status: {summary['overall_status']}")
        self.log(f"Total Issues: {summary['total_issues']}")
        
        for validation_key, validation_data in self.validation_results.items():
            if isinstance(validation_data, dict):
                status = "✓ PASS" if validation_data.get('is_valid', True) else "✗ FAIL"
                issues_count = len(validation_data.get('issues', []))
                self.log(f"{status}: {validation_key} ({issues_count} issues)")
        
        return summary


def create_pipeline_validator(logger=None):
    """Convenience function to create pipeline validator"""
    return PipelineValidator(logger)