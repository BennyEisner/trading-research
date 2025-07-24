#!/usr/bin/env python3

"""
Utility functions for the ML training pipeline
"""

import json
import os
from datetime import timedelta

import numpy as np


class ValidationUtils:
    """Data validation utilities"""

    @staticmethod
    def validate_data_quality(data, column_name="data", logger=None):
        """Validate data quality and log issues"""
        issues = []

        # Check for NaN values
        nan_count = data.isna().sum() if hasattr(data, "isna") else np.isnan(data).sum()
        if nan_count > 0:
            issues.append(f"Contains {nan_count} NaN values")

        # Check for infinite values
        if hasattr(data, "replace"):
            inf_count = np.isinf(data.replace([np.inf, -np.inf], np.nan)).sum()
        else:
            inf_count = np.isinf(data).sum()
        if inf_count > 0:
            issues.append(f"Contains {inf_count} infinite values")

        # Check data range
        if hasattr(data, "min") and hasattr(data, "max"):
            data_min, data_max = data.min(), data.max()
            if abs(data_max - data_min) < 1e-10:
                issues.append("Data has very low variance")

        # Log issues
        if issues and logger:
            logger.log(f"Data quality issues in {column_name}: {'; '.join(issues)}")

        return len(issues) == 0, issues

    @staticmethod
    def validate_sequences(X, y, logger=None):
        """Validate sequence data"""
        issues = []

        # Check shapes
        if len(X) != len(y):
            issues.append(f"Sequence length mismatch: X={len(X)}, y={len(y)}")

        # Check for empty sequences
        if len(X) == 0:
            issues.append("Empty sequences")

        # Check sequence consistency
        if len(X) > 0:
            seq_lengths = [len(seq) for seq in X]
            if len(set(seq_lengths)) > 1:
                issues.append("Inconsistent sequence lengths")

        if issues and logger:
            logger.log(f"Sequence validation issues: {'; '.join(issues)}")

        return len(issues) == 0, issues


class FileUtils:
    """File handling utilities"""

    @staticmethod
    def ensure_directory(filepath):
        """Ensure directory exists for given filepath"""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def save_json(data, filepath, logger=None):
        """Save data to JSON file with error handling"""
        try:
            FileUtils.ensure_directory(filepath)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
            if logger:
                logger.log(f"Data saved to {filepath}")
            return True
        except Exception as e:
            if logger:
                logger.log(f"Failed to save to {filepath}: {e}")
            return False

    @staticmethod
    def load_json(filepath, logger=None):
        """Load data from JSON file with error handling"""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            if logger:
                logger.log(f"Data loaded from {filepath}")
            return data
        except Exception as e:
            if logger:
                logger.log(f"Failed to load from {filepath}: {e}")
            return None


class MemoryUtils:
    """Memory management utilities"""

    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except ImportError:
            return None

    @staticmethod
    def log_memory_usage(logger, phase_name=""):
        """Log current memory usage"""
        memory_mb = MemoryUtils.get_memory_usage()
        if memory_mb and logger:
            logger.log(f"Memory usage {phase_name}: {memory_mb:.1f} MB")


class MetricsUtils:
    """Additional metrics and statistical utilities"""

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio for returns"""
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns)

    @staticmethod
    def calculate_max_drawdown(cumulative_returns):
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)

    @staticmethod
    def calculate_hit_rate(y_true, y_pred, threshold=0.0):
        """Calculate hit rate (percentage of correct direction predictions)"""
        correct_direction = np.sign(y_true) == np.sign(y_pred)
        # Only consider predictions above threshold
        if threshold > 0:
            significant_mask = np.abs(y_true) > threshold
            if np.sum(significant_mask) > 0:
                correct_direction = correct_direction[significant_mask]

        return np.mean(correct_direction) * 100

    @staticmethod
    def calculate_prediction_confidence(y_pred, confidence_levels=[0.5, 0.7, 0.9]):
        """Calculate prediction confidence statistics"""
        abs_pred = np.abs(y_pred)
        percentiles = [np.percentile(abs_pred, level * 100) for level in confidence_levels]

        confidence_stats = {}
        for i, level in enumerate(confidence_levels):
            threshold = percentiles[i]
            high_confidence_mask = abs_pred >= threshold
            confidence_stats[f"confidence_{level}"] = {
                "threshold": threshold,
                "percentage": np.mean(high_confidence_mask) * 100,
                "count": np.sum(high_confidence_mask),
            }

        return confidence_stats


class DateUtils:
    """Date and time utilities"""

    @staticmethod
    def get_market_dates(start_date, end_date):
        """Get list of market dates (weekdays) between start and end"""
        dates = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                dates.append(current)
            current += timedelta(days=1)
        return dates

    @staticmethod
    def is_market_day(date):
        """Check if date is a market day (weekday)"""
        return date.weekday() < 5

    @staticmethod
    def get_next_market_day(date):
        """Get next market day after given date"""
        next_day = date + timedelta(days=1)
        while not DateUtils.is_market_day(next_day):
            next_day += timedelta(days=1)
        return next_day


class ConfigUtils:
    """Configuration utilities"""

    @staticmethod
    def merge_configs(base_config, override_config):
        """Merge two configuration dictionaries"""
        merged = base_config.copy()
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigUtils.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def validate_config_keys(config, required_keys, logger=None):
        """Validate that all required keys are present in config"""
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys and logger:
            logger.log(f"Missing required configuration keys: {missing_keys}")
        return len(missing_keys) == 0


class ModelUtils:
    """Model-related utilities"""

    @staticmethod
    def count_parameters(model):
        """Count total number of trainable parameters in model"""
        return sum(np.prod(layer.get_weights()[0].shape) for layer in model.layers if layer.get_weights())

    @staticmethod
    def get_model_summary_dict(model):
        """Get model summary as dictionary"""
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))

        return {
            "total_params": model.count_params(),
            "trainable_params": sum(np.prod(layer.get_weights()[0].shape) for layer in model.layers if layer.get_weights()),
            "summary_text": "\n".join(summary_lines),
            "layer_count": len(model.layers),
        }


# Convenience functions for common operations
def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value for zero denominator"""
    return numerator / denominator if denominator != 0 else default


def ensure_list(item):
    """Ensure item is a list"""
    return item if isinstance(item, list) else [item]


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
