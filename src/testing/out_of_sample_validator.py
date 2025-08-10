#!/usr/bin/env python3

"""
Out-of-Sample Validator
Comprehensive validation framework for temporal model testing
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from .temporal_data_splitter import TemporalDataSplitter


class OutOfSampleValidator:
    """
    Comprehensive out-of-sample validation framework
    
    Handles temporal splitting, feature generation, model testing,
    and statistical validation for financial ML models.
    """
    
    def __init__(self, config, pattern_engine, pattern_target_generator):
        """
        Initialize out-of-sample validator
        
        Args:
            config: Model configuration object
            pattern_engine: Feature calculation engine
            pattern_target_generator: Target generation engine
        """
        self.config = config
        self.pattern_engine = pattern_engine
        self.pattern_target_generator = pattern_target_generator
        
        # Initialize temporal splitter
        self.temporal_splitter = TemporalDataSplitter(
            gap_months=config.model.out_of_sample_gap_months,
            train_ratio=config.model.temporal_validation_split
        )
        
        # Standard feature names for consistency
        self.feature_names = [
            "price_acceleration", "volume_price_divergence", "volatility_regime_change",
            "return_skewness_7d", "momentum_persistence_7d", "volatility_clustering",
            "trend_exhaustion", "garch_volatility_forecast", "intraday_range_expansion",
            "overnight_gap_behavior", "end_of_day_momentum", "sector_relative_strength",
            "market_beta_instability", "vix_term_structure", "returns_1d", "returns_3d", "returns_7d"
        ]
    
    def prepare_out_of_sample_data(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Prepare out-of-sample testing data with temporal separation
        
        Args:
            ticker_data: Dictionary {ticker: ohlcv_dataframe}
            
        Returns:
            Dictionary with prepared out-of-sample data and metadata
        """
        if not self.config.model.out_of_sample_enabled:
            return {
                "enabled": False, 
                "message": "Out-of-sample testing disabled in configuration"
            }
        
        print(f"\n🔬 PREPARING OUT-OF-SAMPLE DATA:")
        print(f"- Gap: {self.config.model.out_of_sample_gap_months} months")
        print(f"- Train/test split: {self.config.model.temporal_validation_split:.1%}/{1-self.config.model.temporal_validation_split:.1%}")
        
        # Split data temporally
        split_results = self.temporal_splitter.split_multi_ticker_data(ticker_data)
        
        # Process successful splits
        prepared_data = {}
        processing_stats = {"success": 0, "failed_features": 0, "failed_targets": 0, "insufficient_data": 0}
        
        for ticker in split_results["successful_tickers"]:
            ticker_split = split_results["splits"][ticker]
            
            try:
                # Extract train and test data
                train_data = ticker_split["train_data"]
                test_data = ticker_split["test_data"]
                
                # Calculate features for test data
                test_features_dict = self.pattern_engine.calculate_portfolio_features({ticker: test_data})
                
                if ticker not in test_features_dict or test_features_dict[ticker] is None:
                    print(f"⚠️  {ticker}: Failed to calculate features")
                    processing_stats["failed_features"] += 1
                    continue
                
                features_df = test_features_dict[ticker]
                
                # Generate pattern targets for test period
                pattern_targets_dict = self.pattern_target_generator.generate_all_pattern_targets(features_df)
                target_values = pattern_targets_dict["pattern_confidence_score"]
                
                if len(target_values) == 0:
                    print(f"⚠️  {ticker}: Failed to generate targets")
                    processing_stats["failed_targets"] += 1
                    continue
                
                # Check available features
                available_features = [f for f in self.feature_names if f in features_df.columns]
                
                if len(available_features) < 10:
                    print(f"⚠️  {ticker}: Insufficient features ({len(available_features)}/17)")
                    processing_stats["insufficient_data"] += 1
                    continue
                
                # Store prepared data
                prepared_data[ticker] = {
                    "train_data": train_data,
                    "test_data": test_data,
                    "test_features": features_df,
                    "test_targets": target_values,
                    "available_features": available_features,
                    "metadata": ticker_split["metadata"]
                }
                
                processing_stats["success"] += 1
                print(f"✅ {ticker}: {len(test_data)} test samples, {len(available_features)} features")
                
            except Exception as e:
                print(f"❌ {ticker}: Processing error - {e}")
                processing_stats["failed_features"] += 1
                continue
        
        print(f"\n📊 PREPARATION SUMMARY:")
        print(f"- Successful: {processing_stats['success']} tickers")
        print(f"- Failed (features): {processing_stats['failed_features']}")
        print(f"- Failed (targets): {processing_stats['failed_targets']}")
        print(f"- Insufficient data: {processing_stats['insufficient_data']}")
        
        return {
            "enabled": True,
            "data": prepared_data,
            "split_results": split_results,
            "processing_stats": processing_stats,
            "metadata": {
                "gap_months": self.config.model.out_of_sample_gap_months,
                "temporal_split": self.config.model.temporal_validation_split,
                "preparation_timestamp": pd.Timestamp.now(),
                "total_tickers_prepared": len(prepared_data)
            }
        }
    
    def run_out_of_sample_validation(self, trained_model, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run out-of-sample validation on trained model
        
        Args:
            trained_model: Trained TensorFlow/Keras model
            prepared_data: Output from prepare_out_of_sample_data
            
        Returns:
            Comprehensive out-of-sample validation results
        """
        if not prepared_data.get("enabled", False):
            return {"enabled": False, "message": "Out-of-sample testing not enabled"}
        
        if not prepared_data.get("data"):
            return {"enabled": True, "error": "No prepared data available"}
        
        print(f"\n🧪 RUNNING OUT-OF-SAMPLE VALIDATION:")
        
        ticker_results = {}
        all_predictions = []
        all_targets = []
        sequence_generation_stats = {"success": 0, "insufficient_sequences": 0, "errors": 0}
        
        for ticker, ticker_data in prepared_data["data"].items():
            try:
                # Generate test sequences using same parameters as training
                X_test, y_test = self._generate_test_sequences(
                    features_df=ticker_data["test_features"],
                    feature_columns=ticker_data["available_features"],
                    pattern_targets=ticker_data["test_targets"]
                )
                
                if len(X_test) < 5:  # Need minimum sequences for meaningful testing
                    print(f"⚠️  {ticker}: Insufficient test sequences ({len(X_test)})")
                    sequence_generation_stats["insufficient_sequences"] += 1
                    continue
                
                # Generate predictions
                predictions = trained_model.predict(X_test, verbose=0).flatten()
                
                # Calculate comprehensive metrics
                ticker_metrics = self._calculate_validation_metrics(y_test, predictions)
                
                # Store results
                ticker_results[ticker] = {
                    "test_sequences": len(X_test),
                    "test_period": ticker_data["metadata"]["test_period"],
                    "gap_days": ticker_data["metadata"]["gap_days"],
                    **ticker_metrics
                }
                
                # Accumulate for overall metrics
                all_predictions.extend(predictions)
                all_targets.extend(y_test)
                
                sequence_generation_stats["success"] += 1
                print(f"📊 {ticker}: {len(X_test)} sequences, correlation={ticker_metrics['correlation']:.3f}")
                
            except Exception as e:
                print(f"❌ {ticker}: Validation error - {e}")
                sequence_generation_stats["errors"] += 1
                continue
        
        # Calculate overall out-of-sample performance
        if len(all_predictions) > 0:
            overall_metrics = self._calculate_validation_metrics(all_targets, all_predictions)
            
            # Enhanced statistical tests
            statistical_tests = self._run_statistical_tests(all_targets, all_predictions)
            
            results = {
                "enabled": True,
                "success": True,
                "ticker_results": ticker_results,
                "overall_metrics": overall_metrics,
                "statistical_tests": statistical_tests,
                "sequence_stats": sequence_generation_stats,
                "summary": {
                    "total_test_samples": len(all_predictions),
                    "successful_tickers": len(ticker_results),
                    "overall_correlation": overall_metrics["correlation"],
                    "statistical_significance": statistical_tests["pearson_test"]["p_value"] < 0.05
                },
                "metadata": prepared_data["metadata"]
            }
            
            self._print_validation_summary(results)
            return results
        
        else:
            return {
                "enabled": True,
                "success": False,
                "error": "No valid test sequences generated",
                "sequence_stats": sequence_generation_stats,
                "metadata": prepared_data["metadata"]
            }
    
    def _generate_test_sequences(self, features_df: pd.DataFrame, feature_columns: List[str], 
                                pattern_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate test sequences using same logic as training
        
        Args:
            features_df: Features DataFrame
            feature_columns: List of feature column names  
            pattern_targets: Target values array
            
        Returns:
            Tuple of (X_test, y_test) arrays
        """
        # Use validation stride for clean testing (no overlap)
        stride = self.config.model.validation_stride
        sequence_length = self.config.model.lookback_window
        
        # Extract and clean feature matrix
        feature_matrix = features_df[feature_columns].values
        feature_df_clean = pd.DataFrame(feature_matrix, columns=feature_columns)
        feature_df_clean = feature_df_clean.ffill().bfill().fillna(0.0)
        feature_matrix = feature_df_clean.values
        
        # Generate sequences with temporal gap (same as training)
        prediction_horizon = 1
        temporal_gap = 0  # Targets are already leak-free
        total_future_offset = prediction_horizon + temporal_gap
        
        X_sequences = []
        y_targets = []
        
        max_start_idx = len(feature_matrix) - sequence_length - total_future_offset
        
        for i in range(0, max_start_idx, stride):
            sequence = feature_matrix[i : i + sequence_length]
            target_idx = i + sequence_length + total_future_offset - 1
            
            if target_idx < len(pattern_targets):
                target = pattern_targets[target_idx]
                X_sequences.append(sequence)
                y_targets.append(target)
        
        return np.array(X_sequences), np.array(y_targets)
    
    def _calculate_validation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive validation metrics
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of validation metrics
        """
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"correlation": 0.0, "mae": float('inf'), "rmse": float('inf'), 
                   "predictions_variance": 0.0, "r_squared": 0.0}
        
        # Basic metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Correlation (handle constant predictions)
        if np.var(y_pred) > 1e-10:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Prediction statistics
        pred_var = np.var(y_pred)
        pred_mean = np.mean(y_pred)
        pred_std = np.std(y_pred)
        
        return {
            "correlation": float(correlation),
            "mae": float(mae),
            "rmse": float(rmse),
            "r_squared": float(r_squared),
            "predictions_variance": float(pred_var),
            "predictions_mean": float(pred_mean),
            "predictions_std": float(pred_std)
        }
    
    def _run_statistical_tests(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Run comprehensive statistical significance tests
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of statistical test results
        """
        tests = {}
        
        try:
            # Pearson correlation test
            pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
            tests["pearson_test"] = {
                "statistic": float(pearson_r),
                "p_value": float(pearson_p),
                "significant": pearson_p < 0.05
            }
        except:
            tests["pearson_test"] = {"statistic": 0.0, "p_value": 1.0, "significant": False}
        
        try:
            # Spearman rank correlation test  
            spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
            tests["spearman_test"] = {
                "statistic": float(spearman_r),
                "p_value": float(spearman_p),
                "significant": spearman_p < 0.05
            }
        except:
            tests["spearman_test"] = {"statistic": 0.0, "p_value": 1.0, "significant": False}
        
        try:
            # Permutation test for correlation significance
            n_permutations = 1000
            observed_corr = np.corrcoef(y_true, y_pred)[0, 1] if np.var(y_pred) > 1e-10 else 0.0
            
            perm_correlations = []
            for _ in range(n_permutations):
                y_perm = np.random.permutation(y_pred)
                if np.var(y_perm) > 1e-10:
                    perm_corr = np.corrcoef(y_true, y_perm)[0, 1]
                    if not np.isnan(perm_corr):
                        perm_correlations.append(abs(perm_corr))
            
            if perm_correlations:
                p_value_perm = np.mean([pc >= abs(observed_corr) for pc in perm_correlations])
                tests["permutation_test"] = {
                    "observed_correlation": float(observed_corr),
                    "p_value": float(p_value_perm),
                    "significant": p_value_perm < 0.05,
                    "n_permutations": n_permutations
                }
        except:
            tests["permutation_test"] = {"observed_correlation": 0.0, "p_value": 1.0, "significant": False}
        
        return tests
    
    def _print_validation_summary(self, results: Dict[str, Any]) -> None:
        """
        Print comprehensive validation summary
        
        Args:
            results: Validation results dictionary
        """
        overall = results["overall_metrics"]
        stats_tests = results["statistical_tests"]
        summary = results["summary"]
        
        print(f"\n📈 OUT-OF-SAMPLE VALIDATION RESULTS:")
        print(f"- Overall correlation: {overall['correlation']:.4f}")
        print(f"- R-squared: {overall['r_squared']:.4f}")
        print(f"- MAE: {overall['mae']:.4f}")
        print(f"- RMSE: {overall['rmse']:.4f}")
        print(f"- Total test samples: {summary['total_test_samples']:,}")
        print(f"- Successful tickers: {summary['successful_tickers']}")
        
        print(f"\n📊 STATISTICAL SIGNIFICANCE:")
        pearson = stats_tests.get("pearson_test", {})
        print(f"- Pearson test: r={pearson.get('statistic', 0):.4f}, p={pearson.get('p_value', 1):.4f}")
        print(f"- Significant: {'✅ YES' if pearson.get('significant', False) else '❌ NO'}")
        
        spearman = stats_tests.get("spearman_test", {})
        print(f"- Spearman test: ρ={spearman.get('statistic', 0):.4f}, p={spearman.get('p_value', 1):.4f}")
        
        perm = stats_tests.get("permutation_test", {})
        if perm.get("n_permutations", 0) > 0:
            print(f"- Permutation test: p={perm.get('p_value', 1):.4f} (n={perm.get('n_permutations', 0)})")
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive out-of-sample validation report
        
        Args:
            results: Validation results from run_out_of_sample_validation
            
        Returns:
            Formatted validation report
        """
        if not results.get("success", False):
            return f"Out-of-Sample Validation Report\n{'='*50}\nValidation failed: {results.get('error', 'Unknown error')}"
        
        overall = results["overall_metrics"]
        summary = results["summary"]
        metadata = results["metadata"]
        
        report = f"""
Out-of-Sample Validation Report
{'='*50}

Configuration:
- Temporal gap: {metadata['gap_months']} months
- Train/test split: {metadata['temporal_split']:.1%} / {1-metadata['temporal_split']:.1%}
- Validation timestamp: {metadata['preparation_timestamp']}

Overall Performance:
- Correlation: {overall['correlation']:.6f}
- R-squared: {overall['r_squared']:.6f}
- Mean Absolute Error: {overall['mae']:.6f}
- Root Mean Square Error: {overall['rmse']:.6f}
- Prediction Variance: {overall['predictions_variance']:.6f}

Sample Statistics:
- Total test samples: {summary['total_test_samples']:,}
- Successful tickers: {summary['successful_tickers']}
- Statistical significance: {'Yes' if summary['statistical_significance'] else 'No'}

Ticker-Level Results:
"""
        
        # Add top-performing tickers
        ticker_results = results.get("ticker_results", {})
        sorted_tickers = sorted(ticker_results.items(), 
                              key=lambda x: abs(x[1]['correlation']), reverse=True)
        
        for ticker, metrics in sorted_tickers[:10]:  # Top 10
            report += f"\n{ticker}:"
            report += f"  - Correlation: {metrics['correlation']:.4f}"
            report += f"  - Test samples: {metrics['test_sequences']}"
            report += f"  - Test period: {metrics['test_period']}"
            report += f"  - Gap days: {metrics['gap_days']}"
        
        # Statistical tests summary
        stats_tests = results.get("statistical_tests", {})
        if stats_tests:
            report += f"\n\nStatistical Tests:"
            for test_name, test_result in stats_tests.items():
                report += f"\n{test_name.replace('_', ' ').title()}:"
                report += f"  - Statistic: {test_result.get('statistic', 0):.4f}"
                report += f"  - P-value: {test_result.get('p_value', 1):.4f}"
                report += f"  - Significant: {'Yes' if test_result.get('significant', False) else 'No'}"
        
        return report