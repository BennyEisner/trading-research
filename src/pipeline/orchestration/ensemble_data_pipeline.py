#!/usr/bin/env python3

"""
End-to-end data pipeline for ensemble strategy execution
Orchestrates: Raw Data → Feature Engineering → Strategy Validation → Ensemble Execution
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

from ...features.processors.price_features import PriceFeaturesProcessor
from ...features.processors.technical_indicators import TechnicalIndicatorsProcessor
from ...validation.pipeline_validator import PipelineValidator
from ...strategies.adapters.data_adapter import StrategyDataAdapter
from ...strategies.core.compatibility_checker import FeatureValidationError, StrategyCompatibilityChecker
from ..coordination.ensemble_coordinator import EnsembleCoordinator
from ...strategies.validation.strategy_validator import StrategyValidator
from ...strategies.ensemble import EnsembleManager, EnsembleConfig
from ...strategies.base import BaseStrategy


class PipelineError(Exception):
    """Raised when pipeline execution fails"""
    pass


class EnsembleDataPipeline:
    """
    Complete data pipeline for ensemble strategy execution
    
    Pipeline stages:
    1. Raw OHLCV data validation
    2. Feature engineering (price + technical indicators)
    3. Strategy data preparation and validation
    4. Ensemble signal generation
    5. Results validation and output
    """
    
    def __init__(self,
                 strategies: List[BaseStrategy],
                 ensemble_config: Optional[EnsembleConfig] = None,
                 data_adapter: Optional[StrategyDataAdapter] = None,
                 strategy_validator: Optional[StrategyValidator] = None,
                 pipeline_validator: Optional[PipelineValidator] = None,
                 compatibility_checker: Optional[StrategyCompatibilityChecker] = None,
                 ensemble_coordinator: Optional[EnsembleCoordinator] = None,
                 validation_enabled: bool = True):
        """
        Initialize ensemble data pipeline
        
        Args:
            strategies: List of strategy instances for ensemble
            ensemble_config: Configuration for ensemble manager
            data_adapter: Data adapter for data formatting
            strategy_validator: Strategy validator for comprehensive validation
            pipeline_validator: Pipeline validator for data quality validation
            compatibility_checker: Strategy compatibility checker
            ensemble_coordinator: Ensemble coordination manager
            validation_enabled: Whether to perform validation steps
        """
        self.strategies = strategies
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.data_adapter = data_adapter or StrategyDataAdapter()
        self.strategy_validator = strategy_validator or StrategyValidator()
        self.pipeline_validator = pipeline_validator or PipelineValidator()
        self.compatibility_checker = compatibility_checker or StrategyCompatibilityChecker()
        self.ensemble_coordinator = ensemble_coordinator or EnsembleCoordinator(self.compatibility_checker)
        self.validation_enabled = validation_enabled
        
        # Initialize processors
        self.price_processor = PriceFeaturesProcessor()
        self.tech_processor = TechnicalIndicatorsProcessor()
        
        # Initialize ensemble manager
        self.ensemble_manager = EnsembleManager(self.ensemble_config)
        for strategy in self.strategies:
            self.ensemble_manager.register_strategy(strategy)
        
        # Pipeline execution results
        self.execution_results = {}
        self.pipeline_metrics = {}
        
    def execute(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute complete pipeline - thin orchestration layer
        
        Args:
            raw_data: Raw OHLCV data
            
        Returns:
            Complete pipeline results dictionary
        """
        pipeline_start = datetime.now()
        
        try:
            self._execute_data_validation(raw_data)
            engineered_data = self._execute_feature_engineering(raw_data)
            self._execute_strategy_preparation(engineered_data)
            ensemble_signals, individual_signals = self._execute_signal_generation(engineered_data)
            
            return self._package_results(
                raw_data, engineered_data, ensemble_signals, 
                individual_signals, pipeline_start
            )
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.execution_results['error'] = error_msg
            raise PipelineError(error_msg) from e
            
    def _execute_data_validation(self, raw_data: pd.DataFrame) -> None:
        """Execute data validation stage"""
        if not self.validation_enabled:
            return
            
        # Basic OHLCV structure validation
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in raw_data.columns]
        if missing_cols:
            raise PipelineError(f"Missing required OHLCV columns: {missing_cols}")
        
        # Basic data quality checks
        if len(raw_data) == 0:
            raise PipelineError("Empty dataset provided")
        
        # Check OHLC consistency
        invalid_ohlc = (
            (raw_data["high"] < raw_data["low"]) |
            (raw_data["high"] < raw_data["open"]) |
            (raw_data["high"] < raw_data["close"]) |
            (raw_data["low"] > raw_data["open"]) |
            (raw_data["low"] > raw_data["close"])
        )
        
        if invalid_ohlc.sum() > 0:
            warnings.warn(f"Invalid OHLC relationships in {invalid_ohlc.sum()} records")
        
        self.execution_results['raw_data_validation'] = {
            'timestamp': datetime.now().isoformat(),
            'record_count': len(raw_data),
            'date_range': f"{raw_data['date'].min()} to {raw_data['date'].max()}" if 'date' in raw_data.columns else 'Unknown',
            'status': 'PASSED'
        }
        
    def _execute_feature_engineering(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Execute feature engineering stage"""
        try:
            # Process price features then technical indicators
            data_with_price_features = self.price_processor.calculate(raw_data)
            engineered_data = self.tech_processor.calculate(data_with_price_features)
            
            # Validate feature engineering if enabled
            if self.validation_enabled:
                feature_columns = [col for col in engineered_data.columns 
                                 if col not in raw_data.columns]
                is_valid, issues = self.pipeline_validator.validate_feature_data(
                    engineered_data, feature_columns
                )
                if not is_valid:
                    warnings.warn(f"Feature engineering validation issues: {issues}")
            
            self.execution_results['feature_engineering'] = {
                'timestamp': datetime.now().isoformat(),
                'original_columns': len(raw_data.columns),
                'engineered_columns': len(engineered_data.columns),
                'new_features': len(engineered_data.columns) - len(raw_data.columns),
                'status': 'COMPLETED'
            }
            
            return engineered_data
            
        except Exception as e:
            raise PipelineError(f"Feature engineering failed: {str(e)}") from e
    
    def _execute_strategy_preparation(self, engineered_data: pd.DataFrame) -> None:
        """Execute strategy preparation and validation stage"""
        if not self.validation_enabled:
            return
            
        try:
            # Validate backtest readiness
            backtest_valid, backtest_issues = self.strategy_validator.validate_backtest_readiness(
                engineered_data, self.strategies
            )
            
            # Filter warnings from errors
            backtest_error_issues = [
                issue for issue in backtest_issues 
                if not (issue.startswith("Warning:") or "feature issue" in issue.lower())
            ]
            
            if backtest_error_issues:
                warnings.warn(f"Backtest readiness issues: {backtest_error_issues}")
            
            # Validate ensemble strategies
            is_valid, issues = self.strategy_validator.validate_ensemble_strategies(
                engineered_data, self.strategies
            )
            
            if not is_valid:
                error_issues = [issue for issue in issues if not issue.startswith("Warning:")]
                if error_issues:
                    raise PipelineError(f"Strategy validation failed: {error_issues}")
            
            # Validate individual strategy signals
            strategy_signal_results = {}
            for strategy in self.strategies:
                signal_valid, signal_issues = self.strategy_validator.validate_strategy_signals(
                    engineered_data, strategy
                )
                
                strategy_signal_results[strategy.name] = {
                    'valid': signal_valid,
                    'issues': signal_issues
                }
                
                if not signal_valid:
                    warnings.warn(f"Strategy {strategy.name} signal validation issues: {signal_issues}")
            
            self.execution_results['strategy_validation'] = {
                'timestamp': datetime.now().isoformat(),
                'backtest_readiness': {'valid': backtest_valid, 'issues': backtest_issues},
                'ensemble_validation': {'valid': is_valid, 'issues': issues},
                'individual_strategy_results': strategy_signal_results,
                'total_strategies': len(self.strategies),
                'valid_strategies': sum(1 for r in strategy_signal_results.values() if r['valid']),
                'status': 'COMPLETED'
            }
            
        except Exception as e:
            raise PipelineError(f"Strategy validation failed: {str(e)}") from e
    
    def _execute_signal_generation(self, engineered_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Execute signal generation stage"""
        try:
            # Format data and coordinate ensemble execution
            prepared_data = self.data_adapter.format_ensemble_data(engineered_data, self.strategies)
            valid_strategies, coordination_results = self.ensemble_coordinator.coordinate_ensemble_execution(
                prepared_data, self.strategies
            )
            
            # Generate individual strategy signals
            individual_signals = {}
            for strategy in valid_strategies:
                try:
                    signals = strategy.generate_signals(prepared_data)
                    individual_signals[strategy.name] = signals
                except Exception as e:
                    warnings.warn(f"Failed to generate signals for {strategy.name}: {str(e)}")
            
            # Generate ensemble signals
            ensemble_signals = self.ensemble_manager.generate_ensemble_signals(prepared_data)
            
            self.execution_results['signal_generation'] = {
                'timestamp': datetime.now().isoformat(),
                'ensemble_signals_generated': len(ensemble_signals),
                'individual_strategies_executed': len(individual_signals),
                'valid_strategies': coordination_results['valid_strategies'],
                'invalid_strategies': [s['strategy_name'] for s in coordination_results['invalid_strategies']],
                'status': 'COMPLETED'
            }
            
            return ensemble_signals, individual_signals
            
        except Exception as e:
            raise PipelineError(f"Signal generation failed: {str(e)}") from e
    
    def _package_results(self, 
                        raw_data: pd.DataFrame,
                        engineered_features: pd.DataFrame,
                        ensemble_signals: pd.DataFrame,
                        individual_signals: Dict[str, pd.DataFrame],
                        pipeline_start: datetime) -> Dict[str, Any]:
        """Stage 5: Package and validate final results"""
        
        pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
        
        # Calculate pipeline metrics
        self.pipeline_metrics = {
            'execution_time_seconds': pipeline_duration,
            'data_points_processed': len(raw_data),
            'throughput_records_per_second': len(raw_data) / pipeline_duration if pipeline_duration > 0 else 0,
            'features_engineered': len(engineered_features.columns) - len(raw_data.columns),
            'strategies_executed': len(individual_signals),
            'ensemble_signals_generated': len(ensemble_signals),
            'pipeline_stages_completed': len(self.execution_results),
            'validation_enabled': self.validation_enabled
        }
        
        # Prepare final results
        results = {
            'ensemble_signals': ensemble_signals,
            'individual_signals': individual_signals,
            'engineered_features': engineered_features,
            'pipeline_metrics': self.pipeline_metrics,
            'execution_results': self.execution_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add validation summary if enabled
        if self.validation_enabled:
            try:
                validation_summary = self.strategy_validator.get_validation_summary()
                results['validation_summary'] = validation_summary
            except Exception as e:
                # Don't fail pipeline due to validation summary issues
                results['validation_summary'] = {'error': f'Failed to generate validation summary: {str(e)}'}
                warnings.warn(f"Failed to generate validation summary: {str(e)}")
        
        return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline execution status"""
        return {
            'strategies_registered': len(self.strategies),
            'ensemble_config': self.ensemble_config.model_dump(),
            'validation_enabled': self.validation_enabled,
            'execution_results': self.execution_results,
            'pipeline_metrics': self.pipeline_metrics,
            'last_execution': self.execution_results.get('timestamp', 'Never')
        }
    
    def reset_pipeline(self) -> None:
        """Reset pipeline state for new execution"""
        self.execution_results = {}
        self.pipeline_metrics = {}
        self.strategy_validator.reset_validation_results()
        self.data_adapter.reset_validation_results()
        self.pipeline_validator.reset_validation_results()


def create_ensemble_pipeline(strategies: List[BaseStrategy],
                           ensemble_config: Optional[EnsembleConfig] = None,
                           **kwargs) -> EnsembleDataPipeline:
    """
    Convenience function to create ensemble data pipeline
    
    Args:
        strategies: List of strategy instances
        ensemble_config: Optional ensemble configuration
        **kwargs: Additional pipeline configuration
        
    Returns:
        Configured EnsembleDataPipeline instance
    """
    return EnsembleDataPipeline(
        strategies=strategies,
        ensemble_config=ensemble_config,
        **kwargs
    )


class PipelineBatchProcessor:
    """
    Batch processor for running pipeline on multiple datasets
    Useful for backtesting and bulk processing
    """
    
    def __init__(self, pipeline: EnsembleDataPipeline):
        self.pipeline = pipeline
        self.batch_results = []
        
    def process_batch(self, datasets: List[Tuple[str, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Process multiple datasets through the pipeline
        
        Args:
            datasets: List of (identifier, dataframe) tuples
            
        Returns:
            List of pipeline execution results
        """
        batch_results = []
        
        for identifier, data in datasets:
            try:
                # Reset pipeline for clean execution
                self.pipeline.reset_pipeline()
                
                # Execute pipeline
                result = self.pipeline.execute(data)
                result['dataset_identifier'] = identifier
                result['batch_status'] = 'SUCCESS'
                
                batch_results.append(result)
                
            except Exception as e:
                error_result = {
                    'dataset_identifier': identifier,
                    'batch_status': 'FAILED',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                batch_results.append(error_result)
        
        self.batch_results = batch_results
        return batch_results
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get summary of batch processing results"""
        if not self.batch_results:
            return {'status': 'No batch processing completed'}
        
        successful = [r for r in self.batch_results if r.get('batch_status') == 'SUCCESS']
        failed = [r for r in self.batch_results if r.get('batch_status') == 'FAILED']
        
        return {
            'total_datasets': len(self.batch_results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.batch_results) if self.batch_results else 0,
            'average_execution_time': np.mean([
                r.get('pipeline_metrics', {}).get('execution_time_seconds', 0) 
                for r in successful
            ]) if successful else 0,
            'failed_datasets': [r['dataset_identifier'] for r in failed]
        }