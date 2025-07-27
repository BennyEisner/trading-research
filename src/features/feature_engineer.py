#!/usr/bin/env python3

"""
Main Feature Engineering Orchestrator
"""

import warnings
import pandas as pd
from typing import List, Dict, Optional

from .base import FEATURE_REGISTRY
from .utils import DataValidator, FeatureScaler, prepare_lstm_sequences, clean_financial_data
# from .selectors import EnsembleFeatureSelector  # Removed - using simple selector now
from .processors import (
    PriceFeaturesProcessor, TechnicalIndicatorsProcessor, 
    VolumeFeaturesProcessor, VolatilityFeaturesProcessor,
    MomentumFeaturesProcessor, MarketFeaturesProcessor
)
from .processors.cross_sectional_features import CrossSectionalFeaturesProcessor
from .processors.macro_features import MacroFeaturesProcessor

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Main feature engineering orchestrator using modular processors
    """
    
    def __init__(self, symbol: str = "AAPL", market_data=None, mag7_data=None, all_ticker_data=None):
        self.symbol = symbol
        self.market_data = market_data
        self.all_ticker_data = all_ticker_data
        self.mag7_data = mag7_data
        
        # Initialize components
        self.data_validator = DataValidator()
        self.feature_scaler = FeatureScaler()
        # self.feature_selector = EnsembleFeatureSelector()  # Removed - using simple selector now
        
        # Initialize processors
        self.processors = {
            'price': PriceFeaturesProcessor(),
            'technical': TechnicalIndicatorsProcessor(),
            'volume': VolumeFeaturesProcessor(), 
            'volatility': VolatilityFeaturesProcessor(),
            'momentum': MomentumFeaturesProcessor(),
            'market': MarketFeaturesProcessor(symbol, market_data, mag7_data),
            'cross_sectional': CrossSectionalFeaturesProcessor(all_ticker_data),
            'macro': MacroFeaturesProcessor()
        }
        
        # Processing order (respects dependencies)
        # NOTE: Temporarily disabled 'market', 'macro', 'cross_sectional' due to NaN issues
        self.processing_order = [
            'price',      # Basic price features first
            'technical',  # Technical indicators (need EMAs from price)
            'volume',     # Volume features
            'momentum',   # Momentum features
            'volatility', # Volatility features (need daily_return)
            # 'market',     # Market context features - DISABLED (creates all NaNs)
            # 'macro',      # Macro-economic features - DISABLED (creates all NaNs)
            # 'cross_sectional'  # Cross-sectional features - DISABLED (creates all NaNs)
        ]
        
        # Results storage
        self.feature_importance = {}
        self.validation_results = {}
        self.processing_stats = {}
        
    def calculate_all_features(self, data: pd.DataFrame, 
                             validate_data: bool = True,
                             clean_data: bool = True) -> pd.DataFrame:
        """
        Calculate comprehensive feature set using modular processors
        
        Args:
            data: Input DataFrame with OHLCV data
            validate_data: Whether to validate input data
            clean_data: Whether to clean input data
            
        Returns:
            DataFrame with engineered features
        """
        print(f"Calculating features for {self.symbol}")
        
        # Data validation and cleaning
        if validate_data:
            self.validation_results = self.data_validator.validate_ohlcv_data(data, self.symbol)
            if self.validation_results['data_quality_score'] < 0.5:
                print(f"Warning: Low data quality score: {self.validation_results['data_quality_score']:.2f}")
        
        if clean_data:
            data, cleaning_stats = clean_financial_data(data, self.symbol)
            print(f"Data cleaning: {cleaning_stats['data_retention']:.1%} retention rate")
            
        result_data = data.copy()
        self.processing_stats = {}
        
        # Run processors in order
        for processor_name in self.processing_order:
            processor = self.processors[processor_name]
            
            try:
                print(f"Processing: {processor.name}")
                original_cols = len(result_data.columns)
                
                result_data = processor.calculate(result_data)
                
                new_cols = len(result_data.columns) - original_cols
                self.processing_stats[processor_name] = {
                    'features_added': new_cols,
                    'feature_names': processor.get_feature_names()
                }
                
                print(f"  Added {new_cols} features")
                
            except Exception as e:
                print(f"Warning: {processor_name} processor failed: {e}")
                continue
        
        print(f"Feature calculation complete. Total features: {len(result_data.columns)}")
        return result_data
    
    def select_best_features(self, data: pd.DataFrame,
                           target_col: str = "daily_return",
                           max_features: int = 50,
                           method: str = "ensemble") -> List[str]:
        """
        Select best features for modeling
        
        Args:
            data: DataFrame with all features
            target_col: Target column name
            max_features: Maximum features to select
            method: Selection method ('ensemble', 'rf', 'lasso', etc.)
            
        Returns:
            List of selected feature names
        """
        print(f"Selecting best features using {method} method...")
        
        # Get all numeric features (exclude OHLCV and target)
        exclude_features = ["open", "high", "low", "close", "volume", target_col]
        numeric_features = data.select_dtypes(include=['number']).columns.tolist()
        candidate_features = [f for f in numeric_features if f not in exclude_features]
        
        if len(candidate_features) == 0:
            print("Warning: No candidate features found")
            return []
        
        # Prepare data for selection
        X = data[candidate_features].fillna(0)
        y = data[target_col].fillna(0)
        
        # Remove rows with NaN target
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 100:
            print("Warning: Insufficient data for feature selection")
            return candidate_features[:max_features]
        
        # Select features based on method
        if method == "ensemble":
            selected_features = self.feature_selector.select_features(
                X, y, candidate_features, max_features
            )
            
            # Store feature importance from ensemble
            self.feature_importance = self.feature_selector.feature_votes
            
        else:
            # Use individual selector methods
            from .selectors import StatisticalFeatureSelector, ModelBasedFeatureSelector
            
            if method in ['f_regression', 'mutual_info']:
                selector = StatisticalFeatureSelector()
                selected_features = selector.select_features(X, y, candidate_features, max_features, method)
            elif method in ['random_forest', 'lasso', 'rfe']:
                selector = ModelBasedFeatureSelector() 
                selected_features = selector.select_features(X, y, candidate_features, max_features, method)
            else:
                raise ValueError(f"Unknown selection method: {method}")
        
        print(f"Selected {len(selected_features)} features")
        return selected_features
    
    def prepare_for_lstm(self, data: pd.DataFrame, 
                        selected_features: List[str],
                        target_col: str = "daily_return",
                        sequence_length: int = 60,
                        scale_features: bool = True) -> tuple:
        """
        Prepare data for LSTM training
        
        Args:
            data: DataFrame with features
            selected_features: List of selected feature names
            target_col: Target column name
            sequence_length: LSTM sequence length
            scale_features: Whether to scale features
            
        Returns:
            Tuple of (X_sequences, y_targets, feature_names)
        """
        print(f"Preparing LSTM data with {len(selected_features)} features...")
        
        # Get available features
        available_features = [f for f in selected_features if f in data.columns]
        if len(available_features) == 0:
            raise ValueError("No selected features found in data")
        
        # Scale features if requested
        lstm_data = data.copy()
        if scale_features:
            lstm_data = self.feature_scaler.fit_transform(
                lstm_data, method="robust", exclude_features=[target_col]
            )
        
        # Create sequences
        X, y = prepare_lstm_sequences(
            lstm_data, available_features, target_col, sequence_length
        )
        
        print(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y, available_features
    
    def get_processing_summary(self) -> pd.DataFrame:
        """Get summary of feature processing results"""
        if not self.processing_stats:
            return pd.DataFrame()
        
        summary_data = []
        for processor, stats in self.processing_stats.items():
            summary_data.append({
                'processor': processor,
                'features_added': stats['features_added'],
                'sample_features': ', '.join(stats['feature_names'][:5])
            })
        
        return pd.DataFrame(summary_data)
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance scores"""
        return self.feature_importance
    
    def get_validation_results(self) -> Dict:
        """Get data validation results"""
        return self.validation_results
    
    def validate_features(self, data: pd.DataFrame) -> List[str]:
        """
        Validate feature quality and identify issues
        
        Args:
            data: DataFrame with features
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for high NaN ratios
        nan_threshold = 0.1
        for col in data.columns:
            nan_ratio = data[col].isna().sum() / len(data)
            if nan_ratio > nan_threshold:
                issues.append(f"Feature '{col}' has {nan_ratio:.2%} missing values")
        
        # Check for zero variance
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if data[col].var() == 0:
                issues.append(f"Feature '{col}' has zero variance")
        
        # Check for highly correlated features
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                issues.append(f"High correlation (>0.95) between: {len(high_corr_pairs)} pairs")
        
        return issues
    
    def create_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features using processor-based system
        This is an alias for calculate_all_features to match expected interface
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with comprehensive features from all processors
        """
        return self.calculate_all_features(data, validate_data=True, clean_data=True)