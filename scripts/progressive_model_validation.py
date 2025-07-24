#!/usr/bin/env python3

"""
Progressive model validation framework to test predictive power
while incrementally adding feature complexity
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.config.config import Config
from src.data.data_loader import DataLoader
from src.models.model_builder import ModelBuilder
from src.evaluation.cross_validator import ModelEvaluator
from src.features.simple_selector import SimpleCategorySelector
from src.features.feature_engineer import FeatureEngineer
from src.utils.logging_utils import setup_production_logger
from src.utils.temporal_alignment import TemporalAligner
from src.utils.training_diagnostics import TrainingDiagnostics
from src.features.utils.robust_scaling import RobustFeatureScaler


class ProgressiveModelValidator:
    """Test model performance with increasing feature complexity"""
    
    def __init__(self, config_dict=None):
        self.config_dict = config_dict or {
            "tickers": ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA"],  # 7 tickers for robust validation
            "years_of_data": 10,  # 10 years for sufficient data
            "prediction_horizon": "daily",
            "lookback_window": 30,  # Longer lookback for better patterns  
            "target_features": 12,  # Number of features to select
            "database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db"
        }
        
        self.config = Config(self.config_dict)
        self.data_loader = DataLoader(self.config)  # Add data loader for global selection
        self.logger = setup_production_logger({"log_file": "progressive_validation.log"})
        self.results = {}
        
    def create_baseline_features(self, data):
        """Minimal features: just basic price and volume"""
        features = data.copy()
        
        # Only basic features
        features['daily_return'] = features['close'].pct_change()
        features['volume_change'] = features['volume'].pct_change()
        features['price_change'] = (features['close'] - features['open']) / features['open']
        
        return features.dropna()
    
    def create_simple_features(self, data):
        """Simple technical features"""
        features = self.create_baseline_features(data)
        
        # Add simple moving averages
        for window in [5, 20]:
            features[f'sma_{window}'] = features['close'].rolling(window).mean()
            features[f'price_sma_ratio_{window}'] = features['close'] / features[f'sma_{window}']
        
        # Add volatility
        features['volatility_5'] = features['daily_return'].rolling(5).std()
        features['volatility_20'] = features['daily_return'].rolling(20).std()
        
        return features.dropna()
    
    def create_intermediate_features(self, data):
        """Intermediate technical features"""
        features = self.create_simple_features(data)
        
        # RSI
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = features['close'].ewm(span=12).mean()
        ema26 = features['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma20 = features['close'].rolling(20).mean()
        std20 = features['close'].rolling(20).std()
        features['bb_upper'] = sma20 + (std20 * 2)
        features['bb_lower'] = sma20 - (std20 * 2)
        features['bb_ratio'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        return features.dropna()
    
    def create_advanced_features(self, data):
        """Advanced technical features (without problematic ones)"""
        features = self.create_intermediate_features(data)
        
        # Additional momentum indicators
        features['momentum_10'] = features['close'] / features['close'].shift(10)
        features['momentum_20'] = features['close'] / features['close'].shift(20)
        
        # Williams %R
        high_14 = features['high'].rolling(14).max()
        low_14 = features['low'].rolling(14).min()
        features['williams_r'] = -100 * (high_14 - features['close']) / (high_14 - low_14)
        
        # Volume indicators
        features['volume_sma_10'] = features['volume'].rolling(10).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma_10']
        features['price_volume'] = features['close'] * features['volume']
        
        # Stochastic Oscillator
        features['stoch_k'] = 100 * (features['close'] - low_14) / (high_14 - low_14)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        return features.dropna()

    def _perform_global_feature_selection(self):
        """Perform feature selection once on combined data from all tickers"""
        self.logger.log("Loading combined data from all tickers for global feature selection...")
        
        # Load a sample of data from each ticker for global selection
        combined_data = []
        for ticker in ["AAPL", "MSFT", "GOOG"]:  # Use subset for faster global selection
            try:
                ticker_data = self.data_loader.load_single_ticker_data(ticker, 3)  # 3 years for speed
                ticker_data['ticker'] = ticker
                combined_data.append(ticker_data)
                self.logger.log(f"Loaded {len(ticker_data)} records for {ticker}")
            except Exception as e:
                self.logger.log(f"Failed to load {ticker}: {e}")
                continue
        
        if not combined_data:
            self.logger.log("ERROR: No data loaded for global selection, falling back to basic features")
            self._global_selected_features = ['daily_return', 'volume_ratio', 'price_sma_20_ratio'] # fallback
            return
            
        # Combine all ticker data
        all_data = pd.concat(combined_data, ignore_index=True)
        self.logger.log(f"Combined dataset: {len(all_data)} records")
        
        # Create comprehensive features using processor-based system
        engineer = FeatureEngineer()
        comprehensive_features = engineer.create_comprehensive_features(all_data)
        
        # Fix date column/index ambiguity
        if 'date' in comprehensive_features.index.names:
            comprehensive_features = comprehensive_features.reset_index()  # Move date from index to column
            
        # Remove duplicate date columns if they exist
        if comprehensive_features.columns.duplicated().any():
            comprehensive_features = comprehensive_features.loc[:, ~comprehensive_features.columns.duplicated()]
        
        # Prepare for global selection
        target = comprehensive_features['daily_return'].fillna(0).values
        feature_data = comprehensive_features.drop(columns=['daily_return'])
        numerical_features = feature_data.select_dtypes(include=[np.number])
        
        self.logger.log(f"Global selection from {len(numerical_features.columns)} numerical features")
        
        # Run simple category-based feature selection  
        target_features = self.config_dict.get("target_features", 24)  # 4 per category * 6 categories
        features_per_category = self.config_dict.get("features_per_category", 4)
        selector = SimpleCategorySelector(
            features_per_category=features_per_category,
            total_target_features=target_features
        )
        
        try:
            selected_features = selector.fit_transform(numerical_features, target)
            self._global_selected_features = selected_features.columns.tolist()
            self.logger.log(f"GLOBAL SELECTION COMPLETE: {len(self._global_selected_features)} features selected")
            self.logger.log(f"Selected features: {self._global_selected_features}")
            
        except Exception as e:
            self.logger.log(f"Global selection failed: {e}, using fallback")
            # Fallback to most common features (including mandatory GARCH volatility)
            self._global_selected_features = [
                'garch_volatility', 'volatility_clustering', 'volatility_skew',  # MANDATORY GARCH features first
                'daily_return', 'volume_ratio', 'price_sma_20_ratio', 'volatility_10', 
                'momentum_10', 'rsi', 'trend_regime', 'volume_return', 'hl_ratio',
                'price_volume', 'bb_ratio_20', 'macd'
            ][:target_features]

    def create_comprehensive_selected_features(self, data):
        """Create comprehensive features and select optimal subset using ADVANCED methods"""
        
        # Check if this is the first call (AAPL) - do global selection
        if not hasattr(self, '_global_selected_features'):
            self.logger.log("Running GLOBAL COMPREHENSIVE feature selection for all tickers...")
            self._perform_global_feature_selection()
        
        self.logger.log(f"Applying pre-selected features to ticker data...")
        
        # Create all possible features using comprehensive engineering
        try:
            # Make a clean copy to avoid date column conflicts
            clean_data = data.copy()
            
            # Ensure clean state for feature engineering
            if isinstance(clean_data.index, pd.DatetimeIndex):
                # If index is already datetime, reset it to avoid conflicts
                clean_data = clean_data.reset_index(drop=True)
            
            engineer = FeatureEngineer()
            comprehensive_features = engineer.create_comprehensive_features(clean_data)
            
            # Fix date column/index ambiguity
            if 'date' in comprehensive_features.index.names:
                comprehensive_features = comprehensive_features.reset_index()  # Move date from index to column
                
            # Remove duplicate date columns if they exist
            if comprehensive_features.columns.duplicated().any():
                comprehensive_features = comprehensive_features.loc[:, ~comprehensive_features.columns.duplicated()]
                
            # Ensure we have daily_return column
            if 'daily_return' not in comprehensive_features.columns and 'close' in comprehensive_features.columns:
                comprehensive_features['daily_return'] = comprehensive_features['close'].pct_change()
                
        except Exception as e:
            self.logger.log(f"Error creating comprehensive features: {e}")
            # Fallback to basic features if comprehensive fails
            comprehensive_features = self.create_baseline_selected_features(clean_data)
            
        self.logger.log(f"Created {len(comprehensive_features.columns)} comprehensive candidate features")
        
        # Apply globally selected features to this ticker's data
        available_features = [f for f in self._global_selected_features if f in comprehensive_features.columns and f != 'daily_return']
        
        if len(available_features) < len(self._global_selected_features) - 1:  # -1 for daily_return
            missing = set(self._global_selected_features) - set(available_features) - {'daily_return'}
            self.logger.log(f"WARNING: Missing {len(missing)} globally selected features: {missing}")
        
        # Create result with selected features
        result = pd.DataFrame(index=comprehensive_features.index)
        
        # Add selected features
        for feature in available_features:
            result[feature] = comprehensive_features[feature]
        
        # Add target and date
        result['daily_return'] = comprehensive_features['daily_return']
        if 'date' in comprehensive_features.columns:
            result['date'] = comprehensive_features['date']
            
        self.logger.log(f"Applied {len(available_features)} globally selected features to ticker")
        
        # Smart NaN handling - only drop rows where target is NaN
        result = result.dropna(subset=['daily_return'])
        
        # For feature columns, use forward fill then backward fill for any remaining NaNs
        feature_columns = [col for col in result.columns if col not in ['daily_return', 'date']]
        if feature_columns:
            result[feature_columns] = result[feature_columns].fillna(method='ffill').fillna(method='bfill')
            
        return result
    
    def create_baseline_selected_features(self, data):
        """Create basic features and select best ones"""
        features = data.copy()
        
        # Basic financial features
        features['daily_return'] = features['close'].pct_change()
        features['log_return'] = np.log(features['close'] / features['close'].shift(1))
        features['volume_ratio'] = features['volume'] / features['volume'].rolling(20).mean()
        features['hl_ratio'] = features['high'] / features['low']
        features['volatility_10'] = features['daily_return'].rolling(10).std()
        features['momentum_5'] = features['close'] / features['close'].shift(5)
        features['price_sma_20_ratio'] = features['close'] / features['close'].rolling(20).mean()
        
        # Select from basic features (preserve date)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'ticker']
        basic_features = features.drop(columns=[col for col in exclude_cols if col in features.columns])
        basic_features = basic_features.dropna()
        
        target = basic_features['daily_return'].fillna(0).values
        feature_cols = [col for col in basic_features.columns if col not in ['daily_return', 'date']]
        
        # Use simple Random Forest selection for baseline consistency
        target_num_features = min(6, len(feature_cols))
        
        if len(feature_cols) <= target_num_features:
            selected_feature_names = feature_cols
        else:
            # Use Random Forest for consistency with main selector
            from sklearn.ensemble import RandomForestRegressor
            try:
                rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                rf.fit(basic_features[feature_cols].fillna(0), target)
                
                # Select top features by importance
                feature_importance_pairs = list(zip(feature_cols, rf.feature_importances_))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                selected_feature_names = [feat for feat, _ in feature_importance_pairs[:target_num_features]]
            except:
                # Fallback: just take first n features if RF fails
                selected_feature_names = feature_cols[:target_num_features]
        
        selected_features = basic_features[selected_feature_names]
        
        result = selected_features.copy()
        result['daily_return'] = basic_features['daily_return']
        if 'date' in basic_features.columns:
            result['date'] = basic_features['date']
        
        return result.dropna()
    
    def create_technical_selected_features(self, data):
        """Create technical indicators and select best ones (fixed for consistent feature dimensions)"""
        features = data.copy()
        
        # Technical indicators
        features['daily_return'] = features['close'].pct_change()
        
        # RSI
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = features['close'].ewm(span=12).mean()
        ema26 = features['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma20 = features['close'].rolling(20).mean()
        std20 = features['close'].rolling(20).std()
        features['bb_ratio'] = (features['close'] - (sma20 - std20 * 2)) / ((sma20 + std20 * 2) - (sma20 - std20 * 2))
        
        # Moving averages
        for window in [5, 10, 20]:
            features[f'sma_{window}'] = features['close'].rolling(window).mean()
            features[f'price_sma_ratio_{window}'] = features['close'] / features[f'sma_{window}']
            
        # Volatility
        features['volatility_10'] = features['daily_return'].rolling(10).std()
        features['volatility_20'] = features['daily_return'].rolling(20).std()
        
        # Momentum
        features['momentum_10'] = features['close'] / features['close'].shift(10)
        features['momentum_20'] = features['close'] / features['close'].shift(20)
        
        # Williams %R
        high_14 = features['high'].rolling(14).max()
        low_14 = features['low'].rolling(14).min()
        features['williams_r'] = -100 * (high_14 - features['close']) / (high_14 - low_14)
        
        # Volume features
        features['volume_sma_10'] = features['volume'].rolling(10).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma_10']
        
        # FIX: Use fixed set of best technical features instead of per-ticker selection
        # This ensures all tickers have the same feature dimensions
        fixed_technical_features = [
            'daily_return', 'rsi', 'macd', 'macd_signal', 'bb_ratio',
            'price_sma_ratio_5', 'price_sma_ratio_10', 'price_sma_ratio_20',
            'volatility_10', 'volatility_20', 'momentum_10', 'momentum_20',
            'williams_r', 'volume_ratio'
        ]
        
        # Select only available features and add date if present
        result_features = []
        for feature in fixed_technical_features:
            if feature in features.columns:
                result_features.append(feature)
        
        result = features[result_features].copy()
        if 'date' in features.columns:
            result['date'] = features['date']
        
        return result.dropna()
    
    def create_all_features(self, data):
        """Create all features without selection (for comparison) - FIXED for NaN handling"""
        try:
            # Make a clean copy to avoid date column conflicts
            clean_data = data.copy()
            
            # Ensure clean state for feature engineering
            if isinstance(clean_data.index, pd.DatetimeIndex):
                # If index is already datetime, reset it to avoid conflicts
                clean_data = clean_data.reset_index(drop=True)
            
            engineer = FeatureEngineer()
            all_features = engineer.create_comprehensive_features(clean_data)
            
            # Fix date column/index ambiguity
            if 'date' in all_features.index.names:
                all_features = all_features.reset_index()  # Move date from index to column
                
            # Remove duplicate date columns if they exist
            if all_features.columns.duplicated().any():
                all_features = all_features.loc[:, ~all_features.columns.duplicated()]
                
            # Ensure we have daily_return column
            if 'daily_return' not in all_features.columns and 'close' in all_features.columns:
                all_features['daily_return'] = all_features['close'].pct_change()
            
            # SMART NaN HANDLING instead of aggressive dropna()
            print(f"Before NaN handling: {len(all_features)} records, {len(all_features.columns)} features")
            
            # Analyze NaN distribution
            nan_counts = all_features.isna().sum()
            total_features = len(all_features.columns)
            high_nan_features = nan_counts[nan_counts > len(all_features) * 0.5].index.tolist()
            
            if high_nan_features:
                print(f"Removing {len(high_nan_features)} features with >50% NaN values")
                all_features = all_features.drop(columns=high_nan_features)
            
            # Drop rows only if they have NaN in critical columns or >80% NaN across all features
            critical_cols = ['daily_return', 'close']
            available_critical = [col for col in critical_cols if col in all_features.columns]
            
            if available_critical:
                # Drop rows with NaN in critical columns
                all_features = all_features.dropna(subset=available_critical)
            
            # For remaining columns, use forward fill then backward fill for NaN values
            numeric_cols = all_features.select_dtypes(include=[np.number]).columns
            all_features[numeric_cols] = all_features[numeric_cols].fillna(method='ffill').fillna(method='bfill')
            
            # Final cleanup - only drop rows if >80% of features are NaN
            row_nan_threshold = 0.8
            nan_ratio_per_row = all_features.isna().sum(axis=1) / len(all_features.columns)
            all_features = all_features[nan_ratio_per_row <= row_nan_threshold]
            
            print(f"After smart NaN handling: {len(all_features)} records, {len(all_features.columns)} features")
            
            # Ensure minimum data quality
            if len(all_features) < 100:  # Need at least 100 records for meaningful analysis
                print(f"Warning: Only {len(all_features)} records after NaN handling, using fallback")
                return self.create_baseline_selected_features(data)
            
            # Final validation
            if 'daily_return' not in all_features.columns:
                print("Error: daily_return column missing after processing")
                return self.create_baseline_selected_features(data)
                
            return all_features
            
        except Exception as e:
            print(f"Error in create_all_features: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic features if comprehensive fails
            return self.create_baseline_selected_features(data)
    
    def prepare_sequences(self, features_data, lookback_window):
        """Prepare LSTM sequences from features - ENHANCED with validation"""
        # Select feature columns
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        feature_columns = [col for col in features_data.columns if col not in exclude_cols]
        
        # Validation: Check if we have any features
        if len(feature_columns) == 0:
            print(f"ERROR: No feature columns found. Available columns: {list(features_data.columns)}")
            return np.array([]), np.array([]), 0
        
        # Validation: Check if we have daily_return
        if 'daily_return' not in features_data.columns:
            print(f"ERROR: daily_return column missing. Available columns: {list(features_data.columns)}")
            return np.array([]), np.array([]), 0
        
        print(f"Feature columns selected: {len(feature_columns)} - {feature_columns[:5]}{'...' if len(feature_columns) > 5 else ''}")
        
        # Extract features and target
        feature_matrix = features_data[feature_columns].values
        # CRITICAL FIX: Use NEXT day's return as target to prevent data leakage
        target_values = features_data['daily_return'].fillna(0).values
        
        # Validation: Check matrix shapes
        if feature_matrix.shape[0] == 0:
            print(f"ERROR: Empty feature matrix. Shape: {feature_matrix.shape}")
            return np.array([]), np.array([]), 0
        
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        # CRITICAL: Check for NaN values before scaling
        if np.isnan(feature_matrix).any():
            print(f"WARNING: NaN values found in feature matrix, filling with 0")
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
        if np.isnan(target_values).any():
            print(f"WARNING: NaN values found in target values, filling with 0")
            target_values = np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0)

        # NO SCALING HERE - scaling will be done on splits to prevent data leakage
        # Keep original features for proper temporal splitting
        feature_matrix_scaled = feature_matrix.copy()
        
        # Final validation: Check for NaN values
        if np.isnan(feature_matrix_scaled).any():
            print(f"WARNING: NaN values found in feature matrix, filling with 0")
            feature_matrix_scaled = np.nan_to_num(feature_matrix_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create sequences - predict NEXT day's return using CURRENT and PAST features
        X, y = [], []
        # Fixed: Use i+1 for next day's return since we removed shift(-1) from target construction
        for i in range(lookback_window, len(feature_matrix_scaled) - 1):
            X.append(feature_matrix_scaled[i - lookback_window:i, :])  # Features from days [i-30:i]
            y.append(target_values[i + 1])  # Target is NEXT day's return (no longer pre-shifted)
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        print(f"Sequences created: X={X_array.shape}, y={y_array.shape}")
        
        return X_array, y_array, len(feature_columns)
    
    def test_baseline_models(self, X_flat, y, phase_name):
        """Test simple baseline models for comparison"""
        # Flatten X for traditional ML models
        if len(X_flat.shape) == 3:
            X_flat_reshaped = X_flat.reshape(X_flat.shape[0], -1)
        else:
            X_flat_reshaped = X_flat
        
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        }
        
        baseline_results = {}
        
        # Split data
        train_size = int(0.7 * len(X_flat_reshaped))
        val_size = int(0.15 * len(X_flat_reshaped))
        
        X_train = X_flat_reshaped[:train_size]
        y_train = y[:train_size]
        X_val = X_flat_reshaped[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_val)
                
                # Evaluate
                mae = mean_absolute_error(y_val, y_pred)
                dir_acc = np.mean(np.sign(y_val) == np.sign(y_pred)) * 100
                correlation = np.corrcoef(y_val, y_pred)[0, 1] if len(y_val) > 1 else 0
                
                baseline_results[name] = {
                    "mae": mae,
                    "directional_accuracy": dir_acc,
                    "correlation": correlation
                }
                
                self.logger.log(f"{phase_name} - {name}: MAE={mae:.6f}, Dir={dir_acc:.1f}%, Corr={correlation:.3f}")
                
            except Exception as e:
                self.logger.log(f"{phase_name} - {name}: FAILED - {e}")
                
        return baseline_results

    def test_baseline_models_with_splits(self, X_train, y_train, X_val, y_val, phase_name):
        """Test baseline models with proper train/val splits"""
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        }
        
        baseline_results = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                
                # Evaluate
                mae = mean_absolute_error(y_val, y_pred)
                dir_acc = np.mean(np.sign(y_val) == np.sign(y_pred)) * 100
                correlation = np.corrcoef(y_val, y_pred)[0, 1] if len(y_val) > 1 else 0
                
                baseline_results[name] = {
                    "mae": mae,
                    "directional_accuracy": dir_acc,
                    "correlation": correlation
                }
                
                self.logger.log(f"{phase_name} - {name}: MAE={mae:.6f}, Dir={dir_acc:.1f}%, Corr={correlation:.3f}")
                
            except Exception as e:
                self.logger.log(f"{phase_name} - {name}: FAILED - {e}")
                
        return baseline_results
    
    def test_lstm_model(self, X, y, phase_name, num_features):
        """Test LSTM model performance"""
        try:
            # Split data
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            self.logger.log(f"{phase_name} - LSTM data split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
            
            # Build model
            config = Config({
                **self.config_dict,
                "lstm_units": [32, 16],  # Smaller for faster testing
                "epochs": 30,  # Fewer epochs
                "batch_size": 32,
                "learning_rate": 0.001
            })
            
            model_builder = ModelBuilder(config)
            model = model_builder.build_lstm_model(input_shape=X.shape[1:])
            model = model_builder.compile_model(model)
            
            self.logger.log(f"{phase_name} - LSTM model built: {model.count_params()} parameters")
            
            # Train model
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.get("epochs"),
                batch_size=config.get("batch_size"),
                verbose=0
            )
            training_time = time.time() - start_time
            
            # Evaluate
            evaluator = ModelEvaluator()
            
            val_pred = model.predict(X_val, verbose=0).flatten()
            test_pred = model.predict(X_test, verbose=0).flatten()
            
            val_results = evaluator.evaluate_predictions(y_val, val_pred, "Validation")
            test_results = evaluator.evaluate_predictions(y_test, test_pred, "Test")
            
            lstm_results = {
                "val_mae": val_results['mae'],
                "val_directional_accuracy": val_results['directional_accuracy'],
                "val_correlation": val_results['correlation'],
                "test_mae": test_results['mae'],
                "test_directional_accuracy": test_results['directional_accuracy'],
                "test_correlation": test_results['correlation'],
                "training_time": training_time,
                "parameters": model.count_params(),
                "final_loss": history.history['loss'][-1],
                "final_val_loss": history.history['val_loss'][-1]
            }
            
            self.logger.log(f"{phase_name} - LSTM Results:")
            self.logger.log(f"  Val: MAE={val_results['mae']:.6f}, Dir={val_results['directional_accuracy']:.1f}%, Corr={val_results['correlation']:.3f}")
            self.logger.log(f"  Test: MAE={test_results['mae']:.6f}, Dir={test_results['directional_accuracy']:.1f}%, Corr={test_results['correlation']:.3f}")
            self.logger.log(f"  Training time: {training_time:.1f}s")
            
            return lstm_results
            
        except Exception as e:
            self.logger.log(f"{phase_name} - LSTM FAILED: {e}")
            return None

    def test_lstm_model_with_splits(self, X_train, y_train, X_val, y_val, X_test, y_test, phase_name):
        """Test LSTM model with proper train/val/test splits"""
        try:
            self.logger.log(f"{phase_name} - LSTM data splits: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
            
            # Build model
            config = Config({
                **self.config_dict,
                "lstm_units": [32, 16],  # Smaller for faster testing
                "epochs": 30,  # Fewer epochs
                "batch_size": 32,
                "learning_rate": 0.001
            })
            
            model_builder = ModelBuilder(config)
            model = model_builder.build_lstm_model(input_shape=X_train.shape[1:])
            model = model_builder.compile_model(model)
            
            self.logger.log(f"{phase_name} - LSTM model built: {model.count_params()} parameters")
            
            # Train model
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.get("epochs"),
                batch_size=config.get("batch_size"),
                verbose=0
            )
            training_time = time.time() - start_time
            
            # Evaluate on all splits
            evaluator = ModelEvaluator()
            
            val_pred = model.predict(X_val, verbose=0).flatten()
            test_pred = model.predict(X_test, verbose=0).flatten()
            
            val_results = evaluator.evaluate_predictions(y_val, val_pred, "Validation")
            test_results = evaluator.evaluate_predictions(y_test, test_pred, "Test")
            
            lstm_results = {
                "val_mae": val_results['mae'],
                "val_directional_accuracy": val_results['directional_accuracy'],
                "val_correlation": val_results['correlation'],
                "test_mae": test_results['mae'],
                "test_directional_accuracy": test_results['directional_accuracy'],
                "test_correlation": test_results['correlation'],
                "training_time": training_time,
                "parameters": model.count_params(),
                "final_loss": history.history['loss'][-1],
                "final_val_loss": history.history['val_loss'][-1]
            }
            
            self.logger.log(f"{phase_name} - LSTM Results:")
            self.logger.log(f"  Val: MAE={val_results['mae']:.6f}, Dir={val_results['directional_accuracy']:.1f}%, Corr={val_results['correlation']:.3f}")
            self.logger.log(f"  Test: MAE={test_results['mae']:.6f}, Dir={test_results['directional_accuracy']:.1f}%, Corr={test_results['correlation']:.3f}")
            self.logger.log(f"  Training time: {training_time:.1f}s")
            
            return lstm_results
            
        except Exception as e:
            self.logger.log(f"{phase_name} - LSTM FAILED: {e}")
            import traceback
            self.logger.log(traceback.format_exc())
            return None
    
    def create_time_series_splits(self, all_ticker_data):
        """Create proper time series splits with temporal alignment across all tickers"""
        
        print(f"\n🔧 ENSURING TEMPORAL ALIGNMENT ACROSS TICKERS:")
        print("=" * 70)
        
        # Step 1: Ensure temporal alignment across all tickers
        aligner = TemporalAligner()
        aligned_ticker_data = aligner.align_features_data(all_ticker_data)
        
        # Step 2: Verify perfect alignment
        print(f"\n📊 ALIGNMENT VERIFICATION:")
        alignment_stats = {}
        for ticker, ticker_data in aligned_ticker_data.items():
            sample_count = len(ticker_data)
            feature_count = len([col for col in ticker_data.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'daily_return']])
            alignment_stats[ticker] = {'samples': sample_count, 'features': feature_count}
            print(f"   {ticker}: {sample_count:,} samples × {feature_count} features")
        
        # Verify all tickers have same sample count
        sample_counts = [stats['samples'] for stats in alignment_stats.values()]
        if len(set(sample_counts)) > 1:
            raise ValueError(f"CRITICAL: Tickers have different sample counts after alignment: {sample_counts}")
        
        print(f"✅ ALL TICKERS ALIGNED: {sample_counts[0]:,} samples each")
        
        train_sequences, val_sequences, test_sequences = [], [], []
        train_targets, val_targets, test_targets = [], [], []
        
        for ticker, ticker_data in aligned_ticker_data.items():
            # Handle date column/index ambiguity
            if 'date' in ticker_data.index.names:
                ticker_data = ticker_data.reset_index()  # Move date from index to column
            
            # Sort by date to ensure chronological order
            if 'date' in ticker_data.columns:
                ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            else:
                # If no date column, assume data is already chronologically ordered
                ticker_data = ticker_data.reset_index(drop=True)
            
            # Create sequences for this ticker
            X, y, _ = self.prepare_sequences(ticker_data, self.config_dict["lookback_window"])
            
            if len(X) == 0:
                continue
                
            # Split this ticker's data chronologically with proper gaps to prevent leakage
            lookback_window = self.config_dict.get("lookback_window", 30)
            
            # Calculate split points with gaps to prevent lookback window leakage
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))
            
            # Add gaps between splits to prevent leakage from lookback windows
            train_end = train_size
            val_start = train_end + lookback_window  # Gap to prevent leakage
            val_end = min(val_start + val_size, len(X))
            test_start = val_end + lookback_window  # Gap to prevent leakage
            
            # Ensure we have enough data for each split
            if val_start >= len(X):
                # Not enough data for validation set
                X_train_ticker = X[:train_end]
                y_train_ticker = y[:train_end]
                X_val_ticker = np.array([])
                y_val_ticker = np.array([])
                X_test_ticker = np.array([])
                y_test_ticker = np.array([])
            elif test_start >= len(X):
                # Not enough data for test set
                X_train_ticker = X[:train_end]
                y_train_ticker = y[:train_end]
                X_val_ticker = X[val_start:val_end]
                y_val_ticker = y[val_start:val_end]
                X_test_ticker = np.array([])
                y_test_ticker = np.array([])
            else:
                # All splits possible
                X_train_ticker = X[:train_end]
                y_train_ticker = y[:train_end]
                X_val_ticker = X[val_start:val_end]
                y_val_ticker = y[val_start:val_end]
                X_test_ticker = X[test_start:]
                y_test_ticker = y[test_start:]
            
            # Add to combined pools
            if len(X_train_ticker) > 0:
                train_sequences.append(X_train_ticker)
                train_targets.append(y_train_ticker)
            if len(X_val_ticker) > 0:
                val_sequences.append(X_val_ticker)
                val_targets.append(y_val_ticker)
            if len(X_test_ticker) > 0:
                test_sequences.append(X_test_ticker)
                test_targets.append(y_test_ticker)
        
        # FEATURE DIMENSION VALIDATION before concatenation
        def validate_and_concatenate_with_targets(sequences_list, targets_list, sequence_type):
            """Validate feature dimensions and concatenate sequences WITH matching targets"""
            if not sequences_list or not targets_list:
                return np.array([]), np.array([])
            
            # Check shapes for consistency
            shapes = [seq.shape for seq in sequences_list]
            print(f"{sequence_type} shapes: {shapes}")
            
            if len(set(shapes)) > 1:
                print(f"ERROR: {sequence_type} dimension mismatch!")
                print(f"Different shapes found: {set(shapes)}")
                
                # Find the most common shape
                from collections import Counter
                shape_counts = Counter(shapes)
                target_shape = shape_counts.most_common(1)[0][0]
                print(f"Target shape (most common): {target_shape}")
                
                # Filter sequences AND corresponding targets together
                filtered_sequences = []
                filtered_targets = []
                
                for i, seq in enumerate(sequences_list):
                    if seq.shape == target_shape and i < len(targets_list):
                        filtered_sequences.append(seq)
                        filtered_targets.append(targets_list[i])
                
                print(f"Filtered {len(filtered_sequences)}/{len(sequences_list)} sequences to match target shape")
                
                if filtered_sequences and filtered_targets:
                    return np.concatenate(filtered_sequences, axis=0), np.concatenate(filtered_targets, axis=0)
                else:
                    print(f"ERROR: No sequences match target shape for {sequence_type}")
                    return np.array([]), np.array([])
            
            # All shapes match, safe to concatenate
            return np.concatenate(sequences_list, axis=0), np.concatenate(targets_list, axis=0)
        
        # Combine sequences from all tickers with validation (sequences + targets together)
        X_train, y_train = validate_and_concatenate_with_targets(train_sequences, train_targets, "Train")
        X_val, y_val = validate_and_concatenate_with_targets(val_sequences, val_targets, "Val")  
        X_test, y_test = validate_and_concatenate_with_targets(test_sequences, test_targets, "Test")
        
        # Final validation
        print(f"Final combined shapes: Train={X_train.shape if len(X_train) > 0 else 'empty'}, "
              f"Val={X_val.shape if len(X_val) > 0 else 'empty'}, "
              f"Test={X_test.shape if len(X_test) > 0 else 'empty'}")
        
        # CRITICAL FIX: Apply ROBUST scaling to handle outliers that break gradient learning
        if len(X_train) > 0:
            # Get feature names for better debugging
            feature_names = None
            if all_ticker_data:
                first_ticker = list(all_ticker_data.values())[0]
                exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'daily_return']
                feature_names = [col for col in first_ticker.columns if col not in exclude_cols]
            
            # Use robust scaler to handle extreme outliers
            robust_scaler = RobustFeatureScaler(
                outlier_method="quantile",
                outlier_threshold=0.005,  # Clip extreme 0.5% outliers
                scaler_method="standard"
            )
            
            # Fit robust scaler on training data only
            X_train_scaled = robust_scaler.fit_transform(X_train, feature_names)
            
            # Print scaling summary for debugging
            robust_scaler.print_scaling_summary()
            
            # Apply same robust scaling to validation and test
            if len(X_val) > 0:
                X_val_scaled = robust_scaler.transform(X_val)
                X_val = X_val_scaled
                
            if len(X_test) > 0:
                X_test_scaled = robust_scaler.transform(X_test)
                X_test = X_test_scaled
                
            X_train = X_train_scaled
            print(f"✅ Applied robust scaling: {robust_scaler.outlier_method} outlier clipping + {robust_scaler.scaler_method} scaling")
        else:
            print(f"⚠️  No training data to scale")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def run_progressive_validation(self):
        """Run complete progressive validation"""
        self.logger.log("=" * 70)
        self.logger.log("PROGRESSIVE MODEL VALIDATION STARTED")
        self.logger.log("=" * 70)
        
        # Load data per ticker (maintaining separation)
        data_loader = DataLoader(self.config)
        tickers = self.config.get("tickers")
        
        all_ticker_data = {}
        for ticker in tickers:
            try:
                ticker_data = data_loader.load_single_ticker_data(ticker, self.config_dict["years_of_data"])
                ticker_data['ticker'] = ticker
                all_ticker_data[ticker] = ticker_data
                self.logger.log(f"Loaded {len(ticker_data)} records for {ticker}")
            except Exception as e:
                self.logger.log(f"Error loading {ticker}: {e}")
        
        if not all_ticker_data:
            self.logger.log("❌ No data could be loaded")
            return None
        
        total_records = sum(len(data) for data in all_ticker_data.values())
        self.logger.log(f"Total dataset: {total_records} records across {len(all_ticker_data)} tickers")
        
        # Test different feature selection approaches
        feature_levels = [
            ("Comprehensive_Selected", self.create_comprehensive_selected_features),
            ("Baseline_Selected", self.create_baseline_selected_features),
            ("Technical_Selected", self.create_technical_selected_features),
            ("All_Features", self.create_all_features)
        ]
        
        validation_results = {}
        
        for level_name, feature_func in feature_levels:
            self.logger.log(f"\n--- TESTING {level_name.upper()} FEATURES ---")
            
            try:
                # Create features for each ticker separately 
                ticker_features_data = {}
                total_features_created = 0
                
                for ticker, ticker_data in all_ticker_data.items():
                    try:
                        self.logger.log(f"Creating {level_name} features for {ticker} (input: {len(ticker_data)} records)")
                        ticker_features = feature_func(ticker_data)
                        
                        # Enhanced validation of created features
                        if ticker_features is None or len(ticker_features) == 0:
                            self.logger.log(f"WARNING: {ticker} returned empty features")
                            continue
                        
                        # Check for required columns
                        if 'daily_return' not in ticker_features.columns:
                            self.logger.log(f"WARNING: {ticker} missing daily_return column")
                            continue
                        
                        # Check feature quality
                        feature_cols = [col for col in ticker_features.columns 
                                      if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'daily_return']]
                        nan_ratio = ticker_features[feature_cols].isna().sum().sum() / (len(ticker_features) * len(feature_cols))
                        
                        self.logger.log(f"{level_name} {ticker}: {len(ticker_features)} records, "
                                      f"{len(feature_cols)} features, {nan_ratio:.2%} NaN ratio")
                        
                        ticker_features_data[ticker] = ticker_features
                        total_features_created += len(ticker_features)
                        
                    except Exception as e:
                        self.logger.log(f"Feature creation FAILED for {ticker}: {e}")
                        import traceback
                        self.logger.log(f"Traceback: {traceback.format_exc()}")
                        continue
                
                if not ticker_features_data:
                    self.logger.log(f"{level_name} FAILED: No features could be created")
                    validation_results[level_name] = {"error": "Feature creation failed for all tickers"}
                    continue
                
                self.logger.log(f"{level_name} total features created: {total_features_created} records")
                
                # Create proper time series splits with error handling
                try:
                    (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.create_time_series_splits(ticker_features_data)
                    
                    if len(X_train) == 0:
                        self.logger.log(f"{level_name} FAILED: No training sequences created")
                        validation_results[level_name] = {"error": "No training sequences"}
                        continue
                        
                except Exception as e:
                    self.logger.log(f"{level_name} FAILED: {e}")
                    import traceback
                    self.logger.log(f"Traceback:\n{traceback.format_exc()}")
                    validation_results[level_name] = {"error": str(e)}
                    continue
                
                self.logger.log(f"{level_name} sequences: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
                
                # Test baseline models (use flattened data)
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                baseline_results = self.test_baseline_models_with_splits(X_train_flat, y_train, X_val.reshape(X_val.shape[0], -1), y_val, level_name)
                
                # Test LSTM model
                lstm_results = self.test_lstm_model_with_splits(X_train, y_train, X_val, y_val, X_test, y_test, level_name)
                
                validation_results[level_name] = {
                    "num_features": X_train.shape[2] if len(X_train.shape) > 2 else 0,
                    "num_sequences": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
                    "baseline_models": baseline_results,
                    "lstm_model": lstm_results
                }
                
            except Exception as e:
                self.logger.log(f"{level_name} FAILED: {e}")
                import traceback
                self.logger.log(traceback.format_exc())
                validation_results[level_name] = {"error": str(e)}
        
        # Generate summary report
        self.generate_summary_report(validation_results)
        
        return validation_results
    
    def generate_summary_report(self, results):
        """Generate comprehensive summary report"""
        self.logger.log("\n" + "=" * 70)
        self.logger.log("PROGRESSIVE VALIDATION SUMMARY REPORT")
        self.logger.log("=" * 70)
        
        print("\nPROGRESSIVE MODEL VALIDATION RESULTS")
        print("=" * 50)
        
        for level_name, level_results in results.items():
            if "error" in level_results:
                print(f"\n{level_name}: FAILED - {level_results['error']}")
                continue
            
            print(f"\n{level_name} Features ({level_results['num_features']} features, {level_results['num_sequences']} sequences):")
            
            # Baseline models
            if level_results['baseline_models']:
                print("  Baseline Models:")
                for model_name, metrics in level_results['baseline_models'].items():
                    print(f"    {model_name}: Dir={metrics['directional_accuracy']:.1f}%, MAE={metrics['mae']:.6f}, Corr={metrics['correlation']:.3f}")
            
            # LSTM model
            if level_results['lstm_model']:
                lstm = level_results['lstm_model']
                print(f"  LSTM Model:")
                print(f"    Validation: Dir={lstm['val_directional_accuracy']:.1f}%, MAE={lstm['val_mae']:.6f}, Corr={lstm['val_correlation']:.3f}")
                print(f"    Test: Dir={lstm['test_directional_accuracy']:.1f}%, MAE={lstm['test_mae']:.6f}, Corr={lstm['test_correlation']:.3f}")
                print(f"    Parameters: {lstm['parameters']:,}, Training: {lstm['training_time']:.1f}s")
        
        # Best performing models
        print(f"\n" + "=" * 50)
        print("BEST PERFORMING MODELS:")
        
        best_lstm_dir_acc = 0
        best_lstm_level = None
        
        for level_name, level_results in results.items():
            if "lstm_model" in level_results and level_results['lstm_model']:
                lstm = level_results['lstm_model']
                if lstm['test_directional_accuracy'] > best_lstm_dir_acc:
                    best_lstm_dir_acc = lstm['test_directional_accuracy']
                    best_lstm_level = level_name
        
        if best_lstm_level:
            print(f"Best LSTM: {best_lstm_level} Features ({best_lstm_dir_acc:.1f}% directional accuracy)")
        
        print("=" * 50)


def main():
    """Run progressive validation"""
    validator = ProgressiveModelValidator({
        "tickers": ["AAPL", "MSFT", "GOOG"],  # 3 tickers for comprehensive testing
        "years_of_data": 5,  # 5 years for robust testing
        "prediction_horizon": "daily",
        "lookback_window": 20,
        "database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db"
    })
    
    results = validator.run_progressive_validation()
    
    if results:
        print("\n✅ Progressive validation completed!")
        print("Check 'progressive_validation.log' for detailed results")
    else:
        print("\n❌ Progressive validation failed")


if __name__ == "__main__":
    main()