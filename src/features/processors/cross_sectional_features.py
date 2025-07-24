#!/usr/bin/env python3

"""
Cross-sectional features processor for inter-stock relationships and sector analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from ..base import BaseFeatureProcessor, safe_divide


class CrossSectionalFeaturesProcessor(BaseFeatureProcessor):
    """
    Processor for cross-sectional features that capture relationships between stocks
    """
    
    def __init__(self, all_ticker_data: Dict[str, pd.DataFrame] = None):
        super().__init__("cross_sectional_features")
        self.all_ticker_data = all_ticker_data or {}
        
        # Define sector mappings for the 15 tickers
        self.sector_mapping = {
            # Technology
            "AAPL": "tech", "MSFT": "tech", "GOOG": "tech", "NVDA": "tech", 
            "TSLA": "tech", "AMZN": "tech", "META": "tech", "NFLX": "tech", 
            "AMD": "tech", "INTC": "tech",
            # Finance
            "JPM": "finance", "BAC": "finance", "GS": "finance",
            # Healthcare  
            "JNJ": "healthcare", "PFE": "healthcare"
        }
        
        self.feature_names = [
            # Inter-stock correlations
            "market_correlation_5d", "market_correlation_20d", "sector_correlation_20d",
            "peer_correlation_mean", "peer_correlation_max", "peer_correlation_min",
            
            # Relative performance
            "relative_strength_market", "relative_strength_sector", 
            "outperformance_ratio", "beta_to_market", "beta_to_sector",
            
            # Cross-sectional momentum
            "sector_momentum_5d", "sector_momentum_20d", "market_momentum_20d",
            "relative_volume_sector", "relative_volatility_sector",
            
            # Leadership indicators
            "correlation_stability", "beta_stability", "sector_leadership",
            "market_leadership", "flow_divergence", "sentiment_divergence"
        ]
        
    def calculate(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """Main calculate method (required by base class)"""
        return self.calculate_features(data, symbol)
    
    def calculate_features(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """Calculate cross-sectional features for a given stock"""
        if not self.all_ticker_data or symbol not in self.all_ticker_data:
            # Return original data with zero cross-sectional features
            result = data.copy()
            neutral_features = self._create_neutral_features(len(data))
            for col in neutral_features.columns:
                result[col] = neutral_features[col]
            return result
        
        print(f"  Calculating cross-sectional features for {symbol}...")
        
        # Start with copy of original data to preserve existing columns
        result = data.copy()
        features = pd.DataFrame(index=data.index)
        
        # Get current stock data
        current_data = data.copy()
        current_sector = self.sector_mapping.get(symbol, "unknown")
        
        # Calculate market-wide statistics
        market_data = self._calculate_market_aggregates()
        sector_data = self._calculate_sector_aggregates(current_sector, exclude_symbol=symbol)
        
        # 1. INTER-STOCK CORRELATIONS
        features = self._add_correlation_features(features, current_data, market_data, sector_data, symbol)
        
        # 2. RELATIVE PERFORMANCE FEATURES
        features = self._add_relative_performance_features(features, current_data, market_data, sector_data)
        
        # 3. CROSS-SECTIONAL MOMENTUM
        features = self._add_momentum_features(features, current_data, market_data, sector_data)
        
        # 4. LEADERSHIP INDICATORS
        features = self._add_leadership_features(features, current_data, symbol)
        
        # Merge new features with original data
        for col in features.columns:
            result[col] = features[col]
        
        return result
    
    def _calculate_market_aggregates(self) -> pd.DataFrame:
        """Calculate market-wide aggregated data"""
        all_returns = []
        all_volumes = []
        common_dates = None
        
        for symbol, ticker_data in self.all_ticker_data.items():
            if 'daily_return' in ticker_data.columns:
                returns = ticker_data[['daily_return']].rename(columns={'daily_return': symbol})
                volumes = ticker_data[['volume']].rename(columns={'volume': f"{symbol}_vol"})
                
                if common_dates is None:
                    common_dates = ticker_data.index
                else:
                    common_dates = common_dates.intersection(ticker_data.index)
                
                all_returns.append(returns)
                all_volumes.append(volumes)
        
        if not all_returns:
            return pd.DataFrame()
        
        # Combine all returns and calculate market aggregates
        market_returns = pd.concat(all_returns, axis=1).loc[common_dates]
        market_volumes = pd.concat(all_volumes, axis=1).loc[common_dates]
        
        # Calculate market-wide metrics
        market_data = pd.DataFrame(index=common_dates)
        market_data['market_return'] = market_returns.mean(axis=1)
        market_data['market_volume'] = market_volumes.sum(axis=1)
        market_data['market_volatility'] = market_returns.std(axis=1)
        
        # Rolling market statistics
        market_data['market_momentum_5d'] = market_data['market_return'].rolling(5).mean()
        market_data['market_momentum_20d'] = market_data['market_return'].rolling(20).mean()
        
        return market_data
    
    def _calculate_sector_aggregates(self, sector: str, exclude_symbol: str = None) -> pd.DataFrame:
        """Calculate sector-specific aggregated data"""
        sector_returns = []
        sector_volumes = []
        common_dates = None
        
        for symbol, ticker_data in self.all_ticker_data.items():
            if (self.sector_mapping.get(symbol) == sector and 
                symbol != exclude_symbol and
                'daily_return' in ticker_data.columns):
                
                returns = ticker_data[['daily_return']].rename(columns={'daily_return': symbol})
                volumes = ticker_data[['volume']].rename(columns={'volume': f"{symbol}_vol"})
                
                if common_dates is None:
                    common_dates = ticker_data.index
                else:
                    common_dates = common_dates.intersection(ticker_data.index)
                
                sector_returns.append(returns)
                sector_volumes.append(volumes)
        
        if not sector_returns:
            return pd.DataFrame()
        
        # Combine sector data
        sector_returns_df = pd.concat(sector_returns, axis=1).loc[common_dates]
        sector_volumes_df = pd.concat(sector_volumes, axis=1).loc[common_dates]
        
        # Calculate sector metrics
        sector_data = pd.DataFrame(index=common_dates)
        sector_data['sector_return'] = sector_returns_df.mean(axis=1)
        sector_data['sector_volume'] = sector_volumes_df.sum(axis=1)
        sector_data['sector_volatility'] = sector_returns_df.std(axis=1)
        
        # Rolling sector statistics
        sector_data['sector_momentum_5d'] = sector_data['sector_return'].rolling(5).mean()
        sector_data['sector_momentum_20d'] = sector_data['sector_return'].rolling(20).mean()
        
        return sector_data
    
    def _add_correlation_features(self, features: pd.DataFrame, current_data: pd.DataFrame, 
                                market_data: pd.DataFrame, sector_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add correlation-based features"""
        
        # Align indices
        common_idx = features.index.intersection(market_data.index).intersection(current_data.index)
        if len(common_idx) == 0:
            return self._add_zero_columns(features, [
                'market_correlation_5d', 'market_correlation_20d', 'sector_correlation_20d',
                'peer_correlation_mean', 'peer_correlation_max', 'peer_correlation_min'
            ])
        
        # Market correlations
        if 'daily_return' in current_data.columns:
            current_returns = current_data.loc[common_idx, 'daily_return']
            market_returns = market_data.loc[common_idx, 'market_return'] if 'market_return' in market_data.columns else pd.Series(0, index=common_idx)
            
            # Rolling correlations with market
            features.loc[common_idx, 'market_correlation_5d'] = current_returns.rolling(5).corr(market_returns.rolling(5)).fillna(0)
            features.loc[common_idx, 'market_correlation_20d'] = current_returns.rolling(20).corr(market_returns.rolling(20)).fillna(0)
            
            # Sector correlation
            if not sector_data.empty and 'sector_return' in sector_data.columns:
                sector_returns = sector_data.loc[common_idx, 'sector_return']
                features.loc[common_idx, 'sector_correlation_20d'] = current_returns.rolling(20).corr(sector_returns.rolling(20)).fillna(0)
            else:
                features.loc[common_idx, 'sector_correlation_20d'] = 0
            
            # Peer correlations (with other stocks in same sector)
            peer_correlations = []
            current_sector = self.sector_mapping.get(symbol, "unknown")
            
            for peer_symbol, peer_data in self.all_ticker_data.items():
                if (peer_symbol != symbol and 
                    self.sector_mapping.get(peer_symbol) == current_sector and
                    'daily_return' in peer_data.columns):
                    
                    peer_idx = common_idx.intersection(peer_data.index)
                    if len(peer_idx) > 20:  # Need sufficient data for correlation
                        peer_returns = peer_data.loc[peer_idx, 'daily_return']
                        peer_corr = current_returns.loc[peer_idx].rolling(20).corr(peer_returns.rolling(20))
                        peer_correlations.append(peer_corr.fillna(0))
            
            if peer_correlations:
                peer_corr_df = pd.concat(peer_correlations, axis=1)
                features.loc[common_idx, 'peer_correlation_mean'] = peer_corr_df.mean(axis=1).fillna(0)
                features.loc[common_idx, 'peer_correlation_max'] = peer_corr_df.max(axis=1).fillna(0)
                features.loc[common_idx, 'peer_correlation_min'] = peer_corr_df.min(axis=1).fillna(0)
            else:
                features.loc[common_idx, 'peer_correlation_mean'] = 0
                features.loc[common_idx, 'peer_correlation_max'] = 0
                features.loc[common_idx, 'peer_correlation_min'] = 0
        
        return features
    
    def _add_relative_performance_features(self, features: pd.DataFrame, current_data: pd.DataFrame,
                                         market_data: pd.DataFrame, sector_data: pd.DataFrame) -> pd.DataFrame:
        """Add relative performance features"""
        
        common_idx = features.index.intersection(market_data.index).intersection(current_data.index)
        if len(common_idx) == 0:
            return self._add_zero_columns(features, [
                'relative_strength_market', 'relative_strength_sector', 
                'outperformance_ratio', 'beta_to_market', 'beta_to_sector'
            ])
        
        if 'daily_return' in current_data.columns:
            current_returns = current_data.loc[common_idx, 'daily_return']
            
            # Relative strength vs market
            if 'market_return' in market_data.columns:
                market_returns = market_data.loc[common_idx, 'market_return']
                features.loc[common_idx, 'relative_strength_market'] = (
                    current_returns.rolling(20).sum() - market_returns.rolling(20).sum()
                ).fillna(0)
                
                # Beta to market
                covariance = current_returns.rolling(60).cov(market_returns).fillna(0)
                market_variance = market_returns.rolling(60).var().fillna(1e-8)
                features.loc[common_idx, 'beta_to_market'] = safe_divide(covariance, market_variance).fillna(1)
                
                # Outperformance ratio  
                outperf = safe_divide(current_returns.rolling(20).sum(), market_returns.rolling(20).sum().abs() + 1e-8)
                features.loc[common_idx, 'outperformance_ratio'] = outperf.fillna(1)
            else:
                features.loc[common_idx, 'relative_strength_market'] = 0
                features.loc[common_idx, 'beta_to_market'] = 1
                features.loc[common_idx, 'outperformance_ratio'] = 1
            
            # Relative strength vs sector
            if not sector_data.empty and 'sector_return' in sector_data.columns:
                sector_returns = sector_data.loc[common_idx, 'sector_return']
                features.loc[common_idx, 'relative_strength_sector'] = (
                    current_returns.rolling(20).sum() - sector_returns.rolling(20).sum()
                ).fillna(0)
                
                # Beta to sector
                covariance = current_returns.rolling(60).cov(sector_returns).fillna(0)
                sector_variance = sector_returns.rolling(60).var().fillna(1e-8)
                features.loc[common_idx, 'beta_to_sector'] = safe_divide(covariance, sector_variance).fillna(1)
            else:
                features.loc[common_idx, 'relative_strength_sector'] = 0
                features.loc[common_idx, 'beta_to_sector'] = 1
        
        return features
    
    def _add_momentum_features(self, features: pd.DataFrame, current_data: pd.DataFrame,
                             market_data: pd.DataFrame, sector_data: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional momentum features"""
        
        common_idx = features.index.intersection(market_data.index).intersection(current_data.index)
        if len(common_idx) == 0:
            return self._add_zero_columns(features, [
                'sector_momentum_5d', 'sector_momentum_20d', 'market_momentum_20d',
                'relative_volume_sector', 'relative_volatility_sector'
            ])
        
        # Market momentum
        if 'market_momentum_20d' in market_data.columns:
            features.loc[common_idx, 'market_momentum_20d'] = market_data.loc[common_idx, 'market_momentum_20d'].fillna(0)
        else:
            features.loc[common_idx, 'market_momentum_20d'] = 0
        
        # Sector momentum
        if not sector_data.empty:
            if 'sector_momentum_5d' in sector_data.columns:
                features.loc[common_idx, 'sector_momentum_5d'] = sector_data.loc[common_idx, 'sector_momentum_5d'].fillna(0)
            else:
                features.loc[common_idx, 'sector_momentum_5d'] = 0
                
            if 'sector_momentum_20d' in sector_data.columns:
                features.loc[common_idx, 'sector_momentum_20d'] = sector_data.loc[common_idx, 'sector_momentum_20d'].fillna(0)
            else:
                features.loc[common_idx, 'sector_momentum_20d'] = 0
            
            # Relative volume and volatility
            if ('volume' in current_data.columns and 'sector_volume' in sector_data.columns):
                current_vol = current_data.loc[common_idx, 'volume']
                sector_vol = sector_data.loc[common_idx, 'sector_volume']
                features.loc[common_idx, 'relative_volume_sector'] = safe_divide(current_vol, sector_vol).fillna(1)
            else:
                features.loc[common_idx, 'relative_volume_sector'] = 1
                
            if 'sector_volatility' in sector_data.columns:
                current_vol_20d = current_data.loc[common_idx, 'daily_return'].rolling(20).std() if 'daily_return' in current_data.columns else pd.Series(0, index=common_idx)
                sector_vol_20d = sector_data.loc[common_idx, 'sector_volatility']
                features.loc[common_idx, 'relative_volatility_sector'] = safe_divide(current_vol_20d, sector_vol_20d).fillna(1)
            else:
                features.loc[common_idx, 'relative_volatility_sector'] = 1
        else:
            # No sector data available
            for col in ['sector_momentum_5d', 'sector_momentum_20d', 'relative_volume_sector', 'relative_volatility_sector']:
                features.loc[common_idx, col] = 0
        
        return features
    
    def _add_leadership_features(self, features: pd.DataFrame, current_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add leadership and stability indicators"""
        
        # Initialize leadership features with zeros
        leadership_cols = ['correlation_stability', 'beta_stability', 'sector_leadership', 
                          'market_leadership', 'flow_divergence', 'sentiment_divergence']
        
        for col in leadership_cols:
            features[col] = 0
        
        # Calculate stability metrics if we have market correlation data
        if 'market_correlation_20d' in features.columns:
            corr_20d = features['market_correlation_20d']
            # Stability = inverse of rolling standard deviation
            features['correlation_stability'] = 1 / (corr_20d.rolling(20).std().fillna(1) + 0.1)
        
        if 'beta_to_market' in features.columns:
            beta = features['beta_to_market'] 
            features['beta_stability'] = 1 / (beta.rolling(20).std().fillna(1) + 0.1)
        
        # Simple leadership indicators based on relative performance
        if 'relative_strength_sector' in features.columns:
            rel_strength = features['relative_strength_sector']
            features['sector_leadership'] = (rel_strength > rel_strength.rolling(60).quantile(0.7)).astype(int)
        
        if 'relative_strength_market' in features.columns:
            rel_strength_mkt = features['relative_strength_market']
            features['market_leadership'] = (rel_strength_mkt > rel_strength_mkt.rolling(60).quantile(0.7)).astype(int)
        
        # Flow and sentiment divergence (placeholder - would need additional data sources)
        features['flow_divergence'] = np.random.normal(0, 0.1, len(features)) * 0  # Disabled for now
        features['sentiment_divergence'] = np.random.normal(0, 0.1, len(features)) * 0  # Disabled for now
        
        return features
    
    def _create_neutral_features(self, length: int) -> pd.DataFrame:
        """Create dataframe with neutral values for all cross-sectional features"""
        features = pd.DataFrame(index=range(length))
        
        # Use neutral values instead of zeros for better model training
        neutral_values = {
            # Correlations: neutral correlation
            "market_correlation_5d": 0.5, "market_correlation_20d": 0.5, "sector_correlation_20d": 0.5,
            "peer_correlation_mean": 0.5, "peer_correlation_max": 0.5, "peer_correlation_min": 0.5,
            
            # Relative performance: neutral relative strength
            "relative_strength_market": 1.0, "relative_strength_sector": 1.0, 
            "outperformance_ratio": 1.0, "beta_to_market": 1.0, "beta_to_sector": 1.0,
            
            # Momentum: neutral momentum
            "sector_momentum_5d": 0.0, "sector_momentum_20d": 0.0, "market_momentum_20d": 0.0,
            "relative_volume_sector": 1.0, "relative_volatility_sector": 1.0,
            
            # Leadership: neutral leadership
            "correlation_stability": 0.5, "beta_stability": 0.5, "sector_leadership": 0.5,
            "market_leadership": 0.5, "flow_divergence": 0.0, "sentiment_divergence": 0.0
        }
        
        for feature_name in self.feature_names:
            features[feature_name] = neutral_values.get(feature_name, 0.0)
            
        return features
    
    def _add_zero_columns(self, features: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Add zero columns to features dataframe"""
        for col in columns:
            if col not in features.columns:
                features[col] = 0
        return features

    def get_feature_names(self) -> List[str]:
        """Return list of feature names this processor generates"""
        return self.feature_names