#!/usr/bin/env python3

"""
Macro-economic features processor for broader market and economic indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from ..base import BaseFeatureProcessor, safe_divide


class MacroFeaturesProcessor(BaseFeatureProcessor):
    """
    Processor for macro-economic features including VIX, yields, and market regime indicators
    """
    
    def __init__(self, vix_data: Optional[pd.DataFrame] = None):
        super().__init__("macro_features")
        self.vix_data = vix_data
        
        self.feature_names = [
            # VIX and volatility regime
            "vix_level", "vix_change", "vix_percentile", "vix_term_structure",
            "volatility_regime", "vix_mean_reversion", "fear_greed_index",
            
            # Market regime indicators
            "market_regime_bull", "market_regime_bear", "market_regime_sideways",
            "trend_strength", "market_stress_index", "liquidity_conditions",
            
            # Economic cycle indicators
            "economic_cycle", "recession_probability", "growth_momentum",
            "inflation_expectations", "credit_conditions", "currency_strength",
            
            # Cross-asset signals
            "equity_bond_ratio", "risk_on_risk_off", "commodity_momentum",
            "dollar_strength", "yield_curve_slope", "credit_spreads"
        ]
        
    def calculate(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """Main calculate method (required by base class)"""
        return self.calculate_features(data, symbol)
    
    def calculate_features(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """Calculate macro-economic features"""
        
        print(f"  Calculating macro-economic features...")
        
        # Start with copy of original data to preserve existing columns
        result = data.copy()
        features = pd.DataFrame(index=data.index)
        
        # 1. VIX AND VOLATILITY FEATURES
        features = self._add_vix_features(features, data)
        
        # 2. MARKET REGIME FEATURES
        features = self._add_market_regime_features(features, data)
        
        # 3. ECONOMIC CYCLE FEATURES
        features = self._add_economic_cycle_features(features, data)
        
        # 4. CROSS-ASSET FEATURES
        features = self._add_cross_asset_features(features, data)
        
        # Merge new features with original data
        for col in features.columns:
            result[col] = features[col]
        
        return result
    
    def _add_vix_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add VIX and volatility-related features"""
        
        # For now, simulate VIX data based on market volatility patterns
        # In production, this would connect to real VIX data
        
        if 'daily_return' in data.columns:
            returns = data['daily_return']
            
            # Simulate VIX level based on realized volatility
            realized_vol = returns.rolling(20).std() * np.sqrt(252) * 100  # Annualized vol
            simulated_vix = realized_vol * np.random.normal(1.5, 0.2, len(realized_vol))  # VIX typically > realized vol
            simulated_vix = np.clip(simulated_vix, 10, 80)  # Reasonable VIX range
            
            features['vix_level'] = simulated_vix.fillna(20)
            
            # VIX change and dynamics
            features['vix_change'] = features['vix_level'].pct_change().fillna(0)
            
            # VIX percentile (relative to recent history)
            features['vix_percentile'] = (
                features['vix_level'].rolling(252).rank(pct=True).fillna(0.5)
            )
            
            # VIX term structure (contango/backwardation indicator)
            # Simulated: positive when VIX is below long-term average (contango)
            vix_ma = features['vix_level'].rolling(60).mean()
            features['vix_term_structure'] = safe_divide(
                vix_ma - features['vix_level'], features['vix_level']
            ).fillna(0)
            
            # Volatility regime (low/medium/high)
            vix_20_day = features['vix_level'].rolling(20).mean()
            features['volatility_regime'] = np.where(
                vix_20_day < 15, 0,  # Low vol regime
                np.where(vix_20_day < 25, 1, 2)  # Medium/High vol regime
            )
            
            # VIX mean reversion indicator
            vix_z_score = (features['vix_level'] - features['vix_level'].rolling(60).mean()) / (
                features['vix_level'].rolling(60).std() + 1e-8
            )
            features['vix_mean_reversion'] = vix_z_score.fillna(0)
            
            # Fear-Greed Index (0-100, derived from multiple factors)
            vol_component = 100 - np.clip(features['vix_level'], 10, 50)  # Lower VIX = higher greed
            momentum_component = np.clip(returns.rolling(20).mean() * 1000 + 50, 0, 100)
            features['fear_greed_index'] = (vol_component * 0.6 + momentum_component * 0.4).fillna(50)
            
        else:
            # Default values if no return data
            for col in ['vix_level', 'vix_change', 'vix_percentile', 'vix_term_structure',
                       'volatility_regime', 'vix_mean_reversion', 'fear_greed_index']:
                features[col] = 0 if col != 'vix_level' else 20
        
        return features
    
    def _add_market_regime_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification features"""
        
        if 'daily_return' in data.columns and 'close' in data.columns:
            returns = data['daily_return']
            prices = data['close']
            
            # Trend-based regime detection
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean()
            sma_200 = prices.rolling(200).mean()
            
            # Bull market: price > SMA20 > SMA50 > SMA200, positive momentum
            bull_condition = (
                (prices > sma_20) & 
                (sma_20 > sma_50) & 
                (sma_50 > sma_200) & 
                (returns.rolling(20).mean() > 0)
            ).astype(int)
            
            # Bear market: price < SMA20 < SMA50 < SMA200, negative momentum  
            bear_condition = (
                (prices < sma_20) & 
                (sma_20 < sma_50) & 
                (sma_50 < sma_200) & 
                (returns.rolling(20).mean() < 0)
            ).astype(int)
            
            # Sideways: everything else
            sideways_condition = 1 - bull_condition - bear_condition
            sideways_condition = np.clip(sideways_condition, 0, 1)
            
            features['market_regime_bull'] = bull_condition.fillna(0)
            features['market_regime_bear'] = bear_condition.fillna(0) 
            features['market_regime_sideways'] = sideways_condition.fillna(1)
            
            # Trend strength (how strong is the current trend)
            price_vs_sma200 = safe_divide(prices - sma_200, sma_200).fillna(0)
            features['trend_strength'] = np.abs(price_vs_sma200)
            
            # Market stress index (combination of volatility and negative returns)
            vol_stress = np.clip(returns.rolling(20).std() * 100, 0, 10)
            return_stress = np.clip(-returns.rolling(5).mean() * 100, 0, 5)
            features['market_stress_index'] = (vol_stress + return_stress).fillna(0)
            
            # Liquidity conditions (based on volume patterns)
            if 'volume' in data.columns:
                vol_ma = data['volume'].rolling(20).mean()
                vol_ratio = safe_divide(data['volume'], vol_ma).fillna(1)
                # High liquidity when volume is consistently above average
                features['liquidity_conditions'] = vol_ratio.rolling(5).mean().fillna(1)
            else:
                features['liquidity_conditions'] = 1
                
        else:
            # Default values
            for col in ['market_regime_bull', 'market_regime_bear', 'market_regime_sideways',
                       'trend_strength', 'market_stress_index', 'liquidity_conditions']:
                default_val = 0 if col != 'liquidity_conditions' else 1
                features[col] = default_val
            
            # Default sideways market
            features['market_regime_sideways'] = 1
        
        return features
    
    def _add_economic_cycle_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add economic cycle and fundamental indicators"""
        
        # These would typically come from economic data APIs (FRED, Bloomberg, etc.)
        # For now, we'll create proxy indicators based on market behavior
        
        if 'daily_return' in data.columns:
            returns = data['daily_return']
            
            # Economic cycle proxy (based on long-term market trends)
            long_term_momentum = returns.rolling(252).mean()  # 1-year momentum
            features['economic_cycle'] = np.where(
                long_term_momentum > 0.01, 1,  # Expansion
                np.where(long_term_momentum < -0.01, -1, 0)  # Contraction vs Stable
            )
            
            # Recession probability (based on volatility and negative performance)
            vol_20d = returns.rolling(20).std()
            returns_60d = returns.rolling(60).mean()
            recession_signal = (vol_20d > vol_20d.rolling(252).quantile(0.8)) & (returns_60d < -0.005)
            features['recession_probability'] = recession_signal.rolling(20).mean().fillna(0)
            
            # Growth momentum (accelerating vs decelerating growth)
            momentum_1m = returns.rolling(20).mean()
            momentum_3m = returns.rolling(60).mean()
            features['growth_momentum'] = (momentum_1m - momentum_3m).fillna(0)
            
            # Inflation expectations proxy (based on volatility regime)
            vol_regime = returns.rolling(60).std()
            features['inflation_expectations'] = np.clip(vol_regime * 50, 0, 5).fillna(2)
            
        else:
            # Default values
            for col in ['economic_cycle', 'recession_probability', 'growth_momentum', 'inflation_expectations']:
                features[col] = 0
        
        # Credit conditions and currency strength (simulated)
        features['credit_conditions'] = np.random.normal(0, 0.5, len(features)) * 0  # Neutral for now
        features['currency_strength'] = np.random.normal(0, 0.3, len(features)) * 0  # Neutral for now
        
        return features
    
    def _add_cross_asset_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset and multi-market indicators"""
        
        if 'daily_return' in data.columns:
            returns = data['daily_return']
            
            # Equity-bond ratio (proxy based on volatility)
            # Lower volatility might indicate bond outperformance
            vol_20d = returns.rolling(20).std()
            features['equity_bond_ratio'] = (1 / (vol_20d + 0.01)).fillna(1)
            
            # Risk-on/Risk-off indicator
            # Risk-on: positive returns + low volatility
            risk_on_score = returns.rolling(5).mean() - vol_20d
            features['risk_on_risk_off'] = risk_on_score.fillna(0)
            
            # Commodity momentum (simulated based on inflation expectations)
            inflation_proxy = vol_20d.rolling(60).mean()
            features['commodity_momentum'] = (inflation_proxy - inflation_proxy.rolling(120).mean()).fillna(0)
            
        else:
            # Default values
            for col in ['equity_bond_ratio', 'risk_on_risk_off', 'commodity_momentum']:
                features[col] = 0 if col != 'equity_bond_ratio' else 1
        
        # Dollar strength, yield curve, credit spreads (simulated - would need real data)
        features['dollar_strength'] = np.random.normal(0, 0.5, len(features)) * 0  # Neutral
        features['yield_curve_slope'] = np.random.normal(1, 0.3, len(features)) * 0  # Normal slope  
        features['credit_spreads'] = np.random.normal(2, 0.5, len(features)) * 0  # Normal spreads
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this processor generates"""
        return self.feature_names