#!/usr/bin/env python3

"""
Volatility features processor including GARCH
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ..base import BaseFeatureProcessor, safe_divide


class VolatilityFeaturesProcessor(BaseFeatureProcessor):
    """
    Processor for volatility-based features including GARCH
    """
    
    def __init__(self):
        super().__init__("volatility_features")
        self.feature_names = [
            "volatility_5d", "volatility_20d", "volatility_60d",
            "volatility_ratio", "return_volatility_5d", 
            "garch_volatility", "volatility_clustering", "volatility_skew"
        ]
        self.dependencies = ["close", "daily_return"]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features"""
        self.validate_dependencies(data)
        result = data.copy()
        
        # Calculate log returns for volatility analysis
        returns = np.log(result["close"] / result["close"].shift(1)).dropna()
        
        # Basic volatility measures
        result = self._calculate_basic_volatility(result, returns)
        
        # GARCH volatility
        result["garch_volatility"] = self._calculate_garch_volatility(returns, result.index)
        
        # Advanced volatility features
        result = self._calculate_advanced_volatility(result, returns)
        
        return result
    
    def _calculate_basic_volatility(self, data: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Calculate basic volatility measures"""
        # Annualized volatility for different periods
        data["volatility_5d"] = (returns.rolling(5).std() * np.sqrt(252)).reindex(data.index)
        data["volatility_20d"] = (returns.rolling(20).std() * np.sqrt(252)).reindex(data.index)
        data["volatility_60d"] = (returns.rolling(60).std() * np.sqrt(252)).reindex(data.index)
        
        # Volatility ratio (short vs long term)
        data["volatility_ratio"] = safe_divide(data["volatility_5d"], data["volatility_20d"], 1.0)
        
        # Simple rolling volatility
        data["return_volatility_5d"] = (returns.rolling(5).std() * np.sqrt(252)).reindex(data.index)
        
        return data
    
    def _calculate_garch_volatility(self, returns: pd.Series, full_index: pd.Index, 
                                   p: int = 1, q: int = 1, window: int = 100, 
                                   min_periods: int = 30) -> pd.Series:
        """
        Calculate GARCH(p,q) conditional volatility
        
        GARCH(1,1) Model: σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
        """
        volatility = np.full(len(full_index), np.nan)
        returns_array = returns.values
        n_obs = len(returns_array)
        
        if n_obs < min_periods:
            return pd.Series(volatility, index=full_index)
        
        def estimate_garch_params(ret_window):
            """Estimate GARCH parameters using Maximum Likelihood Estimation"""
            def garch_likelihood(params):
                omega, alpha, beta = params
                
                # Parameter constraints for stationarity
                if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                    return 1e6
                    
                n = len(ret_window)
                sigma2 = np.full(n, np.var(ret_window))
                sigma2[0] = np.var(ret_window[:min(10, n)])
                
                # GARCH recursion
                for t in range(1, n):
                    sigma2[t] = omega + alpha * ret_window[t-1]**2 + beta * sigma2[t-1]
                    sigma2[t] = max(sigma2[t], 1e-8)
                
                # Negative log-likelihood
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + ret_window**2 / sigma2)
                return -log_likelihood
            
            initial_guess = [np.var(ret_window) * 0.1, 0.1, 0.8]
            bounds = [(1e-8, None), (0, 1), (0, 1)]
            
            try:
                result = minimize(garch_likelihood, initial_guess, bounds=bounds, method="L-BFGS-B")
                if result.success and result.x[1] + result.x[2] < 0.999:
                    return result.x
                else:
                    return [np.var(ret_window) * 0.05, 0.05, 0.90]
            except:
                return [np.var(ret_window) * 0.05, 0.05, 0.90]
        
        # Rolling window GARCH calculation
        start_idx = min_periods
        
        for i in range(start_idx, n_obs):
            window_start = max(0, i - window + 1)
            window_end = i + 1
            ret_window = returns_array[window_start:window_end]
            
            # Re-estimate parameters periodically
            if i == start_idx or i % max(1, window // 4) == 0:
                omega, alpha, beta = estimate_garch_params(ret_window)
            
            # GARCH volatility calculation
            window_length = len(ret_window)
            sigma2_window = np.zeros(window_length)
            sigma2_window[0] = np.var(ret_window[:min(10, window_length)])
            
            # Apply GARCH recursion
            for t in range(1, window_length):
                sigma2_window[t] = omega + alpha * ret_window[t-1]**2 + beta * sigma2_window[t-1]
                sigma2_window[t] = max(sigma2_window[t], 1e-8)
            
            current_vol = np.sqrt(sigma2_window[-1])
            
            # Map back to original index
            try:
                original_idx = returns.index[i]
                full_idx_position = full_index.get_loc(original_idx)
                volatility[full_idx_position] = current_vol
            except:
                continue  # Skip if index mapping fails
        
        return pd.Series(volatility, index=full_index)
    
    def _calculate_advanced_volatility(self, data: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Calculate advanced volatility features"""
        # Volatility clustering indicator
        squared_returns = returns**2
        clustering_measure = squared_returns.rolling(10).corr(squared_returns.shift(1))
        data["volatility_clustering"] = clustering_measure.reindex(data.index)
        
        # Volatility skewness (asymmetry in volatility distribution)
        vol_rolling = returns.rolling(20).std()
        vol_skew = vol_rolling.rolling(60).skew()
        data["volatility_skew"] = vol_skew.reindex(data.index)
        
        return data