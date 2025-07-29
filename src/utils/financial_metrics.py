#!/usr/bin/env python3

"""
Financial-specific evaluation metrics for trading model validation
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class FinancialMetrics:
    """Calculate financial performance metrics for trading models"""

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0:
            return 0.0

        # Annualize returns (assume daily data)
        annual_returns = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)

        if annual_volatility == 0:
            return 0.0

        return (annual_returns - risk_free_rate) / annual_volatility

    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> Tuple[float, int]:
        """Calculate maximum drawdown and its duration"""
        if len(returns) == 0:
            return 0.0, 0

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        max_drawdown = np.min(drawdown)

        # Calculate drawdown duration
        drawdown_periods = np.where(drawdown < 0)[0]
        if len(drawdown_periods) == 0:
            max_duration = 0
        else:
            # Find longest continuous drawdown period
            durations = []
            current_duration = 1
            for i in range(1, len(drawdown_periods)):
                if drawdown_periods[i] == drawdown_periods[i - 1] + 1:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_duration = 1
            durations.append(current_duration)
            max_duration = max(durations) if durations else 0

        return abs(max_drawdown), max_duration

    @staticmethod
    def calculate_information_ratio(
        predicted_returns: np.ndarray, actual_returns: np.ndarray, benchmark_returns: np.ndarray = None
    ) -> float:
        """Calculate information ratio vs benchmark (or zero if no benchmark)"""
        if benchmark_returns is None:
            benchmark_returns = np.zeros_like(actual_returns)

        excess_returns = actual_returns - benchmark_returns
        tracking_error = np.std(excess_returns)

        if tracking_error == 0:
            return 0.0

        return np.mean(excess_returns) / tracking_error

    @staticmethod
    def calculate_hit_rate(predicted_returns: np.ndarray, actual_returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate hit rate (% of correct directional predictions above threshold)"""
        if len(predicted_returns) == 0:
            return 0.0

        # Only consider predictions above threshold magnitude
        significant_mask = np.abs(actual_returns) > threshold
        if np.sum(significant_mask) == 0:
            return 0.0

        predicted_directions = np.sign(predicted_returns[significant_mask])
        actual_directions = np.sign(actual_returns[significant_mask])

        return np.mean(predicted_directions == actual_directions) * 100

    @staticmethod
    def calculate_profit_factor(predicted_returns: np.ndarray, actual_returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        # Simulate trading based on predictions
        trades = np.where(predicted_returns > 0, actual_returns, np.where(predicted_returns < 0, -actual_returns, 0))

        gross_profit = np.sum(trades[trades > 0])
        gross_loss = abs(np.sum(trades[trades < 0]))

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 1.0

        return gross_profit / gross_loss

    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(returns) == 0:
            return 0.0

        annual_return = np.mean(returns) * 252
        max_drawdown, _ = FinancialMetrics.calculate_max_drawdown(returns)

        if max_drawdown == 0:
            return np.inf if annual_return > 0 else 0.0

        return annual_return / max_drawdown

    @classmethod
    def evaluate_trading_performance(
        cls, predicted_returns: np.ndarray, actual_returns: np.ndarray
    ) -> Dict[str, float]:
        """Comprehensive trading performance evaluation"""
        # Simulate trading returns based on predictions
        trading_returns = np.where(
            predicted_returns > 0, actual_returns, np.where(predicted_returns < 0, -actual_returns, 0)
        )

        metrics = {
            "total_return": np.sum(trading_returns) * 100,
            "annual_return": np.mean(trading_returns) * 252 * 100,
            "sharpe_ratio": cls.calculate_sharpe_ratio(trading_returns),
            "max_drawdown": cls.calculate_max_drawdown(trading_returns)[0] * 100,
            "drawdown_duration": cls.calculate_max_drawdown(trading_returns)[1],
            "hit_rate": cls.calculate_hit_rate(predicted_returns, actual_returns),
            "hit_rate_significant": cls.calculate_hit_rate(predicted_returns, actual_returns, 0.01),
            "profit_factor": cls.calculate_profit_factor(predicted_returns, actual_returns),
            "calmar_ratio": cls.calculate_calmar_ratio(trading_returns),
            "information_ratio": cls.calculate_information_ratio(predicted_returns, actual_returns),
            "volatility": np.std(trading_returns) * np.sqrt(252) * 100,
            "num_trades": np.sum(np.abs(predicted_returns) > 0.001),  # Significant predictions
        }

        return metrics

    @staticmethod
    def print_performance_report(metrics: Dict[str, float], phase: str = ""):
        """Print formatted performance report"""
        print(f"\n{phase} FINANCIAL PERFORMANCE METRICS")
        print("=" * 50)

        print(f"Return Metrics:")
        print(f"  Total Return: {metrics['total_return']:.2f}%")
        print(f"  Annualized Return: {metrics['annual_return']:.2f}%")
        print(f"  Annualized Volatility: {metrics['volatility']:.2f}%")

        print(f"\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        print(f"  Information Ratio: {metrics['information_ratio']:.3f}")

        print(f"\nRisk Metrics:")
        print(f"  Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"  Drawdown Duration: {metrics['drawdown_duration']} periods")

        print(f"\nAccuracy Metrics:")
        print(f"  Hit Rate (All): {metrics['hit_rate']:.1f}%")
        print(f"  Hit Rate (>1%): {metrics['hit_rate_significant']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

        print(f"\nTrading Activity:")
        print(f"  Number of Trades: {int(metrics['num_trades'])}")

        print("=" * 50)

