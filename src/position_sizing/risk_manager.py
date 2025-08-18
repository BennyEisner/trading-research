#!/usr/bin/env python3

"""
Portfolio Risk Manager
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .sigma_unit_calculator import PositionSizeResult, SigmaUnitCalculator


class RiskConstraints(BaseModel):
    """Portfolio-level risk constraints"""

    portfolio_vol_target: float = Field(default=0.12, gt=0.0, description="Annual portfolio volatility target")
    vol_tolerance: float = Field(default=0.02, gt=0.0, description="Volatility tolerance band")

    max_single_position: float = Field(default=0.15, gt=0.0, le=1.0, description="Maximum single asset position")
    max_total_exposure: float = Field(default=1.0, gt=0.0, le=2.0, description="Maximum total portfolio exposure")
    max_net_exposure: float = Field(default=0.8, gt=0.0, le=1.0, description="Maximum net portfolio exposure")

    max_sector_exposure: float = Field(default=0.4, gt=0.0, le=1.0, description="Maximum sector concentration")
    max_correlation_exposure: float = Field(
        default=0.5, gt=0.0, le=1.0, description="Maximum correlated position exposure"
    )

    # Risk monitoring
    max_drawdown_limit: float = Field(default=0.15, gt=0.0, le=1.0, description="Maximum drawdown threshold")
    var_confidence: float = Field(default=0.95, gt=0.0, lt=1.0, description="VaR confidence level")


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics and constraints status"""

    estimated_portfolio_vol: float
    vol_target_ratio: float  # Actual / Target
    vol_constraint_satisfied: bool

    # Exposure metrics
    total_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float

    # Position metrics
    max_position_size: float
    position_count: int
    active_tickers: List[str]

    constraints_satisfied: Dict[str, bool]
    risk_budget_used: float  # Fraction of risk budget utilized


class PortfolioRiskManager:
    """
    Portfolio-level risk management for σ-unit position sizing
    """

    def __init__(self, constraints: RiskConstraints):
        """
        Initialize portfolio risk manager

        Args:
            constraints: Portfolio risk constraints
        """
        self.constraints = constraints
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.asset_volatilities: Dict[str, float] = {}

        print(f"Portfolio Risk Manager initialized:")
        print(f"- Portfolio vol target: {constraints.portfolio_vol_target:.1%}")
        print(f"- Max single position: {constraints.max_single_position:.1%}")
        print(f"- Max total exposure: {constraints.max_total_exposure:.1%}")

    def update_correlation_matrix(self, correlation_data: pd.DataFrame):
        """
        Update asset correlation matrix for portfolio vol calculation

        Args:
            correlation_data: Correlation matrix with tickers as index/columns
        """
        self.correlation_matrix = correlation_data.copy()
        print(f"Updated correlation matrix: {correlation_data.shape[0]} assets")

    def update_asset_volatilities(self, volatilities: Dict[str, float]):
        """
        Update asset volatility estimates

        Args:
            volatilities: Dictionary of {ticker: daily_volatility}
        """
        self.asset_volatilities.update(volatilities)
        print(f"Updated volatilities for {len(volatilities)} assets")

    def calculate_portfolio_volatility(self, positions: Dict[str, PositionSizeResult]) -> float:
        """
        Calculate portfolio volatility using correlation matrix

        Args:
            positions: Dictionary of position results

        Returns:
            Estimated portfolio volatility (daily)
        """
        if not positions or self.correlation_matrix is None:
            return 0.0

        # Extract position weights and tickers
        tickers = list(positions.keys())
        weights = np.array([positions[ticker].f for ticker in tickers])

        # Get volatilities for these tickers
        vols = np.array([self.asset_volatilities.get(ticker, 0.02) for ticker in tickers])

        # Get correlation submatrix
        common_tickers = [t for t in tickers if t in self.correlation_matrix.index]
        if len(common_tickers) < 2:
            # Fallback: weighted average of individual volatilities
            return np.sum(np.abs(weights) * vols)

        # Build correlation matrix for active positions
        ticker_indices = [tickers.index(t) for t in common_tickers]
        active_weights = weights[ticker_indices]
        active_vols = vols[ticker_indices]
        active_corr = self.correlation_matrix.loc[common_tickers, common_tickers].values

        # Portfolio variance: w'Σw where Σ = D*C*D (vol matrix * corr * vol matrix)
        vol_matrix = np.outer(active_vols, active_vols) * active_corr
        portfolio_variance = np.dot(active_weights, np.dot(vol_matrix, active_weights))

        return np.sqrt(max(portfolio_variance, 0.0))

    def scale_positions_to_vol_target(self, positions: Dict[str, PositionSizeResult]) -> Dict[str, PositionSizeResult]:
        """
        Scale portfolio positions to meet volatility target

        Args:
            positions: Raw position sizes from σ-unit calculator

        Returns:
            Scaled positions meeting portfolio vol target
        """
        if not positions:
            return positions

        current_vol = self.calculate_portfolio_volatility(positions)

        if current_vol == 0:
            return positions

        # Calculate scaling factor to meet vol target
        target_vol_daily = self.constraints.portfolio_vol_target / np.sqrt(252)
        vol_scaling_factor = target_vol_daily / current_vol

        # Apply scaling with constraints
        scaled_positions = {}
        for ticker, position in positions.items():
            scaled_f = position.f * vol_scaling_factor

            # Apply individual position limits after scaling
            scaled_f = np.clip(scaled_f, -self.constraints.max_single_position, self.constraints.max_single_position)

            # Recalculate shares with scaled fraction
            if position.n_shares != 0 and position.dollar_amount != 0:
                estimated_price = abs(position.dollar_amount / position.n_shares)
            else:
                # Fallback to reasonable price estimate (placeholder for now)
                estimated_price = 100.0

            if position.f != 0 and not np.isnan(position.f):
                estimated_equity = abs(position.dollar_amount / position.f)
            else:
                estimated_equity = 100000  # Default fallback

            if estimated_price > 0 and not np.isnan(estimated_price):
                new_n_shares = int(np.round(scaled_f * estimated_equity / estimated_price))
                new_dollar_amount = scaled_f * estimated_equity
            else:
                new_n_shares = 0
                new_dollar_amount = 0.0

            scaled_positions[ticker] = PositionSizeResult(
                f=scaled_f,
                n_shares=new_n_shares,
                dollar_amount=new_dollar_amount,
                p_act_used=position.p_act_used,
                z_H_used=position.z_H_used,
                p_tail_used=position.p_tail_used,
                sigma_used=position.sigma_used,
                risk_controls_applied={
                    **position.risk_controls_applied,
                    "vol_target_scaling_applied": vol_scaling_factor != 1.0,
                    "post_scaling_f_max_applied": scaled_f != position.f * vol_scaling_factor,
                },
                ticker=ticker,
                timestamp=position.timestamp,
            )

        return scaled_positions

    def validate_portfolio_constraints(self, positions: Dict[str, PositionSizeResult]) -> PortfolioRiskMetrics:
        """
        Validate portfolio positions against risk constraints

        Args:
            positions: Portfolio positions to validate

        Returns:
            Portfolio risk metrics and constraint status
        """
        if not positions:
            return PortfolioRiskMetrics(
                estimated_portfolio_vol=0.0,
                vol_target_ratio=0.0,
                vol_constraint_satisfied=True,
                total_exposure=0.0,
                net_exposure=0.0,
                long_exposure=0.0,
                short_exposure=0.0,
                max_position_size=0.0,
                position_count=0,
                active_tickers=[],
                constraints_satisfied={},
                risk_budget_used=0.0,
            )

        # Calculate exposures
        f_values = [pos.f for pos in positions.values()]
        abs_f_values = [abs(f) for f in f_values]
        total_exposure = sum(abs_f_values)
        net_exposure = sum(f_values)
        long_exposure = sum(f for f in f_values if f > 0)
        short_exposure = abs(sum(f for f in f_values if f < 0))
        max_position = max(abs_f_values) if abs_f_values else 0.0

        portfolio_vol = self.calculate_portfolio_volatility(positions)
        annual_portfolio_vol = portfolio_vol * np.sqrt(252)
        vol_target_ratio = annual_portfolio_vol / self.constraints.portfolio_vol_target

        # Check constraints
        constraints_satisfied = {
            "vol_target": abs(vol_target_ratio - 1.0)
            <= (self.constraints.vol_tolerance / self.constraints.portfolio_vol_target),
            "max_single_position": max_position <= self.constraints.max_single_position,
            "max_total_exposure": total_exposure <= self.constraints.max_total_exposure,
            "max_net_exposure": abs(net_exposure) <= self.constraints.max_net_exposure,
        }

        risk_budget_used = annual_portfolio_vol / self.constraints.portfolio_vol_target

        return PortfolioRiskMetrics(
            estimated_portfolio_vol=annual_portfolio_vol,
            vol_target_ratio=vol_target_ratio,
            vol_constraint_satisfied=constraints_satisfied["vol_target"],
            total_exposure=total_exposure,
            net_exposure=net_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            max_position_size=max_position,
            position_count=len([p for p in positions.values() if p.f != 0]),
            active_tickers=[ticker for ticker, pos in positions.items() if pos.f != 0],
            constraints_satisfied=constraints_satisfied,
            risk_budget_used=risk_budget_used,
        )

    def apply_portfolio_risk_controls(self, positions: Dict[str, PositionSizeResult]) -> Dict[str, PositionSizeResult]:
        """
        Apply portfolio-level risk controls to position sizes

        Args:
            positions: Raw positions from σ-unit calculator

        Returns:
            Risk-adjusted positions meeting portfolio constraints
        """
        if not positions:
            return positions

        vol_scaled_positions = self.scale_positions_to_vol_target(positions)

        exposure_controlled_positions = self._apply_exposure_limits(vol_scaled_positions)

        final_metrics = self.validate_portfolio_constraints(exposure_controlled_positions)

        print(f"Portfolio risk management applied:")
        print(f"- Target vol: {self.constraints.portfolio_vol_target:.1%}")
        print(f"- Actual vol: {final_metrics.estimated_portfolio_vol:.1%}")
        print(f"- Vol ratio: {final_metrics.vol_target_ratio:.2f}")
        print(f"- Active positions: {final_metrics.position_count}")
        print(f"- Net exposure: {final_metrics.net_exposure:.1%}")
        print(f"- All constraints: {all(final_metrics.constraints_satisfied.values())}")

        return exposure_controlled_positions

    def _apply_exposure_limits(self, positions: Dict[str, PositionSizeResult]) -> Dict[str, PositionSizeResult]:
        """Apply portfolio exposure limits"""

        f_values = [pos.f for pos in positions.values()]
        total_exposure = sum(abs(f) for f in f_values)
        net_exposure = sum(f_values)

        # Check if total exposure exceeds limit
        if total_exposure > self.constraints.max_total_exposure:
            exposure_scale = self.constraints.max_total_exposure / total_exposure
            print(f"Scaling positions by {exposure_scale:.3f} for total exposure limit")

            # Scale all positions proportionally
            scaled_positions = {}
            for ticker, position in positions.items():
                scaled_f = position.f * exposure_scale

                # Recalculate other components
                if position.f != 0:
                    estimated_equity = abs(position.dollar_amount / position.f)
                    new_dollar_amount = scaled_f * estimated_equity
                    price = position.dollar_amount / (position.n_shares if position.n_shares != 0 else 1)
                    new_n_shares = int(np.round(new_dollar_amount / price))
                else:
                    new_dollar_amount = 0
                    new_n_shares = 0

                scaled_positions[ticker] = PositionSizeResult(
                    f=scaled_f,
                    n_shares=new_n_shares,
                    dollar_amount=new_dollar_amount,
                    p_act_used=position.p_act_used,
                    z_H_used=position.z_H_used,
                    p_tail_used=position.p_tail_used,
                    sigma_used=position.sigma_used,
                    risk_controls_applied={**position.risk_controls_applied, "exposure_limit_scaling_applied": True},
                    ticker=ticker,
                    timestamp=position.timestamp,
                )

            return scaled_positions

        return positions

    def monitor_portfolio_risk(
        self, positions: Dict[str, PositionSizeResult], price_history: Optional[Dict[str, pd.Series]] = None
    ) -> Dict:
        """
        Monitor real-time portfolio risk metrics

        Args:
            positions: Current portfolio positions
            price_history: Historical price data for drawdown calculation

        Returns:
            Risk monitoring report
        """
        metrics = self.validate_portfolio_constraints(positions)

        risk_alerts = []

        # Check vol target
        if not metrics.vol_constraint_satisfied:
            vol_deviation = abs(metrics.vol_target_ratio - 1.0)
            risk_alerts.append(f"Portfolio vol deviation: {vol_deviation:.1%}")

        if metrics.max_position_size > self.constraints.max_single_position:
            risk_alerts.append(f"Position size limit exceeded: {metrics.max_position_size:.1%}")

        if metrics.total_exposure > self.constraints.max_total_exposure:
            risk_alerts.append(f"Total exposure limit exceeded: {metrics.total_exposure:.1%}")

        if abs(metrics.net_exposure) > self.constraints.max_net_exposure:
            risk_alerts.append(f"Net exposure limit exceeded: {abs(metrics.net_exposure):.1%}")

        # Portfolio drawdown monitoring
        current_drawdown = None
        if price_history:
            current_drawdown = self._calculate_portfolio_drawdown(positions, price_history)
            if current_drawdown and abs(current_drawdown) > self.constraints.max_drawdown_limit:
                risk_alerts.append(f"Drawdown limit exceeded: {abs(current_drawdown):.1%}")

        return {
            "timestamp": pd.Timestamp.now(),
            "portfolio_metrics": metrics,
            "risk_alerts": risk_alerts,
            "current_drawdown": current_drawdown,
            "risk_status": "GREEN" if not risk_alerts else "YELLOW" if len(risk_alerts) <= 2 else "RED",
        }

    def _calculate_portfolio_drawdown(
        self, positions: Dict[str, PositionSizeResult], price_history: Dict[str, pd.Series]
    ) -> Optional[float]:
        """Calculate current portfolio drawdown"""

        try:
            # Simple implementation for now - calculate based on position P&L
            total_pnl = 0.0
            total_initial_value = 0.0

            for ticker, position in positions.items():
                if position.n_shares != 0 and ticker in price_history:
                    prices = price_history[ticker]
                    if len(prices) >= 2:
                        initial_price = prices.iloc[-2]  # Previous day
                        current_price = prices.iloc[-1]  # Current day

                        pnl = position.n_shares * (current_price - initial_price)
                        initial_value = abs(position.n_shares * initial_price)

                        total_pnl += pnl
                        total_initial_value += initial_value

            if total_initial_value > 0:
                return total_pnl / total_initial_value

        except Exception as e:
            print(f"Warning: Could not calculate portfolio drawdown: {e}")

        return None


def create_conservative_risk_manager() -> PortfolioRiskManager:
    """
    Create conservative portfolio risk manager for production testing


    Returns:
        PortfolioRiskManager with conservative constraints
    """
    constraints = RiskConstraints(
        portfolio_vol_target=0.10,  # 10% annual vol target
        vol_tolerance=0.015,  # 1.5% tolerance
        max_single_position=0.12,  # 12% max single position
        max_total_exposure=0.8,  # 80% max total exposure
        max_net_exposure=0.6,  # 60% max net exposure
        max_drawdown_limit=0.12,  # 12% max drawdown
    )

    return PortfolioRiskManager(constraints)


def create_aggressive_risk_manager() -> PortfolioRiskManager:
    """
    Create aggressive portfolio risk manager for research

    Returns:
        PortfolioRiskManager with aggressive constraints
    """
    constraints = RiskConstraints(
        portfolio_vol_target=0.15,  # 15% annual vol target
        vol_tolerance=0.03,  # 3% tolerance
        max_single_position=0.20,  # 20% max single position
        max_total_exposure=1.2,  # 120% max total exposure
        max_net_exposure=1.0,  # 100% max net exposure
        max_drawdown_limit=0.20,  # 20% max drawdown
    )

    return PortfolioRiskManager(constraints)



