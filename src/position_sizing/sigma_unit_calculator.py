#!/usr/bin/env python3

"""
Sigma Unit Position Sizing Calculator
Implements f = αp_act·max(z_H-θ,0)/(1+βp_tail) formula with risk controls
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class SigmaUnitParams(BaseModel):
    """Parameters for σ-unit position sizing calculation"""
    
    alpha: float = Field(default=1.0, ge=0.0, description="Action probability scaling factor")
    theta: float = Field(default=0.0, ge=0.0, description="Signal strength threshold") 
    beta: float = Field(default=1.0, ge=0.0, description="Tail risk adjustment factor")
    
    # Risk controls
    sigma_floor: float = Field(default=0.005, gt=0.0, description="Minimum volatility for numerical stability")
    f_max: float = Field(default=0.20, gt=0.0, le=1.0, description="Maximum position fraction per asset")
    
    # Numerical safeguards
    p_act_floor: float = Field(default=0.01, gt=0.0, lt=1.0, description="Minimum p_act for division stability")
    p_tail_cap: float = Field(default=0.95, gt=0.0, lt=1.0, description="Maximum p_tail to prevent extreme denominators")


@dataclass
class PositionSizeResult:
    """Result of σ-unit position sizing calculation"""
    
    f: float  # Position fraction
    n_shares: int  # Number of shares
    dollar_amount: float  # Dollar position size
    
    # Intermediate calculations for transparency
    p_act_used: float
    z_H_used: float
    p_tail_used: float
    sigma_used: float
    
    # Risk controls applied
    risk_controls_applied: Dict[str, bool]
    
    # Metadata
    ticker: str
    timestamp: pd.Timestamp


class SigmaUnitCalculator:
    """
    σ-unit position sizing calculator implementing MVS specification
    
    Core formula: f = αp_act·max(z_H-θ,0)/(1+βp_tail)
    Share conversion: n = f·equity/(price·σ)
    """
    
    def __init__(self, params: SigmaUnitParams):
        """
        Initialize σ-unit calculator
        
        Args:
            params: σ-unit calculation parameters
        """
        self.params = params
        
        print(f"σ-Unit Calculator initialized:")
        print(f"  - α={params.alpha}, θ={params.theta}, β={params.beta}")
        print(f"  - σ-floor={params.sigma_floor}, f_max={params.f_max}")
    
    def calculate_position_size(
        self,
        p_act: float,
        z_H: float,
        p_tail: float,
        sigma: float,
        price: float,
        equity: float,
        ticker: str,
        timestamp: Optional[pd.Timestamp] = None
    ) -> PositionSizeResult:
        """
        Calculate σ-unit position size using MVS formula
        
        Args:
            p_act: Action probability [0,1] from model
            z_H: Signal strength in σ-units (standardized)
            p_tail: Tail risk probability [0,1] from model
            sigma: Current volatility estimate
            price: Current asset price
            equity: Available equity for position sizing
            ticker: Asset ticker symbol
            timestamp: Calculation timestamp
            
        Returns:
            PositionSizeResult with position size and metadata
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Track which risk controls are applied
        risk_controls = {
            'sigma_floor_applied': False,
            'p_act_floor_applied': False, 
            'p_tail_cap_applied': False,
            'f_max_clipped': False,
            'negative_z_H_zeroed': False
        }
        
        # Apply risk controls to inputs
        sigma_used = max(sigma, self.params.sigma_floor)
        if sigma_used != sigma:
            risk_controls['sigma_floor_applied'] = True
        
        p_act_used = max(p_act, self.params.p_act_floor)
        if p_act_used != p_act:
            risk_controls['p_act_floor_applied'] = True
        
        p_tail_used = min(p_tail, self.params.p_tail_cap)
        if p_tail_used != p_tail:
            risk_controls['p_tail_cap_applied'] = True
        
        # Calculate signal strength component: max(z_H-θ,0)
        z_H_component = max(z_H - self.params.theta, 0.0)
        if z_H_component == 0.0 and z_H > 0:
            risk_controls['negative_z_H_zeroed'] = True
        
        # Calculate position fraction: f = αp_act·max(z_H-θ,0)/(1+βp_tail)
        numerator = self.params.alpha * p_act_used * z_H_component
        denominator = 1.0 + self.params.beta * p_tail_used
        
        f_raw = numerator / denominator if denominator != 0 else 0.0
        
        # Apply position limits
        f_clipped = np.clip(f_raw, -self.params.f_max, self.params.f_max)
        if abs(f_clipped) != abs(f_raw):
            risk_controls['f_max_clipped'] = True
        
        # Convert to shares: n = f·equity/(price·σ)
        dollar_amount = f_clipped * equity
        notional_position = dollar_amount / sigma_used  # Risk-adjusted notional
        n_shares_float = notional_position / price
        n_shares = int(np.round(n_shares_float))
        
        # Adjust dollar amount for discrete shares
        actual_dollar_amount = n_shares * price
        
        return PositionSizeResult(
            f=f_clipped,
            n_shares=n_shares,
            dollar_amount=actual_dollar_amount,
            p_act_used=p_act_used,
            z_H_used=z_H,
            p_tail_used=p_tail_used,
            sigma_used=sigma_used,
            risk_controls_applied=risk_controls,
            ticker=ticker,
            timestamp=timestamp
        )
    
    def calculate_portfolio_positions(
        self,
        model_outputs: Dict[str, Dict[str, float]],
        market_data: Dict[str, Dict[str, float]],
        equity: float
    ) -> Dict[str, PositionSizeResult]:
        """
        Calculate position sizes for entire portfolio
        
        Args:
            model_outputs: {ticker: {p_act, z_H, p_tail}} from σ-unit model
            market_data: {ticker: {price, sigma}} current market data
            equity: Total available equity
            
        Returns:
            Dictionary of {ticker: PositionSizeResult}
        """
        portfolio_positions = {}
        
        for ticker in model_outputs:
            if ticker in market_data:
                outputs = model_outputs[ticker]
                market = market_data[ticker]
                
                position = self.calculate_position_size(
                    p_act=outputs['p_act'],
                    z_H=outputs['z_H'],
                    p_tail=outputs['p_tail'],
                    sigma=market['sigma'],
                    price=market['price'],
                    equity=equity,
                    ticker=ticker
                )
                
                portfolio_positions[ticker] = position
        
        return portfolio_positions
    
    def validate_calculation_inputs(
        self,
        p_act: float,
        z_H: float, 
        p_tail: float,
        sigma: float,
        price: float,
        equity: float
    ) -> Dict[str, bool]:
        """
        Validate inputs for position sizing calculation
        
        Returns:
            Dictionary of validation results
        """
        validations = {
            'p_act_valid': 0.0 <= p_act <= 1.0,
            'z_H_finite': np.isfinite(z_H),
            'p_tail_valid': 0.0 <= p_tail <= 1.0,
            'sigma_positive': sigma > 0,
            'price_positive': price > 0,
            'equity_positive': equity > 0
        }
        
        return validations
    
    def get_sizing_summary(self, positions: Dict[str, PositionSizeResult]) -> Dict:
        """
        Get portfolio position sizing summary
        
        Args:
            positions: Dictionary of position results
            
        Returns:
            Summary statistics
        """
        if not positions:
            return {'total_positions': 0}
        
        f_values = [pos.f for pos in positions.values()]
        dollar_values = [pos.dollar_amount for pos in positions.values()]
        
        # Count risk controls applied
        risk_control_counts = {}
        for control_type in ['sigma_floor_applied', 'p_act_floor_applied', 
                           'p_tail_cap_applied', 'f_max_clipped', 'negative_z_H_zeroed']:
            risk_control_counts[control_type] = sum(
                1 for pos in positions.values() 
                if pos.risk_controls_applied.get(control_type, False)
            )
        
        return {
            'total_positions': len(positions),
            'active_positions': sum(1 for pos in positions.values() if pos.f != 0),
            'avg_position_fraction': np.mean([abs(f) for f in f_values]),
            'max_position_fraction': max([abs(f) for f in f_values]) if f_values else 0,
            'total_dollar_exposure': sum([abs(d) for d in dollar_values]),
            'net_dollar_exposure': sum(dollar_values),
            'long_positions': sum(1 for f in f_values if f > 0),
            'short_positions': sum(1 for f in f_values if f < 0),
            'risk_controls_applied': risk_control_counts
        }


def create_conservative_sigma_calculator() -> SigmaUnitCalculator:
    """
    Create σ-unit calculator with conservative parameters for production use
    
    Returns:
        Configured SigmaUnitCalculator with conservative risk controls
    """
    params = SigmaUnitParams(
        alpha=0.5,        # Conservative action scaling
        theta=0.5,        # Moderate signal threshold  
        beta=2.0,         # Strong tail risk adjustment
        sigma_floor=0.01, # 1% minimum volatility
        f_max=0.15,       # 15% maximum position
        p_act_floor=0.05, # 5% minimum action probability
        p_tail_cap=0.90   # 90% maximum tail probability
    )
    
    return SigmaUnitCalculator(params)


def create_aggressive_sigma_calculator() -> SigmaUnitCalculator:
    """
    Create σ-unit calculator with aggressive parameters for research
    
    Returns:
        Configured SigmaUnitCalculator with aggressive parameters
    """
    params = SigmaUnitParams(
        alpha=1.5,        # Aggressive action scaling
        theta=0.2,        # Lower signal threshold
        beta=1.0,         # Moderate tail risk adjustment
        sigma_floor=0.005, # 0.5% minimum volatility
        f_max=0.25,       # 25% maximum position
        p_act_floor=0.01, # 1% minimum action probability
        p_tail_cap=0.95   # 95% maximum tail probability
    )
    
    return SigmaUnitCalculator(params)


if __name__ == "__main__":
    # Test σ-unit calculator
    print("Testing σ-Unit Position Sizing Calculator")
    
    calculator = create_conservative_sigma_calculator()
    
    # Test sample calculation
    result = calculator.calculate_position_size(
        p_act=0.7,      # 70% action probability
        z_H=1.5,        # 1.5σ signal strength
        p_tail=0.2,     # 20% tail risk
        sigma=0.02,     # 2% daily volatility
        price=150.0,    # $150 stock price
        equity=100000,  # $100k equity
        ticker="AAPL"
    )
    
    print(f"\nSample calculation:")
    print(f"  - Position fraction: {result.f:.4f}")
    print(f"  - Shares: {result.n_shares}")
    print(f"  - Dollar amount: ${result.dollar_amount:,.2f}")
    print(f"  - Risk controls: {result.risk_controls_applied}")
    
    # Test input validation
    validations = calculator.validate_calculation_inputs(
        p_act=0.7, z_H=1.5, p_tail=0.2, sigma=0.02, price=150.0, equity=100000
    )
    print(f"  - Input validation: {all(validations.values())}")