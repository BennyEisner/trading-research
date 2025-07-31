#!/usr/bin/env python3

"""
Pattern Features Processor
Integrates pattern feature calculation with existing BaseFeatureProcessor framework
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import FEATURE_REGISTRY, BaseFeatureProcessor, FeatureGroup
from ..pattern_feature_calculator import FeatureCalculator


class PatternFeaturesProcessor(BaseFeatureProcessor):
    """
    Pattern features processor following BaseFeatureProcessor interface
    Integrates with existing feature engineering pipeline
    """

    def __init__(
        self,
        market_data: Optional[pd.DataFrame] = None,
        sector_data: Optional[Dict[str, pd.DataFrame]] = None,
        vix_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize pattern features processor

        Args:
            market_data: Market benchmark data (e.g., SPY)
            sector_data: Sector ETF data for cross-asset features
            vix_data: VIX term structure data
        """
        super().__init__("pattern_features")

        self.market_data = market_data
        self.sector_data = sector_data or {}
        self.vix_data = vix_data

        # Define feature names produced by this processor
        self.feature_names = [
            # Non-linear price patterns
            "price_acceleration",
            "volume_price_divergence",
            "volatility_regime_change",
            "return_skewness_7d",
            # Temporal dependencies
            "momentum_persistence_7d",
            "volatility_clustering",
            "trend_exhaustion",
            "garch_volatility_forecast",
            # Market microstructure
            "intraday_range_expansion",
            "overnight_gap_behavior",
            "end_of_day_momentum",
            # Cross-asset relationships
            "sector_relative_strength",
            "market_beta_instability",
            "vix_term_structure",
            # Core context
            "returns_1d",
            "returns_3d",
            "returns_7d",
            "volume_normalized",
            "close",
        ]

        # Define dependencies (required input columns)
        self.dependencies = ["open", "high", "low", "close", "volume"]

        # Register feature groups
        self._register_feature_groups()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pattern features and add to DataFrame

        Args:
            data: Input DataFrame with OHLCV data

        Returns:
            DataFrame with original data plus pattern features
        """

        # Validate dependencies exist
        self.validate_dependencies(data)

        # Determine symbol from data if available
        symbol = getattr(data, "symbol", "UNKNOWN")

        # Get sector data for this symbol if available
        symbol_sector_data = self.sector_data.get(symbol)

        # Create feature calculator
        calculator = FeatureCalculator(
            symbol=symbol, market_data=self.market_data, sector_data=symbol_sector_data, vix_data=self.vix_data
        )

        # Calculate pattern features
        pattern_features = calculator.calculate_all_features(data)

        # Combine with original data
        result = data.copy()

        # Add pattern features to result
        for feature_name in self.feature_names:
            if feature_name in pattern_features.columns:
                result[feature_name] = pattern_features[feature_name]
            else:
                # Handle missing features gracefully
                print(f"Warning: Feature {feature_name} not calculated, filling with zeros")
                result[feature_name] = 0.0

        return result

    def _register_feature_groups(self):
        """Register feature groups with the global registry"""

        # Non-linear price patterns group
        nonlinear_group = FeatureGroup(
            "nonlinear_price_patterns", "Non-linear price pattern features for temporal modeling"
        )
        nonlinear_group.add_features(
            ["price_acceleration", "volume_price_divergence", "volatility_regime_change", "return_skewness_7d"]
        )
        nonlinear_group.set_processor(self)
        FEATURE_REGISTRY.register_group(nonlinear_group)

        # Temporal dependencies group
        temporal_group = FeatureGroup("temporal_dependencies", "Temporal dependency features optimized for LSTM")
        temporal_group.add_features(
            ["momentum_persistence_7d", "volatility_clustering", "trend_exhaustion", "garch_volatility_forecast"]
        )
        temporal_group.set_processor(self)
        FEATURE_REGISTRY.register_group(temporal_group)

        # Market microstructure group
        microstructure_group = FeatureGroup("market_microstructure", "Market microstructure and intraday patterns")
        microstructure_group.add_features(["intraday_range_expansion", "overnight_gap_behavior", "end_of_day_momentum"])
        microstructure_group.set_processor(self)
        FEATURE_REGISTRY.register_group(microstructure_group)

        # Cross-asset relationships group
        cross_asset_group = FeatureGroup("cross_asset_relationships", "Cross-asset and market relationship features")
        cross_asset_group.add_features(["sector_relative_strength", "market_beta_instability", "vix_term_structure"])
        cross_asset_group.set_processor(self)
        FEATURE_REGISTRY.register_group(cross_asset_group)

        # Core context group
        core_context_group = FeatureGroup("core_context", "Core context features for baseline information")
        core_context_group.add_features(["returns_1d", "returns_3d", "returns_7d", "volume_normalized", "close"])
        core_context_group.set_processor(self)
        FEATURE_REGISTRY.register_group(core_context_group)

        # Register processor
        FEATURE_REGISTRY.register_processor(self)

    def get_feature_importance(self) -> Dict[str, str]:
        """Get feature importance categories for selection"""

        return {
            # High importance - core pattern features
            "price_acceleration": "high",
            "volatility_clustering": "high",
            "momentum_persistence_7d": "high",
            "garch_volatility_forecast": "high",
            # Medium importance - contextual features
            "volume_price_divergence": "medium",
            "volatility_regime_change": "medium",
            "trend_exhaustion": "medium",
            "intraday_range_expansion": "medium",
            "overnight_gap_behavior": "medium",
            "end_of_day_momentum": "medium",
            # Lower importance - cross-asset (data dependent)
            "sector_relative_strength": "low",
            "market_beta_instability": "low",
            "vix_term_structure": "low",
            # Context features
            "return_skewness_7d": "medium",
            "returns_1d": "high",
            "returns_3d": "medium",
            "returns_7d": "medium",
            "volume_normalized": "medium",
            "close": "low",  # Context only
        }

    def validate_external_data(self) -> Dict[str, bool]:
        """Validate availability of external data dependencies"""

        validation = {
            "market_data_available": self.market_data is not None,
            "sector_data_available": len(self.sector_data) > 0,
            "vix_data_available": self.vix_data is not None,
        }

        # Log warnings for missing data
        if not validation["market_data_available"]:
            print("Warning: No market data provided - market beta features will be zero")

        if not validation["sector_data_available"]:
            print("Warning: No sector data provided - sector relative strength will be zero")

        if not validation["vix_data_available"]:
            print("Warning: No VIX data provided - VIX term structure will be zero")

        return validation

    def get_processor_info(self) -> Dict[str, any]:
        """Get information about this processor"""

        external_data_status = self.validate_external_data()

        return {
            "name": self.name,
            "feature_count": len(self.feature_names),
            "feature_groups": [
                "nonlinear_price_patterns",
                "temporal_dependencies",
                "market_microstructure",
                "cross_asset_relationships",
                "core_context",
            ],
            "dependencies": self.dependencies,
            "external_data_status": external_data_status,
            "supported_features": self.feature_names,
        }


def create_pattern_processor(
    market_data: Optional[pd.DataFrame] = None,
    sector_data: Optional[Dict[str, pd.DataFrame]] = None,
    vix_data: Optional[pd.DataFrame] = None,
) -> PatternFeaturesProcessor:
    """Convenience function to create pattern features processor"""

    return PatternFeaturesProcessor(market_data=market_data, sector_data=sector_data, vix_data=vix_data)


# Example usage and testing
if __name__ == "__main__":

    # Create synthetic test data
    dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="D")
    n_days = len(dates)

    # Synthetic OHLCV data
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(n_days) * 0.02)

    test_data = pd.DataFrame(
        {
            "open": prices + np.random.randn(n_days) * 0.01,
            "high": prices + np.abs(np.random.randn(n_days) * 0.015),
            "low": prices - np.abs(np.random.randn(n_days) * 0.015),
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n_days),
        },
        index=dates,
    )

    # Add symbol attribute for testing
    test_data.symbol = "TEST"

    # Create market data (synthetic SPY)
    spy_prices = 300 + np.cumsum(np.random.randn(n_days) * 0.015)
    market_data = pd.DataFrame({"close": spy_prices}, index=dates)

    # Create processor
    processor = create_pattern_processor(market_data=market_data)

    # Test processor info
    info = processor.get_processor_info()
    print(f"Processor info: {info}")

    # Calculate features
    result = processor.calculate(test_data)

    print(f"Original shape: {test_data.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Added features: {len(processor.feature_names)}")

    # Check feature registry
    print(f"Registered features: {len(FEATURE_REGISTRY.get_all_features())}")
    print(f"Feature groups: {list(FEATURE_REGISTRY.groups.keys())}")

