#!/usr/bin/env python3

"""
LSTM Pattern Verification Strategy
Converts LSTM pattern confidence scores to strategy signals for ensemble integration
Designed as meta-labeling filter rather than standalone strategy
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import Field

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ..base import BaseStrategy, StrategyConfig
from ...models.shared_backbone_lstm import SharedBackboneLSTMBuilder
from ...features.multi_ticker_engine import MultiTickerPatternEngine
from config.config import get_config


class LSTMPatternStrategyConfig(StrategyConfig):
    """Configuration for LSTM Pattern Verification Strategy"""

    name: str = "lstm_pattern_verification"

    # Pattern confidence thresholds
    min_pattern_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    strong_pattern_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Signal generation modes
    mode: str = Field(default="verification", pattern="^(standalone|verification|enhancement)$")
    
    # Enhanced signal integration parameters
    enhancement_factor: float = Field(default=0.3, ge=0.0, le=1.0, 
                                    description="Factor for enhancing signals with strong patterns")
    dampening_factor: float = Field(default=0.5, ge=0.0, le=1.0,
                                   description="Factor for dampening signals with weak patterns")
    use_enhanced_integration: bool = Field(default=True,
                                          description="Use enhanced additive integration instead of multiplicative")
    
    # Pattern-based position sizing
    confidence_scaling: bool = Field(default=True)
    max_confidence_position: float = Field(default=1.0, ge=0.1, le=1.0)
    
    # Model parameters
    model_path: Optional[str] = Field(default=None)
    lookback_window: int = Field(default=20, ge=10, le=50)
    
    # Pattern interpretation
    bullish_threshold: float = Field(default=0.6, ge=0.5, le=1.0)
    bearish_threshold: float = Field(default=0.4, ge=0.0, le=0.5)


class LSTMPatternStrategy(BaseStrategy):
    """LSTM Pattern Verification Strategy
    
    Modes:
        - standalone: Generate signals based purely on LSTM pattern confidence
        - verification: Act as filter for other strategies (multiply signals)
        - enhancement: Add pattern confidence as signal strength multiplier
    
    Logic:
        - Pattern confidence > bullish_threshold → Long bias
        - Pattern confidence < bearish_threshold → Short bias  
        - Between thresholds → Neutral (no position in standalone mode)
        - In verification mode: multiply other signals by confidence
    """

    def __init__(self, config: LSTMPatternStrategyConfig):
        self.config: LSTMPatternStrategyConfig = config
        
        # Initialize components first
        self.system_config = get_config()
        self.model = None
        self.pattern_engine = None
        
        # Performance tracking specific to pattern verification
        self.verification_stats = {
            'signals_filtered': 0,
            'signals_enhanced': 0,
            'avg_confidence': 0.0,
            'high_confidence_signals': 0
        }
        
        # Initialize model before calling super().__init__ which calls validate_parameters
        self._initialize_model()
        
        # Now call parent initialization
        super().__init__(config)
        
    def _initialize_model(self):
        """Initialize LSTM model and pattern engine"""
        try:
            # Priority order for finding trained models
            model_candidates = [
                self.config.model_path,  # User-specified path
                "models/best_shared_backbone_model.keras",  # Best available model
                "models/trained/best_directional_lstm_model.keras",  # Alternative trained model
            ]
            
            # Try to load existing trained model
            model_loaded = False
            for candidate_path in model_candidates:
                if candidate_path and Path(candidate_path).exists():
                    try:
                        # Try multiple loading approaches
                        model = None
                        
                        # Approach 1: Load with custom objects
                        try:
                            custom_objects = {
                                'directional_mse_loss': self._dummy_loss_function,
                                '_pattern_detection_accuracy': self._dummy_metric_function,
                                'correlation_metric': self._dummy_correlation_metric,
                            }
                            model = tf.keras.models.load_model(candidate_path, custom_objects=custom_objects)
                        except Exception as e1:
                            print(f"   Custom objects approach failed: {e1}")
                            
                            # Approach 2: Load only weights and architecture
                            try:
                                model = tf.keras.models.load_model(candidate_path, compile=False)
                                # Recompile with simple loss and optimizer for inference
                                model.compile(
                                    optimizer='adam',
                                    loss='mse',
                                    metrics=['mae']
                                )
                            except Exception as e2:
                                print(f"   Architecture-only approach failed: {e2}")
                                raise e2  # Re-raise to trigger the outer exception handler
                        
                        if model is not None:
                            self.model = model
                            print(f"✅ Loaded trained LSTM model from: {candidate_path}")
                            model_loaded = True
                            break
                            
                    except Exception as e:
                        print(f"⚠️  Failed to load model from {candidate_path}: {e}")
                        continue
            
            if not model_loaded:
                # Create new model (would need training)
                lstm_builder = SharedBackboneLSTMBuilder(self.system_config.dict())
                input_shape = (self.config.lookback_window, 16)  # 16 pattern features
                self.model = lstm_builder.build_model(input_shape)
                print(f"⚠️  Created new untrained LSTM model - accuracy will be random until trained")
            
            # Initialize pattern engine for feature generation
            self.pattern_engine = MultiTickerPatternEngine(
                tickers=["TEMP"],  # Will be set dynamically
                max_workers=1
            )
            
            print(f"✅ LSTM Pattern Strategy initialized in '{self.config.mode}' mode")
            
        except Exception as e:
            print(f"❌ Failed to initialize LSTM model: {e}")
            self.model = None

    def _dummy_loss_function(self, y_true, y_pred):
        """Dummy loss function for model loading compatibility"""
        return tf.keras.losses.mse(y_true, y_pred)
    
    def _dummy_metric_function(self, y_true, y_pred):
        """Dummy metric function for model loading compatibility"""
        return tf.keras.metrics.binary_accuracy(y_true, y_pred)
    
    def _dummy_correlation_metric(self, y_true, y_pred):
        """Dummy correlation metric for model loading compatibility"""
        return tf.constant(0.0)

    def _apply_pattern_enhancement(self, original_signal: float, pattern_confidence: float) -> float:
        """
        Apply enhanced signal integration based on pattern confidence
        
        Enhanced logic:
        - Strong patterns (confidence > 0.5): Enhance signals additively
        - Weak patterns (confidence < 0.5): Dampen signals multiplicatively
        - Neutral patterns (confidence ≈ 0.5): Preserve original signal
        
        Args:
            original_signal: Original signal strength [-1, 1]
            pattern_confidence: Pattern confidence [0, 1]
            
        Returns:
            Enhanced signal strength
        """
        if not self.config.use_enhanced_integration:
            # Fall back to original multiplicative approach if disabled
            return original_signal * pattern_confidence
        
        if pattern_confidence > 0.5:
            # Strong pattern: enhance signal additively
            # Enhancement = signal + (confidence - 0.5) * enhancement_factor * |signal|
            enhancement = (pattern_confidence - 0.5) * self.config.enhancement_factor * abs(original_signal)
            if original_signal >= 0:
                return original_signal + enhancement
            else:
                return original_signal - enhancement
                
        elif pattern_confidence < 0.5:
            # Weak pattern: dampen signal multiplicatively
            # Dampening = signal * (1 - (0.5 - confidence) * dampening_factor)
            dampening_multiplier = 1 - (0.5 - pattern_confidence) * self.config.dampening_factor
            return original_signal * dampening_multiplier
            
        else:
            # Neutral pattern (confidence ≈ 0.5): preserve original signal
            return original_signal

    def _validate_enhancement_parameters(self) -> None:
        """Validate enhanced integration parameters"""
        if self.config.enhancement_factor < 0 or self.config.enhancement_factor > 1:
            raise ValueError(f"Enhancement factor must be between 0 and 1, got {self.config.enhancement_factor}")
        
        if self.config.dampening_factor < 0 or self.config.dampening_factor > 1:
            raise ValueError(f"Dampening factor must be between 0 and 1, got {self.config.dampening_factor}")
        
        if self.config.min_pattern_confidence < 0 or self.config.min_pattern_confidence > 1:
            raise ValueError(f"Min pattern confidence must be between 0 and 1, got {self.config.min_pattern_confidence}")

    def _log_integration_statistics(self, original_positions: pd.Series, 
                                   enhanced_positions: pd.Series,
                                   pattern_confidence: pd.Series) -> None:
        """Log statistics about signal integration for monitoring"""
        
        non_zero_mask = original_positions.abs() > 1e-6
        
        if non_zero_mask.sum() > 0:
            # Calculate enhancement/dampening statistics
            enhancement_ratio = (enhanced_positions[non_zero_mask].abs() / 
                               original_positions[non_zero_mask].abs()).mean()
            
            strong_patterns = (pattern_confidence > 0.5).sum()
            weak_patterns = (pattern_confidence < 0.5).sum()
            neutral_patterns = ((pattern_confidence >= 0.5) & (pattern_confidence <= 0.5)).sum()
            
            print(f"   📊 Signal Integration Stats:")
            print(f"      Enhancement ratio: {enhancement_ratio:.3f}")
            print(f"      Strong patterns: {strong_patterns} ({strong_patterns/len(pattern_confidence)*100:.1f}%)")
            print(f"      Weak patterns: {weak_patterns} ({weak_patterns/len(pattern_confidence)*100:.1f}%)")
            print(f"      Signals filtered: {(enhanced_positions == 0).sum()}")
            print(f"      Mean confidence: {pattern_confidence.mean():.3f}")
        else:
            print(f"   📊 Signal Integration Stats: No non-zero signals to analyze")

    def get_required_features(self) -> List[str]:
        """Required features for LSTM pattern verification"""
        base_features = super().get_required_features()
        
        # Pattern features used by LSTM model
        pattern_features = [
            "price_acceleration",
            "volume_price_divergence", 
            "volatility_regime_change",
            "return_skewness_7d",
            "momentum_persistence_7d",
            "volatility_clustering",
            "trend_exhaustion",
            "garch_volatility_forecast",
            "intraday_range_expansion",
            "overnight_gap_behavior",
            "end_of_day_momentum",
            "sector_relative_strength",
            "market_beta_instability",
            # "vix_term_structure",  # Removed - not available in backtest environment
            "returns_1d",
            "returns_3d", 
            "returns_7d"
        ]
        
        return base_features + pattern_features

    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if self.config.bullish_threshold <= self.config.bearish_threshold:
            raise ValueError("Bullish threshold must be greater than bearish threshold")
        
        if self.config.min_pattern_confidence > self.config.strong_pattern_threshold:
            raise ValueError("Min confidence cannot exceed strong pattern threshold")
        
        # Validate enhanced integration parameters
        self._validate_enhancement_parameters()
            
        if self.model is None:
            print("⚠️  Warning: No trained LSTM model available")
            
        return True

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate LSTM pattern-based signals"""
        
        signals = pd.DataFrame(index=data.index)
        signals["position"] = 0.0
        signals["entry_price"] = np.nan
        signals["stop_loss"] = np.nan 
        signals["take_profit"] = np.nan
        
        if self.model is None:
            print("❌ No LSTM model available for signal generation")
            return signals
            
        try:
            # Generate pattern confidence scores
            pattern_confidence = self._calculate_pattern_confidence(data)
            
            if pattern_confidence is None:
                return signals
                
            # Generate signals based on mode
            if self.config.mode == "standalone":
                signals = self._generate_standalone_signals(data, pattern_confidence, signals)
            elif self.config.mode == "verification":
                signals = self._generate_verification_signals(data, pattern_confidence, signals)
            elif self.config.mode == "enhancement": 
                signals = self._generate_enhancement_signals(data, pattern_confidence, signals)
                
            return signals
            
        except Exception as e:
            print(f"❌ Error generating LSTM signals: {e}")
            return signals

    def _calculate_pattern_confidence(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate LSTM pattern confidence scores"""
        
        try:
            # Check if we have required features
            required_features = [
                "price_acceleration", "volume_price_divergence", "volatility_regime_change",
                "return_skewness_7d", "momentum_persistence_7d", "volatility_clustering",
                "trend_exhaustion", "garch_volatility_forecast", "intraday_range_expansion",
                "overnight_gap_behavior", "end_of_day_momentum", "sector_relative_strength", 
                "market_beta_instability", "returns_1d", "returns_3d", "returns_7d"
            ]
            
            available_features = [f for f in required_features if f in data.columns]
            
            if len(available_features) < 14:
                print(f"❌ Insufficient pattern features: {len(available_features)}/16 available")
                return None
                
            # Prepare sequences for LSTM prediction
            sequences = self._prepare_sequences(data[available_features])
            
            if len(sequences) == 0:
                print("❌ No valid sequences for LSTM prediction")
                return None
                
            # Generate predictions
            predictions = self.model.predict(sequences, verbose=0)
            
            # Create confidence series aligned with data index
            confidence_series = pd.Series(0.0, index=data.index)
            
            # Map predictions to corresponding timestamps
            sequence_length = self.config.lookback_window
            for i, pred in enumerate(predictions.flatten()):
                data_idx = sequence_length + i
                if data_idx < len(data):
                    confidence_series.iloc[data_idx] = float(pred)
                    
            # Update stats
            self.verification_stats['avg_confidence'] = confidence_series.mean()
            self.verification_stats['high_confidence_signals'] = (confidence_series > self.config.strong_pattern_threshold).sum()
            
            return confidence_series
            
        except Exception as e:
            print(f"❌ Error calculating pattern confidence: {e}")
            return None

    def _prepare_sequences(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare sequences for LSTM model input"""
        
        # Clean data
        features_clean = features_df.ffill().bfill().fillna(0.0)
        
        sequences = []
        sequence_length = self.config.lookback_window
        
        for i in range(sequence_length, len(features_clean)):
            sequence = features_clean.iloc[i-sequence_length:i].values
            sequences.append(sequence)
            
        return np.array(sequences)

    def _generate_standalone_signals(self, data: pd.DataFrame, confidence: pd.Series, signals: pd.DataFrame) -> pd.DataFrame:
        """Generate standalone signals based on pattern confidence"""
        
        # Long signals when confidence > bullish threshold
        long_mask = confidence > self.config.bullish_threshold
        signals.loc[long_mask, "position"] = 1.0
        signals.loc[long_mask, "entry_price"] = data.loc[long_mask, "close"]
        
        # Short signals when confidence < bearish threshold  
        short_mask = confidence < self.config.bearish_threshold
        signals.loc[short_mask, "position"] = -1.0
        signals.loc[short_mask, "entry_price"] = data.loc[short_mask, "close"]
        
        # Scale by confidence if enabled
        if self.config.confidence_scaling:
            # For longs: scale by how much confidence exceeds bullish threshold
            long_scaling = ((confidence - self.config.bullish_threshold) / (1.0 - self.config.bullish_threshold)).clip(0, 1)
            signals.loc[long_mask, "position"] *= long_scaling[long_mask]
            
            # For shorts: scale by how much confidence is below bearish threshold
            short_scaling = ((self.config.bearish_threshold - confidence) / self.config.bearish_threshold).clip(0, 1)
            signals.loc[short_mask, "position"] *= short_scaling[short_mask]
            
        return signals

    def _generate_verification_signals(self, data: pd.DataFrame, confidence: pd.Series, signals: pd.DataFrame) -> pd.DataFrame:
        """Generate verification multipliers for other strategies"""
        
        # In verification mode, output confidence as position multiplier
        # Other strategies will multiply their signals by this
        signals["position"] = confidence
        
        # Filter out low confidence signals
        low_confidence = confidence < self.config.min_pattern_confidence
        signals.loc[low_confidence, "position"] = 0.0
        
        self.verification_stats['signals_filtered'] = low_confidence.sum()
        
        return signals

    def _generate_enhancement_signals(self, data: pd.DataFrame, confidence: pd.Series, signals: pd.DataFrame) -> pd.DataFrame:
        """Generate enhancement multipliers with directional bias"""
        
        # Enhancement mode: provide both direction and confidence
        signals["position"] = 0.0
        
        # Convert confidence to directional signal
        # > 0.5 = bullish, < 0.5 = bearish
        directional_signal = (confidence - 0.5) * 2  # Scale to -1 to +1 range
        
        # Only keep signals above minimum confidence
        strong_signals = np.abs(directional_signal) > (self.config.min_pattern_confidence - 0.5) * 2
        
        signals.loc[strong_signals, "position"] = directional_signal[strong_signals]
        
        self.verification_stats['signals_enhanced'] = strong_signals.sum()
        
        return signals

    def calculate_signal_strength(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """Calculate signal strength based on pattern confidence"""
        
        # In all modes, signal strength is based on how confident the pattern is
        if self.config.mode == "standalone":
            # For standalone, strength is distance from neutral (0.5)
            strength = np.abs(signals["position"]).clip(0, 1)
        else:
            # For verification/enhancement, strength is the confidence itself
            strength = signals["position"].abs().clip(0, 1)
            
        return strength.fillna(0)

    def apply_pattern_verification(self, other_signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Apply pattern verification to signals from other strategies"""
        
        if self.model is None:
            print("⚠️  No LSTM model available for pattern verification")
            return other_signals
            
        # Get pattern confidence
        pattern_confidence = self._calculate_pattern_confidence(data)
        
        if pattern_confidence is None:
            print("⚠️  Failed to calculate pattern confidence")
            return other_signals
            
        verified_signals = other_signals.copy()
        original_positions = verified_signals["position"].copy()
        
        # Apply enhanced signal integration
        if self.config.mode == "verification":
            print(f"   🔧 Applying enhanced pattern verification (enhanced={self.config.use_enhanced_integration})")
            
            # Apply pattern enhancement to each signal
            enhanced_positions = pd.Series(index=verified_signals.index, dtype=float)
            
            for idx in verified_signals.index:
                if idx in pattern_confidence.index and not pd.isna(pattern_confidence[idx]):
                    original_signal = original_positions[idx]
                    confidence = pattern_confidence[idx]
                    
                    # Apply enhanced integration logic
                    enhanced_signal = self._apply_pattern_enhancement(original_signal, confidence)
                    
                    # Filter out signals with insufficient confidence
                    if confidence < self.config.min_pattern_confidence:
                        enhanced_signal = 0.0
                        
                    enhanced_positions[idx] = enhanced_signal
                else:
                    # No pattern confidence available, keep original signal
                    enhanced_positions[idx] = original_positions[idx]
            
            verified_signals["position"] = enhanced_positions
            
            # Handle signal_strength column if it exists
            if "signal_strength" in verified_signals.columns:
                # Update signal strength proportionally
                strength_multiplier = enhanced_positions.abs() / (original_positions.abs() + 1e-8)
                verified_signals["signal_strength"] *= strength_multiplier.fillna(1.0)
            elif "confidence" in verified_signals.columns:
                # Blend original confidence with pattern confidence
                verified_signals["confidence"] = (verified_signals["confidence"] * 0.7 + 
                                                pattern_confidence * 0.3).fillna(verified_signals["confidence"])
            
            # Log integration statistics for monitoring
            self._log_integration_statistics(original_positions, enhanced_positions, pattern_confidence)
            
        return verified_signals

    def get_verification_stats(self) -> Dict[str, Any]:
        """Get pattern verification performance statistics"""
        return {
            **self.verification_stats,
            'mode': self.config.mode,
            'min_confidence': self.config.min_pattern_confidence,
            'model_loaded': self.model is not None
        }

    def get_strategy_description(self) -> str:
        """Get human-readable strategy description"""
        return (
            f"LSTM Pattern {self.config.mode.title()} Strategy: "
            f"Min confidence={self.config.min_pattern_confidence:.2f}, "
            f"Bullish>{self.config.bullish_threshold:.2f}, "
            f"Bearish<{self.config.bearish_threshold:.2f}"
        )