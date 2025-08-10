#!/usr/bin/env python3

"""
Leakage Detector
Advanced data leakage detection and prevention utilities
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))


class LeakageDetector:
    """
    Advanced data leakage detection and prevention framework
    
    Provides comprehensive tools for detecting various forms of data leakage
    in financial machine learning pipelines.
    """
    
    def __init__(self, correlation_threshold: float = 0.10, early_epoch_threshold: float = 0.15):
        """
        Initialize leakage detector
        
        Args:
            correlation_threshold: Threshold for detecting suspicious correlations
            early_epoch_threshold: Threshold for early epoch leakage detection
        """
        self.correlation_threshold = correlation_threshold
        self.early_epoch_threshold = early_epoch_threshold
        self.detection_history = []
    
    def detect_feature_target_leakage(self, features: pd.DataFrame, targets: np.ndarray, 
                                    feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect leakage between input features and targets
        
        Args:
            features: Input features DataFrame
            targets: Target values array
            feature_names: Optional list of feature names to check
            
        Returns:
            Dictionary with leakage detection results
        """
        print(f"🔍 DETECTING FEATURE-TARGET LEAKAGE:")
        
        if feature_names is None:
            feature_names = features.columns.tolist()
        
        leakage_results = {}
        suspicious_features = []
        high_correlations = []
        
        for feature_name in feature_names:
            if feature_name in features.columns:
                feature_values = features[feature_name].values
                
                # Handle NaN values
                valid_mask = ~(np.isnan(feature_values) | np.isnan(targets))
                if np.sum(valid_mask) < 10:  # Need sufficient data points
                    continue
                
                valid_features = feature_values[valid_mask]
                valid_targets = targets[valid_mask]
                
                # Calculate correlation
                if np.var(valid_features) > 1e-10 and np.var(valid_targets) > 1e-10:
                    correlation = np.corrcoef(valid_features, valid_targets)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0
                
                abs_correlation = abs(correlation)
                
                # Classify correlation level
                if abs_correlation > self.early_epoch_threshold:
                    severity = "CRITICAL"
                    suspicious_features.append(feature_name)
                elif abs_correlation > self.correlation_threshold:
                    severity = "WARNING"
                    suspicious_features.append(feature_name)
                else:
                    severity = "NORMAL"
                
                leakage_results[feature_name] = {
                    "correlation": float(correlation),
                    "abs_correlation": float(abs_correlation),
                    "severity": severity,
                    "valid_samples": int(np.sum(valid_mask)),
                    "suspicious": abs_correlation > self.correlation_threshold
                }
                
                if abs_correlation > self.correlation_threshold:
                    high_correlations.append((feature_name, abs_correlation, severity))
                    print(f"⚠️  {feature_name}: {abs_correlation:.3f} correlation ({severity})")
        
        # Sort by correlation strength
        high_correlations.sort(key=lambda x: x[1], reverse=True)
        
        summary = {
            "total_features_checked": len(feature_names),
            "suspicious_features": len(suspicious_features),
            "critical_features": len([f for f in leakage_results.values() if f["severity"] == "CRITICAL"]),
            "warning_features": len([f for f in leakage_results.values() if f["severity"] == "WARNING"]),
            "max_correlation": max([r["abs_correlation"] for r in leakage_results.values()], default=0.0),
            "mean_correlation": np.mean([r["abs_correlation"] for r in leakage_results.values()]),
            "leakage_detected": len(suspicious_features) > 0
        }
        
        result = {
            "feature_results": leakage_results,
            "suspicious_features": suspicious_features,
            "high_correlations": high_correlations,
            "summary": summary,
            "detection_timestamp": pd.Timestamp.now()
        }
        
        self.detection_history.append(result)
        
        print(f"📊 LEAKAGE DETECTION SUMMARY:")
        print(f"- Features checked: {summary['total_features_checked']}")
        print(f"- Suspicious features: {summary['suspicious_features']}")
        print(f"- Max correlation: {summary['max_correlation']:.3f}")
        print(f"- Overall status: {'🚨 LEAKAGE DETECTED' if summary['leakage_detected'] else '✅ CLEAN'}")
        
        return result
    
    def detect_temporal_leakage(self, sequences: np.ndarray, targets: np.ndarray, 
                              sequence_length: int, stride: int) -> Dict[str, Any]:
        """
        Detect temporal leakage in sequence generation
        
        Args:
            sequences: Input sequences array [n_samples, sequence_length, n_features]
            targets: Target values array
            sequence_length: Length of each sequence
            stride: Stride between sequences
            
        Returns:
            Dictionary with temporal leakage analysis
        """
        print(f"🔍 DETECTING TEMPORAL LEAKAGE:")
        print(f"- Sequences: {sequences.shape}")
        print(f"- Sequence length: {sequence_length}")
        print(f"- Stride: {stride}")
        
        # Calculate overlap percentage
        overlap_pct = (sequence_length - stride) / sequence_length * 100 if sequence_length > 0 else 0
        
        # Analyze sequence overlap patterns
        n_samples = len(sequences)
        overlap_analysis = {
            "total_sequences": n_samples,
            "sequence_length": sequence_length,
            "stride": stride,
            "overlap_percentage": float(overlap_pct),
            "data_reuse_factor": sequence_length / stride if stride > 0 else float('inf')
        }
        
        # Test correlations between overlapping sequences
        overlap_correlations = []
        if n_samples > stride and len(sequences) > 1:
            # Compare sequences that should have overlap
            for i in range(min(10, n_samples - stride)):  # Test first 10 overlapping pairs
                seq1 = sequences[i].flatten()
                seq2 = sequences[i + stride].flatten()
                
                if np.var(seq1) > 1e-10 and np.var(seq2) > 1e-10:
                    overlap_corr = np.corrcoef(seq1, seq2)[0, 1]
                    if not np.isnan(overlap_corr):
                        overlap_correlations.append(abs(overlap_corr))
        
        # Analyze target autocorrelation (another leakage indicator)
        target_autocorr = []
        if len(targets) > stride:
            for lag in [1, stride, stride*2]:
                if len(targets) > lag:
                    target_current = targets[:-lag]
                    target_lagged = targets[lag:]
                    
                    if np.var(target_current) > 1e-10 and np.var(target_lagged) > 1e-10:
                        autocorr = np.corrcoef(target_current, target_lagged)[0, 1]
                        if not np.isnan(autocorr):
                            target_autocorr.append({
                                "lag": lag,
                                "autocorr": float(abs(autocorr))
                            })
        
        # Assess leakage risk
        high_overlap_risk = overlap_pct > 90
        high_sequence_correlation = np.mean(overlap_correlations) > 0.8 if overlap_correlations else False
        high_target_autocorr = any(ta["autocorr"] > 0.3 for ta in target_autocorr)
        
        leakage_risk = "HIGH" if (high_overlap_risk and (high_sequence_correlation or high_target_autocorr)) else \
                      "MEDIUM" if high_overlap_risk else "LOW"
        
        result = {
            "overlap_analysis": overlap_analysis,
            "sequence_correlations": overlap_correlations,
            "mean_sequence_correlation": float(np.mean(overlap_correlations)) if overlap_correlations else 0.0,
            "target_autocorrelation": target_autocorr,
            "leakage_risk": leakage_risk,
            "recommendations": self._generate_temporal_recommendations(overlap_pct, leakage_risk),
            "detection_timestamp": pd.Timestamp.now()
        }
        
        print(f"📊 TEMPORAL LEAKAGE ANALYSIS:")
        print(f"- Sequence overlap: {overlap_pct:.1f}%")
        print(f"- Leakage risk: {leakage_risk}")
        print(f"- Mean sequence correlation: {result['mean_sequence_correlation']:.3f}")
        
        return result
    
    def detect_early_epoch_leakage(self, correlation_history: List[Dict[str, Any]], 
                                 max_epochs: int = 5) -> Dict[str, Any]:
        """
        Detect data leakage based on early epoch performance
        
        Args:
            correlation_history: List of correlation results by epoch
            max_epochs: Maximum epochs to consider for early detection
            
        Returns:
            Dictionary with early epoch leakage analysis
        """
        print(f"🔍 DETECTING EARLY EPOCH LEAKAGE:")
        
        if not correlation_history:
            return {"error": "No correlation history provided"}
        
        early_epochs = correlation_history[:max_epochs]
        alerts = []
        
        for epoch_data in early_epochs:
            epoch = epoch_data.get("epoch", 0)
            val_corr = abs(epoch_data.get("val_corr", 0))
            
            if val_corr > self.early_epoch_threshold:
                alert = {
                    "epoch": epoch,
                    "validation_correlation": val_corr,
                    "threshold": self.early_epoch_threshold,
                    "severity": "CRITICAL" if val_corr > 0.25 else "WARNING"
                }
                alerts.append(alert)
                print(f"🚨 Epoch {epoch}: {val_corr:.3f} correlation > {self.early_epoch_threshold:.3f} threshold")
        
        # Analyze learning curve slope (steep early learning suggests leakage)
        if len(early_epochs) >= 3:
            correlations = [abs(e.get("val_corr", 0)) for e in early_epochs]
            learning_slope = (correlations[-1] - correlations[0]) / len(correlations)
            steep_learning = learning_slope > 0.05  # >5% per epoch is suspicious
        else:
            learning_slope = 0.0
            steep_learning = False
        
        result = {
            "alerts": alerts,
            "alert_count": len(alerts),
            "max_early_correlation": max([abs(e.get("val_corr", 0)) for e in early_epochs], default=0.0),
            "learning_slope": float(learning_slope),
            "steep_learning_detected": steep_learning,
            "leakage_detected": len(alerts) > 0 or steep_learning,
            "epochs_analyzed": len(early_epochs),
            "detection_timestamp": pd.Timestamp.now()
        }
        
        print(f"📊 EARLY EPOCH ANALYSIS:")
        print(f"- Epochs analyzed: {result['epochs_analyzed']}")
        print(f"- Leakage alerts: {result['alert_count']}")
        print(f"- Max early correlation: {result['max_early_correlation']:.3f}")
        print(f"- Learning slope: {result['learning_slope']:.3f}")
        print(f"- Status: {'🚨 LEAKAGE DETECTED' if result['leakage_detected'] else '✅ CLEAN'}")
        
        return result
    
    def _generate_temporal_recommendations(self, overlap_pct: float, risk_level: str) -> List[str]:
        """
        Generate recommendations for reducing temporal leakage
        
        Args:
            overlap_pct: Sequence overlap percentage
            risk_level: Assessed risk level
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.append("🚨 CRITICAL: Reduce sequence overlap immediately")
            if overlap_pct > 95:
                recommendations.append("- Increase stride to at least sequence_length/2")
            recommendations.append("- Use separate validation data with zero overlap")
            recommendations.append("- Consider walk-forward validation")
        
        elif risk_level == "MEDIUM":
            recommendations.append("⚠️  Consider reducing sequence overlap")
            recommendations.append("- Current overlap may cause overfitting")
            recommendations.append("- Monitor validation performance closely")
        
        else:
            recommendations.append("✅ Temporal configuration appears acceptable")
        
        if overlap_pct > 90:
            optimal_stride = max(1, int(overlap_pct / 75))  # Target ~75% overlap
            recommendations.append(f"- Recommended stride: {optimal_stride} (would reduce overlap to ~75%)")
        
        return recommendations
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive leakage detection report
        
        Returns:
            Formatted report string
        """
        if not self.detection_history:
            return "No leakage detection history available"
        
        latest = self.detection_history[-1]
        
        report = f"""
Data Leakage Detection Report
{'='*50}

Detection Summary:
- Timestamp: {latest.get('detection_timestamp', 'Unknown')}
- Features analyzed: {latest['summary']['total_features_checked']}
- Suspicious features: {latest['summary']['suspicious_features']}
- Critical features: {latest['summary']['critical_features']}
- Warning features: {latest['summary']['warning_features']}

Correlation Analysis:
- Maximum correlation: {latest['summary']['max_correlation']:.6f}
- Mean correlation: {latest['summary']['mean_correlation']:.6f}
- Threshold: {self.correlation_threshold:.3f}

High-Risk Features:
"""
        
        high_corr = latest.get("high_correlations", [])[:10]  # Top 10
        for feature, correlation, severity in high_corr:
            report += f"  - {feature}: {correlation:.4f} ({severity})\n"
        
        if not high_corr:
            report += "  None detected ✅\n"
        
        # Add recommendations
        report += f"\nRecommendations:\n"
        if latest['summary']['leakage_detected']:
            report += "  🚨 Data leakage detected - immediate action required:\n"
            report += "    1. Review feature engineering pipeline\n"
            report += "    2. Verify temporal gaps in target generation\n"
            report += "    3. Check for forward-looking calculations\n"
            report += "    4. Validate train/test split methodology\n"
        else:
            report += "  ✅ No significant leakage detected\n"
            report += "  - Continue monitoring during training\n"
            report += "  - Validate with out-of-sample testing\n"
        
        return report
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """
        Get summary of all detection results
        
        Returns:
            Summary dictionary
        """
        if not self.detection_history:
            return {"status": "no_history"}
        
        latest = self.detection_history[-1]
        
        return {
            "latest_detection": latest["detection_timestamp"],
            "total_detections": len(self.detection_history),
            "current_status": "LEAKAGE_DETECTED" if latest['summary']['leakage_detected'] else "CLEAN",
            "max_correlation": latest['summary']['max_correlation'],
            "suspicious_feature_count": latest['summary']['suspicious_features'],
            "critical_feature_count": latest['summary']['critical_features'],
            "thresholds": {
                "correlation_threshold": self.correlation_threshold,
                "early_epoch_threshold": self.early_epoch_threshold
            }
        }