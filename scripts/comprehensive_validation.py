#!/usr/bin/env python3

"""
Comprehensive validation runner - combines all validation approaches
to thoroughly test model predictive power and architecture
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from progressive_model_validation import ProgressiveModelValidator
from architecture_validation import ArchitectureValidator
from src.utils.logging_utils import setup_production_logger


class ComprehensiveValidator:
    """Master validator that runs all validation approaches"""
    
    def __init__(self):
        self.logger = setup_production_logger({"log_file": "comprehensive_validation.log"})
        self.start_time = time.time()
    
    def run_all_validations(self):
        """Run complete validation suite"""
        
        self.logger.log("=" * 80)
        self.logger.log("COMPREHENSIVE MODEL VALIDATION SUITE")
        self.logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.log("=" * 80)
        
        validation_results = {}
        
        print("🚀 Starting Comprehensive Model Validation...")
        print("This will test model architecture and predictive power systematically.\n")
        
        # Phase 1: Architecture Validation
        print("📐 Phase 1: Architecture Validation")
        print("Testing different LSTM architectures with synthetic data...")
        
        try:
            arch_validator = ArchitectureValidator()
            arch_results = arch_validator.run_architecture_validation()
            validation_results['architecture'] = arch_results
            
            # Quick assessment
            if arch_results:
                working_archs = sum(1 for r in arch_results.values() 
                                  if 'error' not in r and r.get('converged', False))
                total_archs = len([r for r in arch_results.values() if 'error' not in r])
                
                if working_archs >= total_archs * 0.7:
                    print("   ✅ Architecture validation PASSED")
                    print(f"   ✅ {working_archs}/{total_archs} architectures working correctly")
                else:
                    print("   ⚠️  Architecture validation PARTIAL")
                    print(f"   ⚠️  Only {working_archs}/{total_archs} architectures working")
            else:
                print("   ❌ Architecture validation FAILED")
                
        except Exception as e:
            print(f"   ❌ Architecture validation ERROR: {e}")
            validation_results['architecture'] = {"error": str(e)}
        
        print()
        
        # Phase 2: Progressive Feature Validation
        print("🎯 Phase 2: Progressive Feature Validation")
        print("Testing predictive power with increasing feature complexity...")
        
        try:
            prog_validator = ProgressiveModelValidator({
                "tickers": ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"],  # 5 tickers for robust testing
                "years_of_data": 10,  # 10 years for sufficient data
                "prediction_horizon": "daily",
                "lookback_window": 30,  # Longer lookback
                "target_features": 12,  # Optimal feature count
                "database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db"
            })
            
            prog_results = prog_validator.run_progressive_validation()
            validation_results['progressive'] = prog_results
            
            # Quick assessment
            if prog_results:
                successful_levels = sum(1 for r in prog_results.values() if 'error' not in r)
                total_levels = len(prog_results)
                
                # Check if LSTM performs better than baseline in comprehensive selected features
                comp_results = prog_results.get('Comprehensive_Selected', {})
                if 'lstm_model' in comp_results and comp_results['lstm_model']:
                    lstm_acc = comp_results['lstm_model']['test_directional_accuracy']
                    baseline_accs = [m['directional_accuracy'] for m in comp_results.get('baseline_models', {}).values()]
                    best_baseline = max(baseline_accs) if baseline_accs else 0
                    
                    if lstm_acc > best_baseline + 2:  # 2% better than best baseline
                        print("   ✅ Progressive validation PASSED")
                        print(f"   ✅ LSTM ({lstm_acc:.1f}%) outperforms best baseline ({best_baseline:.1f}%)")
                    elif lstm_acc > 50:  # At least better than random
                        print("   ⚠️  Progressive validation PARTIAL")
                        print(f"   ⚠️  LSTM accuracy {lstm_acc:.1f}% (better than random)")
                    else:
                        print("   ❌ Progressive validation FAILED")
                        print(f"   ❌ LSTM accuracy {lstm_acc:.1f}% (worse than random)")
                else:
                    print("   ❌ Progressive validation FAILED - LSTM training failed")
            else:
                print("   ❌ Progressive validation FAILED")
                
        except Exception as e:
            print(f"   ❌ Progressive validation ERROR: {e}")
            validation_results['progressive'] = {"error": str(e)}
        
        print()
        
        # Phase 3: Advanced Feature Selection Validation
        print("🧬 Phase 3: Advanced Feature Selection Validation")
        print("Testing comprehensive feature selection pipeline with advanced methods...")
        
        try:
            # Test the advanced feature selection pipeline
            from src.config.config import Config
            from src.data.data_loader import DataLoader
            from src.features.feature_selector import FinancialFeatureSelector
            from src.features.feature_engineer import FeatureEngineer
            
            # Quick validation of advanced feature selection
            config = Config({"database_url": "sqlite:////Users/beneisner/financial-returns-api/returns.db"})
            data_loader = DataLoader(config)
            
            # Load test data (smaller subset for speed)
            test_ticker = "AAPL"
            test_data = data_loader.load_single_ticker_data(test_ticker, 3)  # 3 years for speed
            
            print(f"   Testing on {len(test_data)} records from {test_ticker}")
            
            # Create comprehensive features using processor-based system
            engineer = FeatureEngineer()
            comprehensive_features = engineer.create_comprehensive_features(test_data)
            
            # Prepare for advanced selection
            target = comprehensive_features['daily_return'].fillna(0).values
            feature_data = comprehensive_features.drop(columns=['daily_return'])
            
            # Run advanced feature selection (only on numerical features)
            numerical_features = feature_data.select_dtypes(include=[np.number])
            print(f"   Filtering to {len(numerical_features.columns)} numerical features")
            
            advanced_selector = FinancialFeatureSelector(target_features=10, min_variance_threshold=0.001)
            selected_features = advanced_selector.fit_transform(numerical_features, target)
            
            # Analyze results
            selection_results = {
                'total_candidates': len(feature_data.columns),
                'final_selected': len(selected_features.columns),
                'reduction_ratio': len(selected_features.columns) / len(feature_data.columns),
                'methods_tested': list(advanced_selector.feature_scores_.keys()) if hasattr(advanced_selector, 'feature_scores_') else []
            }
            
            validation_results['advanced_feature_selection'] = selection_results
            
            # Assessment
            if len(selection_results['methods_tested']) >= 4:  # Should test at least 4 advanced methods
                print(f"   ✅ Advanced feature selection PASSED")
                print(f"   ✅ {selection_results['total_candidates']} → {selection_results['final_selected']} features")
                print(f"   ✅ Methods applied: {', '.join(selection_results['methods_tested'][:4])}...")
            else:
                print(f"   ⚠️  Advanced feature selection PARTIAL")
                print(f"   ⚠️  Only {len(selection_results['methods_tested'])} advanced methods applied")
                
        except Exception as e:
            print(f"   ❌ Advanced feature selection ERROR: {e}")
            validation_results['advanced_feature_selection'] = {"error": str(e)}
        
        print()
        
        # Generate final assessment
        self.generate_final_assessment(validation_results)
        
        total_time = time.time() - self.start_time
        self.logger.log(f"\nTotal validation time: {total_time/60:.1f} minutes")
        
        return validation_results
    
    def generate_final_assessment(self, results):
        """Generate comprehensive final assessment"""
        
        self.logger.log("\n" + "=" * 80)
        self.logger.log("COMPREHENSIVE VALIDATION FINAL ASSESSMENT")
        self.logger.log("=" * 80)
        
        print("🏁 FINAL VALIDATION ASSESSMENT")
        print("=" * 60)
        
        # Architecture Assessment
        arch_score = 0
        if 'architecture' in results and 'error' not in results['architecture']:
            working_archs = sum(1 for r in results['architecture'].values() 
                              if 'error' not in r and r.get('converged', False))
            total_archs = len([r for r in results['architecture'].values() if 'error' not in r])
            arch_score = working_archs / total_archs if total_archs > 0 else 0
        
        # Progressive Assessment
        prog_score = 0
        lstm_better_than_baseline = False
        
        if 'progressive' in results and 'error' not in results['progressive']:
            # Check if comprehensive selected LSTM beats baseline
            comp_results = results['progressive'].get('Comprehensive_Selected', {})
            if 'lstm_model' in comp_results and comp_results['lstm_model']:
                lstm_acc = comp_results['lstm_model']['test_directional_accuracy']
                baseline_accs = [m['directional_accuracy'] for m in comp_results.get('baseline_models', {}).values()]
                best_baseline = max(baseline_accs) if baseline_accs else 50
                
                if lstm_acc > best_baseline + 2:
                    prog_score = 1.0
                    lstm_better_than_baseline = True
                elif lstm_acc > 50:
                    prog_score = 0.7
                else:
                    prog_score = 0.3
        
        # Advanced Feature Selection Assessment
        feature_score = 0
        if 'advanced_feature_selection' in results and 'error' not in results['advanced_feature_selection']:
            feature_results = results['advanced_feature_selection']
            methods_tested = len(feature_results.get('methods_tested', []))
            
            if methods_tested >= 6:  # All advanced methods working
                feature_score = 1.0
            elif methods_tested >= 4:  # Most methods working
                feature_score = 0.8
            elif methods_tested >= 2:  # Some methods working
                feature_score = 0.5
            else:
                feature_score = 0.2
        
        # Overall Assessment (now includes 3 components)
        overall_score = (arch_score + prog_score + feature_score) / 3
        
        print(f"Architecture Validation: {arch_score*100:.0f}%")
        print(f"Progressive Validation: {prog_score*100:.0f}%")
        print(f"Advanced Feature Selection: {feature_score*100:.0f}%")
        print(f"Overall Score: {overall_score*100:.0f}%")
        print()
        
        # Recommendations
        print("📋 RECOMMENDATIONS:")
        
        if overall_score >= 0.8:
            print("✅ EXCELLENT - Model is ready for production training")
            print("   • Architecture is working correctly across complexity levels")
            print("   • LSTM shows superior predictive power vs baselines")
            print("   • Advanced feature selection pipeline fully operational")
            print("   • Proceed with full feature engineering and extended training")
            
        elif overall_score >= 0.6:
            print("⚠️  GOOD - Model shows promise with some issues")
            print("   • Most components working correctly")
            if not lstm_better_than_baseline:
                print("   • Consider feature engineering improvements")
            if feature_score < 0.8:
                print("   • Some advanced feature selection methods may need attention")
            print("   • Proceed with caution, monitor training closely")
            
        elif overall_score >= 0.4:
            print("⚠️  MIXED - Significant issues detected")
            print("   • Some architectures or features not working optimally")
            if feature_score < 0.5:
                print("   • Advanced feature selection pipeline needs debugging")
            print("   • Review model building and feature engineering")
            print("   • Consider simpler approaches initially")
            
        else:
            print("❌ POOR - Major issues require attention")
            print("   • Architecture or training pipeline problems")
            if feature_score < 0.3:
                print("   • Advanced feature selection pipeline failing")
            print("   • Systematic debugging required")
            print("   • Do not proceed with production training")
        
        print()
        
        # Next Steps
        print("🎯 NEXT STEPS:")
        
        if overall_score >= 0.7:
            print("1. Run full production training with complete feature set")
            print("2. Use 10-15 years of historical data")
            print("3. Enable advanced features and longer training")
            print("4. Implement model monitoring and evaluation")
            
        elif overall_score >= 0.5:
            print("1. Fix identified issues before production training")
            print("2. Start with simpler feature sets")
            print("3. Gradually increase complexity")
            print("4. Monitor training stability closely")
            
        else:
            print("1. Debug architecture and training pipeline")
            print("2. Verify data quality and loading")
            print("3. Test with synthetic data first")
            print("4. Consider consulting model architecture")
        
        print("=" * 60)
        
        # Log detailed results
        self.logger.log(f"Architecture Score: {arch_score:.2f}")
        self.logger.log(f"Progressive Score: {prog_score:.2f}")
        self.logger.log(f"Advanced Feature Selection Score: {feature_score:.2f}")
        self.logger.log(f"Overall Assessment: {overall_score:.2f}")
        self.logger.log(f"LSTM Better Than Baseline: {lstm_better_than_baseline}")
        
        # Log advanced feature selection details
        if 'advanced_feature_selection' in results:
            fs_results = results['advanced_feature_selection']
            self.logger.log(f"Feature Selection Methods Tested: {fs_results.get('methods_tested', 'None')}")
            self.logger.log(f"Feature Reduction: {fs_results.get('total_candidates', 'N/A')} → {fs_results.get('final_selected', 'N/A')}")


def main():
    """Run comprehensive validation"""
    
    print("🔬 COMPREHENSIVE MODEL VALIDATION SUITE")
    print("=" * 50)
    print("This will systematically test:")
    print("• Model architecture correctness")
    print("• Feature engineering effectiveness") 
    print("• Advanced feature selection pipeline (NEW)")
    print("• Predictive power vs baselines")
    print("• Training stability and convergence")
    print()
    
    validator = ComprehensiveValidator()
    results = validator.run_all_validations()
    
    print(f"\n📊 Validation complete! Check logs for detailed results:")
    print("   • comprehensive_validation.log - Master validation results")
    print("   • architecture_validation.log - Architecture testing details") 
    print("   • progressive_validation.log - Progressive feature testing")
    print("   • Note: Advanced feature selection tested within comprehensive validation")
    
    return results


if __name__ == "__main__":
    results = main()