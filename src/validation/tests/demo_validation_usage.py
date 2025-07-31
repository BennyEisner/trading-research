#!/usr/bin/env python3

"""
Demonstration script showing how to use the validation framework
Shows correct usage patterns and best practices
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add the src directory to the path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from validation.robust_time_series_validator import create_robust_validator
from validation.gapped_time_series_cv import create_gapped_cv
from validation.enhanced_robustness_tests import create_robustness_tester
from validation.pipeline_validator import create_pipeline_validator
from validation.tests.fixtures.test_data_generators import TestDataGenerator, ValidationTestFixtures


def demo_basic_model_validation():
    """Demonstrate basic model validation workflow"""
    print("\n" + "="*60)
    print("DEMO: Basic Model Validation Workflow")
    print("="*60)
    
    # Generate synthetic data
    test_data = TestDataGenerator()
    actual_returns, predictions = test_data.create_biased_predictions(n_samples=500, bias=0.65)
    
    print(f"Generated {len(actual_returns)} samples for validation")
    print(f"Actual returns: μ={np.mean(actual_returns):.4f}, σ={np.std(actual_returns):.4f}")
    print(f"Predictions: μ={np.mean(predictions):.4f}, σ={np.std(predictions):.4f}")
    
    # Step 1: Robust time-series validation
    print("\n🔬 Step 1: Robust Time-Series Validation")
    robust_validator = create_robust_validator(
        block_size=5,          # 5-day blocks for swing trading
        n_bootstrap=1000,      # 1000 bootstrap samples
        random_state=42        # For reproducible results
    )
    
    results = robust_validator.validate_model_significance(
        actual_returns, predictions, 
        test_types=['bootstrap', 'permutation']
    )
    
    print(f"Bootstrap p-value: {results['moving_block_bootstrap']['p_value']:.4f}")
    print(f"Permutation p-value: {results['conditional_permutation']['p_value']:.4f}")
    print(f"Overall significant: {results['overall_assessment']['all_significant']}")
    print(f"Recommendation: {results['overall_assessment']['recommendation']}")
    
    return results['overall_assessment']['all_significant']


def demo_cross_validation_testing():
    """Demonstrate cross-validation workflow"""
    print("\n" + "="*60)
    print("DEMO: Cross-Validation Testing")
    print("="*60)
    
    # Generate sequence data
    test_data = TestDataGenerator()
    X, y = test_data.create_sequences_3d(n_samples=400, sequence_length=30, n_features=8)
    dates = pd.date_range(start='2020-01-01', periods=len(X), freq='D')
    
    print(f"Generated sequences: {X.shape}")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    
    # Set up cross-validation
    print("\n🔄 Setting up Gapped Time-Series Cross-Validation")
    cv = create_gapped_cv(
        n_splits=3,           # 3 validation splits
        test_size=0.2,        # 20% test size
        gap_size=10,          # 10-day gap to prevent leakage
        expanding_window=True # Expanding training window
    )
    
    # Validate no data leakage
    no_leakage = cv.validate_no_leakage(dates)
    print(f"Data leakage check: {'PASSED' if no_leakage else 'FAILED'}")
    
    if no_leakage:
        # Show split information
        split_info = cv.get_split_dates(dates)
        print(f"\nGenerated {len(split_info)} CV splits:")
        
        for info in split_info:
            print(f"  Split {info['split']}: "
                  f"Train {info['train_start'].date()}-{info['train_end'].date()} "
                  f"({info['train_size']} samples), "
                  f"Test {info['test_start'].date()}-{info['test_end'].date()} "
                  f"({info['test_size']} samples)")
    
    return no_leakage


def demo_robustness_testing():
    """Demonstrate comprehensive robustness testing"""
    print("\n" + "="*60)
    print("🧪 DEMO: Comprehensive Robustness Testing")
    print("="*60)
    
    # Generate test data
    test_data = TestDataGenerator()
    n_samples = 300
    
    # Create autocorrelated returns
    actual_returns = test_data.create_time_series_with_autocorrelation(n_samples, alpha=0.4)
    predictions = actual_returns * 0.6 + np.random.normal(0, 0.01, n_samples)
    
    # Create permuted returns and regime labels
    np.random.seed(42)
    permuted_returns = np.random.permutation(actual_returns)
    regime_labels = test_data.create_regime_labels(n_samples, regime_prob=0.25)
    
    print(f"Testing {n_samples} samples")
    print(f"Regime breakdown: {np.sum(regime_labels == 0)} stable, {np.sum(regime_labels == 1)} transition")
    
    # Run comprehensive robustness tests
    print("\n🔍 Running Comprehensive Robustness Tests")
    robustness_tester = create_robustness_tester(
        significance_level=0.05,
        min_sample_size=30
    )
    
    results = robustness_tester.comprehensive_robustness_test(
        predictions, actual_returns,
        permuted_returns=permuted_returns,
        regime_labels=regime_labels
    )
    
    # Print results
    print(f"\nTest Results:")
    
    if 'autocorrelation_test' in results:
        acf_test = results['autocorrelation_test']
        print(f"  Autocorrelation Test: {acf_test['validation']}")
        print(f"    ACF difference: {acf_test['avg_acf_difference']:.4f}")
    
    if 'regime_transition_test' in results:
        regime_test = results['regime_transition_test']
        print(f"  Regime Transition Test: {regime_test['validation']}")
        print(f"    Stable accuracy: {regime_test['stable_accuracy']:.3f}")
        print(f"    Transition accuracy: {regime_test['transition_accuracy']:.3f}")
        print(f"    Degradation: {regime_test['degradation_pct']:.1f}%")
    
    if 'market_stress_test' in results:
        stress_test = results['market_stress_test']
        print(f"  Market Stress Test: {stress_test['validation']}")
        print(f"    Normal accuracy: {stress_test['normal_accuracy']:.1f}%")
        print(f"    Stress accuracy: {stress_test['stress_accuracy']:.1f}%")
    
    overall = results['overall_assessment']
    print(f"\nOverall Assessment: {overall['recommendation']}")
    print(f"All tests passed: {overall['all_tests_passed']}")
    
    return overall['all_tests_passed']


def demo_pipeline_validation():
    """Demonstrate data pipeline validation"""
    print("\n" + "="*60)
    print("DEMO: Data Pipeline Validation")
    print("="*60)
    
    # Get test fixtures
    fixtures = ValidationTestFixtures()
    clean_data = fixtures.get_clean_pipeline_data()
    problematic_data = fixtures.get_problematic_pipeline_data()
    
    # Create pipeline validator
    pipeline_validator = create_pipeline_validator(
        extreme_price_threshold=0.5,
        correlation_threshold=0.95,
        min_directional_accuracy=40.0
    )
    
    print("🟢 Testing Clean Data Pipeline")
    
    # Test clean OHLCV data
    raw_valid, raw_issues = pipeline_validator.validate_raw_data(
        clean_data['ohlcv_data'], 'CLEAN_TEST'
    )
    print(f"  OHLCV validation: {'PASSED' if raw_valid else 'FAILED'} ({len(raw_issues)} issues)")
    
    # Test clean feature data
    feature_valid, feature_issues = pipeline_validator.validate_feature_data(
        clean_data['feature_data'], clean_data['feature_columns']
    )
    print(f"  Feature validation: {'PASSED' if feature_valid else 'FAILED'} ({len(feature_issues)} issues)")
    
    print("\n🔴 Testing Problematic Data Pipeline")
    
    # Test problematic OHLCV data
    prob_raw_valid, prob_raw_issues = pipeline_validator.validate_raw_data(
        problematic_data['ohlcv_data'], 'PROBLEMATIC_TEST'
    )
    print(f"  OHLCV validation: {'PASSED' if prob_raw_valid else 'FAILED'} ({len(prob_raw_issues)} issues)")
    
    if prob_raw_issues:
        print("    Issues detected:")
        for issue in prob_raw_issues[:3]:  # Show first 3 issues
            print(f"      - {issue}")
    
    # Test problematic feature data
    prob_feature_valid, prob_feature_issues = pipeline_validator.validate_feature_data(
        problematic_data['feature_data'], problematic_data['feature_columns']
    )
    print(f"  Feature validation: {'PASSED' if prob_feature_valid else 'FAILED'} ({len(prob_feature_issues)} issues)")
    
    if prob_feature_issues:
        print("    Issues detected:")
        for issue in prob_feature_issues[:3]:  # Show first 3 issues
            print(f"      - {issue}")
    
    return raw_valid and feature_valid


def demo_complete_validation_workflow():
    """Demonstrate complete end-to-end validation workflow"""
    print("\n" + "="*60)
    print("🏆 DEMO: Complete Validation Workflow")
    print("="*60)
    
    # Get perfect model scenario
    fixtures = ValidationTestFixtures()
    scenario = fixtures.get_perfect_model_scenario()
    
    print("Using perfect model scenario for demonstration")
    print(f"Samples: {len(scenario['actual_returns'])}")
    print(f"Expected result: {scenario['expected_validation_result']}")
    
    validation_results = {}
    
    # Step 1: Pipeline validation
    print("\nStep 1: Pipeline Validation")
    pipeline_validator = create_pipeline_validator()
    
    # Create synthetic OHLCV data
    test_data = TestDataGenerator()
    ohlcv_data = test_data.create_ohlcv_data(n_samples=len(scenario['actual_returns']))
    
    raw_valid, raw_issues = pipeline_validator.validate_raw_data(ohlcv_data, 'WORKFLOW_TEST')
    seq_valid, seq_issues = pipeline_validator.validate_sequences(
        scenario['sequences_X'], scenario['sequences_y']
    )
    pred_valid, pred_issues = pipeline_validator.validate_model_predictions(
        scenario['actual_returns'], scenario['predictions']
    )
    
    validation_results['pipeline'] = raw_valid and seq_valid and pred_valid
    print(f"Pipeline validation: {'PASSED' if validation_results['pipeline'] else 'FAILED'}")
    
    # Step 2: Statistical validation
    print("\nStep 2: Statistical Validation")
    robust_validator = create_robust_validator(n_bootstrap=100, random_state=42)
    
    statistical_results = robust_validator.validate_model_significance(
        scenario['actual_returns'], scenario['predictions']
    )
    
    validation_results['statistical'] = statistical_results['overall_assessment']['all_significant']
    print(f"Statistical validation: {'PASSED' if validation_results['statistical'] else 'FAILED'}")
    print(f"  Min p-value: {statistical_results['overall_assessment']['min_p_value']:.4f}")
    
    # Step 3: Cross-validation structure
    print("\n🔄 Step 3: Cross-Validation Structure")
    dates = pd.date_range(start='2020-01-01', periods=len(scenario['sequences_X']), freq='D')
    cv = create_gapped_cv(n_splits=3, gap_size=5, test_size=0.2)
    
    no_leakage = cv.validate_no_leakage(dates)
    validation_results['cv_structure'] = no_leakage
    print(f"CV structure validation: {'PASSED' if no_leakage else 'FAILED'}")
    
    # Step 4: Robustness testing
    print("\n🧪 Step 4: Robustness Testing")
    robustness_tester = create_robustness_tester()
    
    # Create test data
    permuted_returns = np.random.permutation(scenario['actual_returns'])
    regime_labels = test_data.create_regime_labels(len(scenario['actual_returns']))
    
    robustness_results = robustness_tester.comprehensive_robustness_test(
        scenario['predictions'], scenario['actual_returns'],
        permuted_returns=permuted_returns,
        regime_labels=regime_labels
    )
    
    validation_results['robustness'] = robustness_results['overall_assessment']['all_tests_passed']
    print(f"Robustness testing: {'PASSED' if validation_results['robustness'] else 'NEEDS INVESTIGATION'}")
    
    # Final assessment
    print("\nFinal Assessment:")
    all_passed = all(validation_results.values())
    
    for test_name, passed in validation_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {'ALL VALIDATIONS PASSED' if all_passed else 'SOME VALIDATIONS FAILED'}")
    
    if all_passed:
        print("\nRECOMMENDATION: Model is ready for production deployment")
        print("   - All statistical tests passed")
        print("   - Data quality is excellent")
        print("   - Cross-validation structure is sound")
        print("   - Robustness tests show stable performance")
    else:
        print("\n⚠️  RECOMMENDATION: Address validation failures before deployment")
        print("   - Review failed validation components")
        print("   - Improve model or data quality as needed")
        print("   - Re-run validation after fixes")
    
    return all_passed


def main():
    """Run all validation demos"""
    print("VALIDATION FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print("This demo shows how to properly use the validation framework")
    print("to validate machine learning models for financial predictions.")
    
    demo_results = {}
    
    try:
        # Run individual demos
        demo_results['basic_validation'] = demo_basic_model_validation()
        demo_results['cross_validation'] = demo_cross_validation_testing()
        demo_results['robustness_testing'] = demo_robustness_testing()
        demo_results['pipeline_validation'] = demo_pipeline_validation()
        demo_results['complete_workflow'] = demo_complete_validation_workflow()
        
        # Summary
        print("\n" + "="*80)
        print("DEMO SUMMARY")
        print("="*80)
        
        for demo_name, result in demo_results.items():
            status = "SUCCESS" if result else "ISSUES DETECTED"
            print(f"{demo_name.replace('_', ' ').title():<25}: {status}")
        
        all_successful = all(demo_results.values())
        
        print(f"\nOverall Demo Result: {'ALL DEMOS SUCCESSFUL' if all_successful else 'SOME DEMOS SHOWED ISSUES'}")
        
        print("\nKEY TAKEAWAYS:")
        print("1. Always validate data quality before model evaluation")
        print("2. Use robust statistical tests for significance testing")  
        print("3. Implement proper cross-validation with gap to prevent leakage")
        print("4. Test model robustness across different market conditions")
        print("5. Follow the complete validation workflow for production models")
        
        return all_successful
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(f"\nDemo completed: {'Successfully' if success else 'With issues'}")
    sys.exit(0 if success else 1)