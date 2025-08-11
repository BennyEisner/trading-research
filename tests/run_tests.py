#!/usr/bin/env python3
"""
Consolidated Test Runner
Single entry point for all testing with clear modes and no redundancy
"""

import sys
import argparse
import time
import unittest
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_unit_tests(verbose: bool = False) -> bool:
    """Run all unit tests"""
    print("🧪 Running Unit Tests...")
    
    # Discover and run unit tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests_new/unit', pattern='test_*.py')
    
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests(verbose: bool = False) -> bool:
    """Run integration tests"""
    print("🔗 Running Integration Tests...")
    
    loader = unittest.TestLoader()
    suite = loader.discover('tests_new/integration', pattern='test_*.py')
    
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_comprehensive_test(mode: str = "fast") -> bool:
    """Run comprehensive system test"""
    
    if mode == "fast":
        print("🚀 Running Fast Comprehensive Test...")
        from tests.system.test_comprehensive import run_fast_comprehensive_test
        results = run_fast_comprehensive_test()
    else:
        print("🏁 Running Full Comprehensive Test...")
        from tests.system.test_comprehensive import run_full_comprehensive_test  
        results = run_full_comprehensive_test()
    
    return results.get('overall_success', False)


def run_correlation_validation() -> bool:
    """Run correlation validation test"""
    print("📊 Running Correlation Validation...")
    
    try:
        from test_correlation_final_validation import final_correlation_validation
        final_correlation_validation()
        print("✅ Correlation validation PASSED")
        return True
    except Exception as e:
        print(f"❌ Correlation validation FAILED: {e}")
        return False


def main():
    """Main test runner with various modes"""
    
    parser = argparse.ArgumentParser(description="Consolidated Test Runner")
    parser.add_argument(
        'mode',
        choices=['unit', 'integration', 'comprehensive', 'correlation', 'all', 'fast', 'full'],
        nargs='?',
        default='fast',
        help='Test mode to run'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--comprehensive-mode', choices=['fast', 'full'], default='fast',
                       help='Mode for comprehensive test')
    
    args = parser.parse_args()
    
    start_time = time.time()
    results = []
    
    print(f"{'='*60}")
    print(f"CONSOLIDATED TEST SUITE")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Verbose: {args.verbose}")
    
    try:
        if args.mode == 'unit':
            results.append(('Unit Tests', run_unit_tests(args.verbose)))
            
        elif args.mode == 'integration':
            results.append(('Integration Tests', run_integration_tests(args.verbose)))
            
        elif args.mode == 'comprehensive':
            results.append(('Comprehensive Test', run_comprehensive_test(args.comprehensive_mode)))
            
        elif args.mode == 'correlation':
            results.append(('Correlation Validation', run_correlation_validation()))
            
        elif args.mode == 'fast':
            # Fast development testing
            print("🚀 FAST TEST MODE (for development)")
            results.append(('Unit Tests', run_unit_tests(args.verbose)))
            results.append(('Fast Comprehensive', run_comprehensive_test('fast')))
            
        elif args.mode == 'full':
            # Full test suite  
            print("🏁 FULL TEST MODE (for CI/release)")
            results.append(('Unit Tests', run_unit_tests(args.verbose)))
            results.append(('Integration Tests', run_integration_tests(args.verbose)))
            results.append(('Full Comprehensive', run_comprehensive_test('full')))
            results.append(('Correlation Validation', run_correlation_validation()))
            
        elif args.mode == 'all':
            # All tests including both comprehensive modes
            print("🎯 ALL TESTS MODE")
            results.append(('Unit Tests', run_unit_tests(args.verbose)))
            results.append(('Integration Tests', run_integration_tests(args.verbose)))
            results.append(('Fast Comprehensive', run_comprehensive_test('fast')))
            results.append(('Full Comprehensive', run_comprehensive_test('full')))
            results.append(('Correlation Validation', run_correlation_validation()))
            
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        sys.exit(1)
    
    # Results summary
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Runtime: {elapsed_time:.1f}s")
    print(f"Tests Run: {len(results)}")
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:25}: {status}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    
    overall_success = failed == 0
    if overall_success:
        print("🎉 ALL TESTS PASSED!")
        exit_code = 0
    else:
        print("💥 SOME TESTS FAILED!")
        exit_code = 1
    
    print(f"{'='*60}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()