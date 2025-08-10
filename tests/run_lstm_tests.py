#!/usr/bin/env python3

"""
LSTM Pattern Detection Test Suite
Runs focused tests to validate LSTM pipeline components work end-to-end

This orchestrates the individual test modules to provide a comprehensive validation
of the LSTM pattern detection system before ensemble integration.
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_module(test_module_path: Path, test_name: str) -> bool:
    """Run a test module and return success status"""
    
    print(f"\n{'='*60}")
    print(f"RUNNING: {test_name}")
    print(f"{'='*60}")
    
    try:
        # Run the test as a subprocess to isolate any import issues
        result = subprocess.run([
            sys.executable, str(test_module_path)
        ], capture_output=True, text=True, cwd=project_root)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        success = result.returncode == 0
        
        if success:
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED (exit code: {result.returncode})")
        
        return success
        
    except Exception as e:
        print(f"✗ {test_name} ERROR: {e}")
        return False


def main():
    """Run the complete LSTM test suite"""
    
    print("LSTM PATTERN DETECTION TEST SUITE")
    print("=" * 60)
    print("Testing components in isolation before integration")
    print("Expected runtime: 2-5 minutes")
    print()
    
    tests_dir = Path(__file__).parent
    test_results = {}
    
    # Test sequence: basic to complex
    test_sequence = [
        (tests_dir / "data_loader.py", "Data Loading Utilities"),
        (tests_dir / "test_trainer_initialization.py", "LSTM Trainer Initialization"), 
        (tests_dir / "test_data_pipeline.py", "Data Pipeline Components"),
    ]
    
    # Run each test
    overall_success = True
    for test_path, test_name in test_sequence:
        if not test_path.exists():
            print(f"✗ Test file not found: {test_path}")
            test_results[test_name] = False
            overall_success = False
            continue
        
        success = run_test_module(test_path, test_name)
        test_results[test_name] = success
        
        if not success:
            overall_success = False
    
    # Final report
    print(f"\n{'='*60}")
    print("LSTM TEST SUITE FINAL REPORT")
    print(f"{'='*60}")
    
    for test_name, success in test_results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n{'='*60}")
    if overall_success:
        print("🎉 ALL LSTM TESTS PASSED - READY FOR FULL PIPELINE TESTING")
        print("\nNext Steps:")
        print("  1. STEP 1.2: Run full LSTM training on MAG7 stocks")
        print("  2. STEP 1.3: Validate LSTM baseline performance") 
    else:
        print("❌ SOME TESTS FAILED - FIX ISSUES BEFORE PROCEEDING")
        print("\nRecommendation:")
        print("  Fix failing components before attempting full training")
    print(f"{'='*60}")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)