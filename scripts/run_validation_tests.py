#!/usr/bin/env python3

"""
Comprehensive test runner for validation framework
Runs unit tests, property tests, and integration tests
"""

import argparse
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))


# Test discovery and execution
def discover_tests(test_dir: str, pattern: str = "test_*.py") -> unittest.TestSuite:
    """Discover tests in a directory"""
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    return suite


def run_test_suite(suite: unittest.TestSuite, verbosity: int = 2) -> unittest.TestResult:
    """Run a test suite and return results"""
    runner = unittest.TextTestRunner(
        verbosity=verbosity, stream=sys.stdout, failfast=False, buffer=True  # Capture stdout/stderr
    )
    return runner.run(suite)


def count_tests(suite: unittest.TestSuite) -> int:
    """Count total number of tests in a suite"""
    count = 0
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            count += count_tests(test)
        else:
            count += 1
    return count


class ValidationTestRunner:
    """Main test runner for validation framework"""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.results = {}

        # Test directories
        self.test_dirs = {
            "unit": self.base_dir / "unit",
            "integration": self.base_dir / "integration",
            "property": self.base_dir / "property",
        }

    def run_unit_tests(self, verbosity: int = 2) -> bool:
        """Run unit tests for validation components"""
        print("=" * 60)
        print("RUNNING UNIT TESTS")
        print("=" * 60)

        unit_dir = self.test_dirs["unit"]
        if not unit_dir.exists():
            print(f"Unit test directory not found: {unit_dir}")
            self.results["unit"] = {"status": "SKIPPED", "reason": "Directory not found"}
            return True

        try:
            suite = discover_tests(str(unit_dir))
            test_count = count_tests(suite)

            if test_count == 0:
                print("No unit tests found")
                self.results["unit"] = {"status": "SKIPPED", "reason": "No tests found"}
                return True

            print(f"Found {test_count} unit tests")
            start_time = time.time()

            result = run_test_suite(suite, verbosity)

            duration = time.time() - start_time
            success = result.wasSuccessful()

            self.results["unit"] = {
                "status": "PASSED" if success else "FAILED",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, "skipped") else 0,
                "duration": f"{duration:.2f}s",
            }

            if not success:
                print(f"\nUnit tests FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
                for test, error in result.failures + result.errors:
                    print(f"  - {test}: {error.split(chr(10))[0]}")
            else:
                print(f"\nUnit tests PASSED: {result.testsRun} tests in {duration:.2f}s")

            return success

        except Exception as e:
            print(f"Error running unit tests: {e}")
            self.results["unit"] = {"status": "ERROR", "error": str(e)}
            return False

    def run_property_tests(self, verbosity: int = 2) -> bool:
        """Run property-based tests"""
        print("\n" + "=" * 60)
        print("RUNNING PROPERTY-BASED TESTS")
        print("=" * 60)

        try:
            import hypothesis

            print(f"Using Hypothesis version {hypothesis.__version__}")
        except ImportError:
            print("Hypothesis not available, skipping property tests")
            self.results["property"] = {"status": "SKIPPED", "reason": "Hypothesis not available"}
            return True

        property_dir = self.test_dirs["property"]
        if not property_dir.exists():
            print(f"Property test directory not found: {property_dir}")
            self.results["property"] = {"status": "SKIPPED", "reason": "Directory not found"}
            return True

        try:
            # Set hypothesis profile for testing
            os.environ["HYPOTHESIS_PROFILE"] = os.environ.get("HYPOTHESIS_PROFILE", "ci")

            suite = discover_tests(str(property_dir), pattern="test_*.py")
            test_count = count_tests(suite)

            if test_count == 0:
                print("No property tests found")
                self.results["property"] = {"status": "SKIPPED", "reason": "No tests found"}
                return True

            print(f"Found {test_count} property tests")
            print(f"Hypothesis profile: {os.environ.get('HYPOTHESIS_PROFILE', 'default')}")

            start_time = time.time()
            result = run_test_suite(suite, verbosity)
            duration = time.time() - start_time

            success = result.wasSuccessful()

            self.results["property"] = {
                "status": "PASSED" if success else "FAILED",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, "skipped") else 0,
                "duration": f"{duration:.2f}s",
            }

            if not success:
                print(f"\nProperty tests FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
            else:
                print(f"\nProperty tests PASSED: {result.testsRun} tests in {duration:.2f}s")

            return success

        except Exception as e:
            print(f"Error running property tests: {e}")
            self.results["property"] = {"status": "ERROR", "error": str(e)}
            return False

    def run_integration_tests(self, verbosity: int = 2) -> bool:
        """Run integration tests"""
        print("\n" + "=" * 60)
        print("RUNNING INTEGRATION TESTS")
        print("=" * 60)

        integration_dir = self.test_dirs["integration"]
        if not integration_dir.exists():
            print(f"Integration test directory not found: {integration_dir}")
            self.results["integration"] = {"status": "SKIPPED", "reason": "Directory not found"}
            return True

        try:
            suite = discover_tests(str(integration_dir))
            test_count = count_tests(suite)

            if test_count == 0:
                print("No integration tests found")
                self.results["integration"] = {"status": "SKIPPED", "reason": "No tests found"}
                return True

            print(f"Found {test_count} integration tests")
            start_time = time.time()

            result = run_test_suite(suite, verbosity)
            duration = time.time() - start_time

            success = result.wasSuccessful()

            self.results["integration"] = {
                "status": "PASSED" if success else "FAILED",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, "skipped") else 0,
                "duration": f"{duration:.2f}s",
            }

            if not success:
                print(f"\nIntegration tests FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
            else:
                print(f"\nIntegration tests PASSED: {result.testsRun} tests in {duration:.2f}s")

            return success

        except Exception as e:
            print(f"Error running integration tests: {e}")
            self.results["integration"] = {"status": "ERROR", "error": str(e)}
            return False

    def run_specific_test(self, test_path: str, verbosity: int = 2) -> bool:
        """Run a specific test file or test method"""
        print(f"\n" + "=" * 60)
        print(f"RUNNING SPECIFIC TEST: {test_path}")
        print("=" * 60)

        try:
            # Load specific test
            loader = unittest.TestLoader()

            if "::" in test_path:
                # Format: module::class::method
                module_path, test_name = test_path.split("::", 1)
                suite = loader.loadTestsFromName(test_name, module=__import__(module_path))
            elif test_path.endswith(".py"):
                # Test file
                suite = loader.discover(".", pattern=os.path.basename(test_path))
            else:
                # Test module or class
                suite = loader.loadTestsFromName(test_path)

            result = run_test_suite(suite, verbosity)
            return result.wasSuccessful()

        except Exception as e:
            print(f"Error running specific test {test_path}: {e}")
            return False

    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("VALIDATION TEST SUMMARY")
        print("=" * 80)

        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        overall_success = True

        for test_type, result in self.results.items():
            status = result.get("status", "UNKNOWN")
            tests_run = result.get("tests_run", 0)
            failures = result.get("failures", 0)
            errors = result.get("errors", 0)
            skipped = result.get("skipped", 0)
            duration = result.get("duration", "N/A")

            if status == "FAILED" or status == "ERROR":
                overall_success = False

            if status != "SKIPPED":
                total_tests += tests_run
                total_failures += failures
                total_errors += errors
                total_skipped += skipped

            print(f"\n{test_type.upper()} TESTS:")
            print(f"  Status: {status}")

            if status == "SKIPPED":
                reason = result.get("reason", "Unknown")
                print(f"  Reason: {reason}")
            elif status in ["PASSED", "FAILED"]:
                print(f"  Tests run: {tests_run}")
                print(f"  Failures: {failures}")
                print(f"  Errors: {errors}")
                print(f"  Skipped: {skipped}")
                print(f"  Duration: {duration}")
            elif status == "ERROR":
                error = result.get("error", "Unknown error")
                print(f"  Error: {error}")

        print(f"\n" + "-" * 50)
        print(f"OVERALL RESULTS:")
        print(f"  Total tests: {total_tests}")
        print(f"  Failures: {total_failures}")
        print(f"  Errors: {total_errors}")
        print(f"  Skipped: {total_skipped}")
        print(f"  Success rate: {((total_tests - total_failures - total_errors) / max(total_tests, 1)) * 100:.1f}%")
        print(f"  Overall status: {'PASSED' if overall_success else 'FAILED'}")

        print("=" * 80)

        return overall_success

    def run_all_tests(
        self, verbosity: int = 2, include_property: bool = True, include_integration: bool = True
    ) -> bool:
        """Run all test suites"""
        print("STARTING COMPREHENSIVE VALIDATION TEST SUITE")
        print(f"Base directory: {self.base_dir}")
        print(f"Python version: {sys.version}")

        overall_success = True

        # Always run unit tests
        if not self.run_unit_tests(verbosity):
            overall_success = False

        # Optionally run property tests
        if include_property:
            if not self.run_property_tests(verbosity):
                overall_success = False

        # Optionally run integration tests
        if include_integration:
            if not self.run_integration_tests(verbosity):
                overall_success = False

        # Print comprehensive summary
        final_success = self.print_summary()

        return final_success and overall_success


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description="Run validation framework tests")
    parser.add_argument(
        "--type",
        choices=["unit", "property", "integration", "all"],
        default="all",
        help="Type of tests to run (default: all)",
    )
    parser.add_argument(
        "--verbosity", "-v", type=int, choices=[0, 1, 2], default=2, help="Test output verbosity (default: 2)"
    )
    parser.add_argument(
        "--specific",
        "-s",
        type=str,
        help="Run specific test file or method (e.g., test_robust_validator.py or module::class::method)",
    )
    parser.add_argument(
        "--hypothesis-profile",
        choices=["ci", "dev", "thorough"],
        default="ci",
        help="Hypothesis testing profile (default: ci)",
    )
    parser.add_argument("--base-dir", type=str, help="Base directory for test discovery")

    args = parser.parse_args()

    # Set hypothesis profile
    os.environ["HYPOTHESIS_PROFILE"] = args.hypothesis_profile

    # Create test runner
    runner = ValidationTestRunner(args.base_dir)

    success = True

    if args.specific:
        # Run specific test
        success = runner.run_specific_test(args.specific, args.verbosity)
    elif args.type == "unit":
        success = runner.run_unit_tests(args.verbosity)
        runner.print_summary()
    elif args.type == "property":
        success = runner.run_property_tests(args.verbosity)
        runner.print_summary()
    elif args.type == "integration":
        success = runner.run_integration_tests(args.verbosity)
        runner.print_summary()
    elif args.type == "all":
        success = runner.run_all_tests(args.verbosity)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

