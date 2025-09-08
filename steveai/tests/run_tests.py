# SteveAI - Test Runner

import argparse
import os
import sys
import time
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from steveai.tests.test_integration import run_integration_tests
from steveai.tests.test_performance import run_performance_tests

# Import test modules
from steveai.tests.test_steveai import run_tests


def run_all_tests():
    """Run all test suites."""
    print("🚀 Starting SteveAI Test Suite")
    print("=" * 50)

    start_time = time.time()

    # Run unit tests
    print("\n📋 Running Unit Tests...")
    print("-" * 30)
    unit_success = run_tests()

    # Run integration tests
    print("\n🔗 Running Integration Tests...")
    print("-" * 30)
    integration_success = run_integration_tests()

    # Run performance tests
    print("\n⚡ Running Performance Tests...")
    print("-" * 30)
    performance_success = run_performance_tests()

    end_time = time.time()
    total_time = end_time - start_time

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    print(f"Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
    print(f"Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print(f"Performance Tests: {'✅ PASSED' if performance_success else '❌ FAILED'}")

    print(f"\nTotal Time: {total_time:.2f} seconds")

    all_success = unit_success and integration_success and performance_success

    if all_success:
        print("\n🎉 All tests passed! SteveAI is ready for production!")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")

    return all_success


def run_specific_tests(test_type):
    """Run specific test type."""
    print(f"🚀 Running {test_type.title()} Tests")
    print("=" * 50)

    start_time = time.time()

    if test_type == "unit":
        success = run_unit_tests()
    elif test_type == "integration":
        success = run_integration_tests()
    elif test_type == "performance":
        success = run_performance_tests()
    else:
        print(f"❌ Unknown test type: {test_type}")
        return False

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal Time: {total_time:.2f} seconds")

    if success:
        print(f"\n✅ {test_type.title()} tests passed!")
    else:
        print(f"\n❌ {test_type.title()} tests failed!")

    return success


def run_quick_tests():
    """Run quick tests (unit tests only)."""
    print("🚀 Running Quick Tests (Unit Tests Only)")
    print("=" * 50)

    start_time = time.time()
    success = run_unit_tests()
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal Time: {total_time:.2f} seconds")

    if success:
        print("\n✅ Quick tests passed!")
    else:
        print("\n❌ Quick tests failed!")

    return success


def run_ci_tests():
    """Run tests suitable for CI/CD."""
    print("🚀 Running CI/CD Tests")
    print("=" * 50)

    start_time = time.time()

    # Run unit and integration tests (skip performance tests for CI)
    print("\n📋 Running Unit Tests...")
    unit_success = run_tests()

    print("\n🔗 Running Integration Tests...")
    integration_success = run_integration_tests()

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal Time: {total_time:.2f} seconds")

    all_success = unit_success and integration_success

    if all_success:
        print("\n✅ CI/CD tests passed!")
    else:
        print("\n❌ CI/CD tests failed!")

    return all_success


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="SteveAI Test Runner")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "performance", "quick", "ci"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--failfast", "-f", action="store_true", help="Stop on first failure"
    )

    args = parser.parse_args()

    # Set up environment
    if args.verbose:
        os.environ["TEST_VERBOSE"] = "1"

    if args.failfast:
        os.environ["TEST_FAILFAST"] = "1"

    # Run tests based on type
    if args.type == "all":
        success = run_all_tests()
    elif args.type == "quick":
        success = run_quick_tests()
    elif args.type == "ci":
        success = run_ci_tests()
    else:
        success = run_specific_tests(args.type)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
