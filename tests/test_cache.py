#!/usr/bin/env python3
"""
Test script to verify weight distribution caching works correctly.
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from framework.core import FrameworkMain

def test_caching():
    """Test that caching speeds up subsequent runs."""
    config_path = Path('examples/basic_example_share_gnn/main.yml')

    print("=" * 80)
    print("CACHE TEST: First Run (Cache Miss Expected)")
    print("=" * 80)

    # First run - should compute and save to cache
    start_time = time.time()
    experiment1 = FrameworkMain(config_path)
    experiment1.preprocessing(num_threads=1)
    first_run_time = time.time() - start_time

    print(f"\nFirst run completed in {first_run_time:.2f} seconds")

    print("\n" + "=" * 80)
    print("CACHE TEST: Second Run (Cache Hit Expected)")
    print("=" * 80)

    # Second run - should load from cache
    start_time = time.time()
    experiment2 = FrameworkMain(config_path)
    experiment2.preprocessing(num_threads=1)
    second_run_time = time.time() - start_time

    print(f"\nSecond run completed in {second_run_time:.2f} seconds")

    print("\n" + "=" * 80)
    print("CACHE TEST RESULTS")
    print("=" * 80)
    print(f"First run time:  {first_run_time:.2f}s (compute + cache save)")
    print(f"Second run time: {second_run_time:.2f}s (cache load)")

    if second_run_time < first_run_time * 0.5:
        speedup = first_run_time / second_run_time
        print(f"✓ Cache speedup: {speedup:.1f}x")
        print("✓ CACHE TEST PASSED")
    else:
        print(f"⚠ Warning: Second run not significantly faster")
        print(f"  Expected <50% of first run, got {(second_run_time/first_run_time)*100:.1f}%")

if __name__ == '__main__':
    test_caching()
