#!/usr/bin/env python3
"""
Test script to verify the indices and counts caching implementation.

This script runs the basic ShareGNN example twice:
1. First run with clean cache (should show cache misses and create cache files)
2. Second run should hit the cache (much faster initialization)

Expected behavior:
- First run: "⊗ Cache miss" messages, cache files created
- Second run: "✓ Cache hit" messages, faster initialization
"""

import subprocess
import sys
import shutil
from pathlib import Path
import time

def run_test():
    """Run the caching test."""

    # Define paths
    repo_root = Path(__file__).parent
    src_dir = repo_root / 'src'
    cache_dir = repo_root / 'results' / 'MUTAG' / 'ShareGNN_indices_cache'

    print("=" * 80)
    print("ShareGNN Indices/Counts Caching Test")
    print("=" * 80)
    print()

    # Step 1: Clean cache
    print("Step 1: Cleaning cache...")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"✓ Removed cache directory: {cache_dir}")
    else:
        print(f"✓ Cache directory doesn't exist (clean start)")
    print()

    # Step 2: First run (cache miss expected)
    print("Step 2: First run (expecting cache misses)...")
    print("-" * 80)
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, '-m', 'examples.basic_example_share_gnn.main'],
            cwd=src_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        first_run_time = time.time() - start_time

        # Check for cache miss messages
        cache_misses = result.stdout.count('⊗ Cache miss')
        cache_hits = result.stdout.count('✓ Cache hit')

        print(f"First run completed in {first_run_time:.2f} seconds")
        print(f"Cache misses: {cache_misses}")
        print(f"Cache hits: {cache_hits}")

        if cache_misses == 0:
            print("⚠ WARNING: Expected cache misses but found none!")

        # Check if cache files were created
        if cache_dir.exists():
            cache_files = list(cache_dir.glob('**/*.pt'))
            json_files = list(cache_dir.glob('**/*.json'))
            print(f"✓ Cache directory created: {cache_dir}")
            print(f"  - Cache files (.pt): {len(cache_files)}")
            print(f"  - Metadata files (.json): {len(json_files)}")

            # Show sample cache file
            if cache_files:
                print(f"  - Sample cache file: {cache_files[0].name}")
        else:
            print("✗ ERROR: Cache directory was not created!")
            return False

    except subprocess.TimeoutExpired:
        print("✗ ERROR: First run timed out!")
        return False
    except Exception as e:
        print(f"✗ ERROR during first run: {e}")
        return False

    print()

    # Step 3: Second run (cache hit expected)
    print("Step 3: Second run (expecting cache hits)...")
    print("-" * 80)
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, '-m', 'examples.basic_example_share_gnn.main'],
            cwd=src_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        second_run_time = time.time() - start_time

        # Check for cache hit messages
        cache_misses = result.stdout.count('⊗ Cache miss')
        cache_hits = result.stdout.count('✓ Cache hit')

        print(f"Second run completed in {second_run_time:.2f} seconds")
        print(f"Cache misses: {cache_misses}")
        print(f"Cache hits: {cache_hits}")

        if cache_hits == 0:
            print("⚠ WARNING: Expected cache hits but found none!")
            return False

        # Calculate speedup
        if first_run_time > 0:
            speedup = first_run_time / second_run_time
            print(f"Speedup: {speedup:.2f}x")

            if speedup < 1.1:
                print("⚠ WARNING: Expected significant speedup but got minimal improvement")
            elif speedup > 2:
                print(f"✓ Excellent speedup achieved!")
            else:
                print(f"✓ Modest speedup achieved")

    except subprocess.TimeoutExpired:
        print("✗ ERROR: Second run timed out!")
        return False
    except Exception as e:
        print(f"✗ ERROR during second run: {e}")
        return False

    print()
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    return True

if __name__ == '__main__':
    success = run_test()
    sys.exit(0 if success else 1)
