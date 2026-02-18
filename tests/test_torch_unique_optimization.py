"""
Test script to compare runtime and correctness of torch.unique optimization
in InvariantBasedMessagePassingLayer initialization.

This script tests the optimization that replaces 2D torch.unique with 1D encoding.
"""

import time
import torch
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))


def old_implementation(property_subdict, source_labels, target_labels):
    """Original implementation using 2D torch.unique (slow)"""
    labeled_subdict = property_subdict.clone()
    labeled_subdict[:, 0] = source_labels[property_subdict[:, 0]]
    labeled_subdict[:, 1] = target_labels[property_subdict[:, 1]]

    # Handle invalid indices
    invalid_indices = torch.where(torch.logical_or(labeled_subdict[:, 0] == -1,
                                                    labeled_subdict[:, 1] == -1))[0]
    do_invalid_indices_exist = len(invalid_indices) > 0
    if do_invalid_indices_exist:
        max_first = torch.max(labeled_subdict[:, 0]) + 1
        max_second = torch.max(labeled_subdict[:, 1]) + 1
        labeled_subdict[invalid_indices] = torch.tensor([max_first, max_second])

    # 2D unique (SLOW)
    _, indices, counts = torch.unique(labeled_subdict, dim=0, return_inverse=True,
                                     return_counts=True, sorted=False)

    if do_invalid_indices_exist:
        counts[-1] = 0

    return indices, counts, do_invalid_indices_exist


def new_implementation(property_subdict, source_labels, target_labels):
    """Optimized implementation using 1D encoding (fast)"""
    # Build labeled_subdict directly without clone
    labeled_subdict = torch.stack([
        source_labels[property_subdict[:, 0]],
        target_labels[property_subdict[:, 1]]
    ], dim=1)

    # Handle invalid indices with masking
    invalid_mask = (labeled_subdict[:, 0] == -1) | (labeled_subdict[:, 1] == -1)
    do_invalid_indices_exist = invalid_mask.any().item()

    if do_invalid_indices_exist:
        valid_mask = ~invalid_mask
        max_first = labeled_subdict[valid_mask, 0].max().item() + 1 if valid_mask.any() else 0
        max_second = labeled_subdict[valid_mask, 1].max().item() + 1 if valid_mask.any() else 0
        labeled_subdict[invalid_mask, 0] = max_first
        labeled_subdict[invalid_mask, 1] = max_second
    else:
        max_first = labeled_subdict[:, 0].max().item() + 1
        max_second = labeled_subdict[:, 1].max().item() + 1

    # Encode 2D rows as 1D scalars (KEY OPTIMIZATION)
    max_label = max(max_first, max_second) + 1
    encoded_labels = labeled_subdict[:, 0] * max_label + labeled_subdict[:, 1]

    # 1D unique (FAST)
    _, indices, counts = torch.unique(encoded_labels, return_inverse=True,
                                     return_counts=True, sorted=False)

    if do_invalid_indices_exist:
        counts[-1] = 0

    return indices, counts, do_invalid_indices_exist


def generate_test_data(num_edges, num_source_labels, num_target_labels,
                       invalid_fraction=0.0, seed=42):
    """Generate synthetic test data resembling graph edge labels"""
    torch.manual_seed(seed)

    # Generate edge indices
    property_subdict = torch.randint(0, num_edges * 2, (num_edges, 2), dtype=torch.int64)

    # Generate label arrays
    total_nodes = property_subdict.max().item() + 1
    source_labels = torch.randint(0, num_source_labels, (total_nodes,), dtype=torch.int64)
    target_labels = torch.randint(0, num_target_labels, (total_nodes,), dtype=torch.int64)

    # Add some invalid labels (-1)
    if invalid_fraction > 0:
        num_invalid = int(total_nodes * invalid_fraction)
        invalid_indices = torch.randperm(total_nodes)[:num_invalid]
        source_labels[invalid_indices] = -1
        target_labels[invalid_indices] = -1

    return property_subdict, source_labels, target_labels


def test_correctness():
    """Test that old and new implementations produce identical results"""
    print("=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)

    test_cases = [
        ("Small dataset", 100, 5, 5, 0.0),
        ("Medium dataset", 1000, 10, 10, 0.0),
        ("Large dataset", 10000, 20, 20, 0.0),
        ("With invalid labels (10%)", 1000, 10, 10, 0.1),
        ("With invalid labels (30%)", 1000, 10, 10, 0.3),
        ("Many labels", 5000, 50, 50, 0.0),
    ]

    all_passed = True

    for name, num_edges, num_source, num_target, invalid_frac in test_cases:
        property_subdict, source_labels, target_labels = generate_test_data(
            num_edges, num_source, num_target, invalid_frac
        )

        # Run both implementations
        old_indices, old_counts, old_invalid = old_implementation(
            property_subdict, source_labels, target_labels
        )
        new_indices, new_counts, new_invalid = new_implementation(
            property_subdict, source_labels, target_labels
        )

        # Check equivalence
        indices_match = torch.equal(old_indices, new_indices)
        counts_match = torch.equal(old_counts, new_counts)
        invalid_match = old_invalid == new_invalid

        passed = indices_match and counts_match and invalid_match
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{status} - {name}")
        print(f"  Edges: {num_edges}, Source labels: {num_source}, Target labels: {num_target}")
        print(f"  Invalid fraction: {invalid_frac:.1%}")
        print(f"  Unique combinations found: {len(old_counts)}")

        if not passed:
            print(f"  ERROR: Results differ!")
            print(f"    Indices match: {indices_match}")
            print(f"    Counts match: {counts_match}")
            print(f"    Invalid flag match: {invalid_match}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CORRECTNESS TESTS PASSED")
    else:
        print("✗ SOME CORRECTNESS TESTS FAILED")
    print("=" * 80 + "\n")

    return all_passed


def benchmark_performance():
    """Benchmark runtime comparison between old and new implementations"""
    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)

    test_configs = [
        ("Small (MUTAG-like)", 500, 5, 5, 0.05, 100),
        ("Medium (ZINC-like)", 5000, 15, 15, 0.02, 50),
        ("Large (QM9-like)", 20000, 30, 30, 0.01, 20),
        ("XLarge", 50000, 50, 50, 0.01, 10),
    ]

    print("\nRunning benchmarks (this may take a minute)...\n")

    results = []

    for name, num_edges, num_source, num_target, invalid_frac, num_runs in test_configs:
        property_subdict, source_labels, target_labels = generate_test_data(
            num_edges, num_source, num_target, invalid_frac
        )

        # Warmup
        old_implementation(property_subdict, source_labels, target_labels)
        new_implementation(property_subdict, source_labels, target_labels)

        # Benchmark old implementation
        start = time.time()
        for _ in range(num_runs):
            old_implementation(property_subdict, source_labels, target_labels)
        old_time = (time.time() - start) / num_runs

        # Benchmark new implementation
        start = time.time()
        for _ in range(num_runs):
            new_implementation(property_subdict, source_labels, target_labels)
        new_time = (time.time() - start) / num_runs

        speedup = old_time / new_time
        results.append((name, num_edges, old_time, new_time, speedup))

        print(f"{name}:")
        print(f"  Edges: {num_edges:,}, Labels: {num_source}×{num_target}")
        print(f"  Old implementation: {old_time*1000:.2f} ms")
        print(f"  New implementation: {new_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Time saved: {(old_time - new_time)*1000:.2f} ms ({(1-new_time/old_time)*100:.1f}%)")
        print()

    # Summary
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"\n{'Dataset':<25} {'Edges':<12} {'Old (ms)':<12} {'New (ms)':<12} {'Speedup':<10}")
    print("-" * 80)
    for name, num_edges, old_time, new_time, speedup in results:
        print(f"{name:<25} {num_edges:<12,} {old_time*1000:<12.2f} {new_time*1000:<12.2f} {speedup:<10.1f}x")

    avg_speedup = sum(r[4] for r in results) / len(results)
    print("-" * 80)
    print(f"{'Average speedup:':<49} {avg_speedup:.1f}x")
    print("=" * 80 + "\n")

    return results


def stress_test():
    """Stress test with extreme cases"""
    print("=" * 80)
    print("STRESS TEST (Edge Cases)")
    print("=" * 80)

    test_cases = [
        ("Tiny (1 edge)", 1, 2, 2, 0.0),
        ("All same labels", 1000, 1, 1, 0.0),
        ("Many unique labels", 1000, 100, 100, 0.0),
        ("50% invalid", 1000, 10, 10, 0.5),
        ("90% invalid", 1000, 10, 10, 0.9),
        ("Sparse labels", 10000, 2, 2, 0.0),
    ]

    all_passed = True

    for name, num_edges, num_source, num_target, invalid_frac in test_cases:
        try:
            property_subdict, source_labels, target_labels = generate_test_data(
                num_edges, num_source, num_target, invalid_frac
            )

            old_indices, old_counts, old_invalid = old_implementation(
                property_subdict, source_labels, target_labels
            )
            new_indices, new_counts, new_invalid = new_implementation(
                property_subdict, source_labels, target_labels
            )

            passed = (torch.equal(old_indices, new_indices) and
                     torch.equal(old_counts, new_counts) and
                     old_invalid == new_invalid)

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} - {name}")

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"✗ ERROR - {name}: {e}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL STRESS TESTS PASSED")
    else:
        print("✗ SOME STRESS TESTS FAILED")
    print("=" * 80 + "\n")

    return all_passed


def test_encoding_correctness():
    """Verify that 1D encoding preserves uniqueness"""
    print("=" * 80)
    print("ENCODING VERIFICATION TEST")
    print("=" * 80)

    # Create known label pairs
    label_pairs = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],  # duplicate
        [5, 10],
        [10, 5],  # different from above
        [5, 10],  # duplicate
    ], dtype=torch.int64)

    # Expected unique pairs: [0,0], [0,1], [1,0], [1,1], [5,10], [10,5]
    # Expected indices: 0, 1, 2, 3, 0, 4, 5, 4

    # 2D unique
    unique_2d, indices_2d, counts_2d = torch.unique(
        label_pairs, dim=0, return_inverse=True, return_counts=True, sorted=False
    )

    # 1D encoded unique
    max_label = label_pairs.max().item() + 2  # +2 for safety
    encoded = label_pairs[:, 0] * max_label + label_pairs[:, 1]
    unique_1d, indices_1d, counts_1d = torch.unique(
        encoded, return_inverse=True, return_counts=True, sorted=False
    )

    # Verify indices match
    indices_match = torch.equal(indices_2d, indices_1d)
    counts_match = torch.equal(counts_2d, counts_1d)

    print(f"\nLabel pairs:\n{label_pairs}")
    print(f"\n2D unique found: {len(unique_2d)} unique pairs")
    print(f"1D unique found: {len(unique_1d)} unique encodings")
    print(f"\nIndices match: {indices_match}")
    print(f"Counts match: {counts_match}")

    if indices_match and counts_match:
        print("\n✓ Encoding preserves uniqueness correctly")
    else:
        print("\n✗ Encoding verification FAILED")
        print(f"\n2D indices: {indices_2d}")
        print(f"1D indices: {indices_1d}")
        print(f"\n2D counts: {counts_2d}")
        print(f"1D counts: {counts_1d}")

    print("=" * 80 + "\n")

    return indices_match and counts_match


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("TORCH.UNIQUE OPTIMIZATION TEST SUITE")
    print("=" * 80 + "\n")

    # Test encoding principle
    encoding_ok = test_encoding_correctness()

    # Test correctness
    correctness_ok = test_correctness()

    # Benchmark performance
    if correctness_ok:
        benchmark_performance()
    else:
        print("⚠ Skipping performance benchmark due to correctness failures\n")

    # Stress test
    stress_ok = stress_test()

    # Final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Encoding verification: {'✓ PASS' if encoding_ok else '✗ FAIL'}")
    print(f"Correctness tests:     {'✓ PASS' if correctness_ok else '✗ FAIL'}")
    print(f"Stress tests:          {'✓ PASS' if stress_ok else '✗ FAIL'}")

    all_passed = encoding_ok and correctness_ok and stress_ok

    if all_passed:
        print("\n✓ ALL TESTS PASSED - Optimization is correct and faster!")
    else:
        print("\n✗ SOME TESTS FAILED - Review results above")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
