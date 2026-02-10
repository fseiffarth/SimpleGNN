"""
Test to verify weight distribution caching implementation.

This test validates that the caching code is present and correctly structured.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))


def test_cache_implementation():
    """Test that caching implementation is present"""
    print("\n" + "=" * 80)
    print("CACHE IMPLEMENTATION TEST")
    print("=" * 80)

    # Read the source file directly
    file_path = repo_root / 'src' / 'models' / 'ShareGNN' / 'layers' / 'inv_based_message_passing.py'

    if not file_path.exists():
        print(f"\n✗ Source file not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        source = f.read()

    # Check for required imports
    has_hashlib = "import hashlib" in source
    has_json = "import json" in source
    has_yaml = "import yaml" in source
    has_datetime = "from datetime import datetime" in source

    print("\nChecking imports:")
    print(f"  {'✓' if has_hashlib else '✗'} hashlib")
    print(f"  {'✓' if has_json else '✗'} json")
    print(f"  {'✓' if has_yaml else '✗'} yaml")
    print(f"  {'✓' if has_datetime else '✗'} datetime")

    # Check for helper methods
    has_build_cache_key = "_build_cache_key_dict" in source
    has_generate_cache_key = "_generate_cache_key" in source
    has_build_cache_metadata = "_build_cache_metadata" in source
    has_load_cache = "_load_distribution_cache" in source
    has_save_cache = "_save_distribution_cache" in source

    print("\nChecking helper methods:")
    print(f"  {'✓' if has_build_cache_key else '✗'} _build_cache_key_dict")
    print(f"  {'✓' if has_generate_cache_key else '✗'} _generate_cache_key")
    print(f"  {'✓' if has_build_cache_metadata else '✗'} _build_cache_metadata")
    print(f"  {'✓' if has_load_cache else '✗'} _load_distribution_cache")
    print(f"  {'✓' if has_save_cache else '✗'} _save_distribution_cache")

    # Check for cache logic in __init__
    has_cache_key_generation = "cache_key_dict = self._build_cache_key_dict" in source
    has_cache_hash = "cache_hash = self._generate_cache_key" in source
    has_cache_dir_creation = "weight_distributions" in source
    has_cache_check = "if self._load_distribution_cache" in source
    has_cache_miss_message = "Cache miss" in source or "⊗" in source
    has_cache_hit_message = "Cache hit" in source or "✓" in source
    has_cache_save_call = "self._save_distribution_cache" in source

    print("\nChecking cache logic in __init__:")
    print(f"  {'✓' if has_cache_key_generation else '✗'} Cache key generation")
    print(f"  {'✓' if has_cache_hash else '✗'} Cache hash computation")
    print(f"  {'✓' if has_cache_dir_creation else '✗'} Cache directory setup")
    print(f"  {'✓' if has_cache_check else '✗'} Cache load check")
    print(f"  {'✓' if has_cache_miss_message else '✗'} Cache miss message")
    print(f"  {'✓' if has_cache_hit_message else '✗'} Cache hit message")
    print(f"  {'✓' if has_cache_save_call else '✗'} Cache save call")

    # Check for specific cache file patterns
    has_pt_extension = '.pt' in source
    has_yml_extension = '.yml' in source
    has_torch_save = "torch.save" in source
    has_torch_load = "torch.load" in source
    has_yaml_dump = "yaml.dump" in source

    print("\nChecking cache file operations:")
    print(f"  {'✓' if has_pt_extension else '✗'} .pt extension")
    print(f"  {'✓' if has_yml_extension else '✗'} .yml extension")
    print(f"  {'✓' if has_torch_save else '✗'} torch.save")
    print(f"  {'✓' if has_torch_load else '✗'} torch.load")
    print(f"  {'✓' if has_yaml_dump else '✗'} yaml.dump")

    # Check for cache key components
    has_layer_id_in_key = "'layer_id':" in source or "'layer_id'" in source
    has_dataset_name_in_key = "'dataset_name':" in source or "'dataset_name'" in source
    has_thresholds_in_key = "rule_occurrence_threshold" in source

    print("\nChecking cache key components:")
    print(f"  {'✓' if has_layer_id_in_key else '✗'} layer_id")
    print(f"  {'✓' if has_dataset_name_in_key else '✗'} dataset_name")
    print(f"  {'✓' if has_thresholds_in_key else '✗'} threshold parameters")

    # Check for SHA256 hashing
    has_sha256 = "sha256" in source
    has_hexdigest = "hexdigest" in source

    print("\nChecking hashing implementation:")
    print(f"  {'✓' if has_sha256 else '✗'} SHA256")
    print(f"  {'✓' if has_hexdigest else '✗'} hexdigest")

    # Count all checks
    all_checks = [
        has_hashlib, has_json, has_yaml, has_datetime,
        has_build_cache_key, has_generate_cache_key, has_build_cache_metadata, has_load_cache, has_save_cache,
        has_cache_key_generation, has_cache_hash, has_cache_dir_creation, has_cache_check,
        has_cache_miss_message, has_cache_hit_message, has_cache_save_call,
        has_pt_extension, has_yml_extension, has_torch_save, has_torch_load, has_yaml_dump,
        has_layer_id_in_key, has_dataset_name_in_key, has_thresholds_in_key,
        has_sha256, has_hexdigest
    ]

    passed = sum(all_checks)
    total = len(all_checks)

    print("\n" + "=" * 80)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 80)

    if passed == total:
        print("\n✓ ALL CACHE IMPLEMENTATION CHECKS PASSED")
        return True
    else:
        print(f"\n⚠ Warning: {total - passed} checks failed")
        return False


def test_cache_structure():
    """Verify cache structure in code"""
    print("\n" + "=" * 80)
    print("CACHE STRUCTURE VALIDATION")
    print("=" * 80)

    file_path = repo_root / 'src' / 'models' / 'ShareGNN' / 'layers' / 'inv_based_message_passing.py'

    if not file_path.exists():
        print(f"\n✗ Source file not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find key sections
    init_start = None
    for i, line in enumerate(lines):
        if "def __init__" in line and "InvariantBasedMessagePassingLayer" not in line:
            init_start = i
            break

    if init_start is None:
        print("✗ Could not find __init__ method")
        return False

    print(f"✓ Found __init__ method at line {init_start + 1}")

    # Look for cache logic early in __init__ (should be before computation)
    cache_logic_found = False
    computation_start = None

    for i in range(init_start, min(init_start + 200, len(lines))):
        if "cache_key_dict" in lines[i]:
            cache_logic_found = True
            print(f"✓ Found cache key generation at line {i + 1}")
        if "weight_distribution_chunks" in lines[i]:
            computation_start = i
            print(f"✓ Found computation start at line {i + 1}")
            break

    if not cache_logic_found:
        print("✗ Cache logic not found in __init__")
        return False

    if computation_start and cache_logic_found:
        print("✓ Cache logic appears before computation (correct order)")

    print("\n✓ CACHE STRUCTURE VALIDATION PASSED")
    return True


if __name__ == '__main__':
    test1 = test_cache_implementation()
    test2 = test_cache_structure()

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    if test1 and test2:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("⚠ Some tests failed")
        sys.exit(1)
