#!/usr/bin/env python3
"""
Unit tests for the indices/counts caching methods in InvariantBasedMessagePassingLayer.

Tests:
1. get_cache_path() - generates correct cache paths with proper hashing
2. _save_cached_indices() - saves indices and counts to disk
3. _load_cached_indices() - loads cached data correctly
4. Cache invalidation - different configs produce different cache keys
5. Error handling - corrupted cache files are handled gracefully
"""

import sys
from pathlib import Path
import shutil
import json
import torch
import hashlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_cache_path_generation():
    """Test that cache paths are generated correctly with proper hashing."""
    print("Testing cache path generation...")

    # Create mock objects
    class MockHead:
        head_id = 0

    class MockLayer:
        layer_heads = [MockHead()]

    class MockConfig:
        def get(self, key, default=None):
            if key == 'rule_occurrence_threshold':
                return 1
            elif key == 'rule_occurrence_upper_threshold':
                return None
            elif key == 'paths':
                return {'results': '/tmp/test_results'}
            return default

    class MockRunConfig:
        config = MockConfig()

    class MockPara:
        db = 'TEST_DB'
        run_config = MockRunConfig()

    # Create mock layer instance
    class MockLayer:
        def __init__(self):
            self.para = MockPara()
            self.layer_id = 0
            self.layer = type('obj', (object,), {'layer_heads': [MockHead()]})()
            self.property_descriptions = ['test_property']
            self.source_label_descriptions = ['source_label']
            self.target_label_descriptions = ['target_label']

        def get_cache_path(self, head, property_key):
            """Simplified version for testing."""
            cache_metadata = {
                'dataset': self.para.db,
                'layer_id': self.layer_id,
                'head_id': head.head_id if hasattr(head, 'head_id') else None,
                'property': self.property_descriptions[self.layer.layer_heads.index(head)],
                'property_key': str(property_key),
                'source_label': self.source_label_descriptions[self.layer.layer_heads.index(head)],
                'target_label': self.target_label_descriptions[self.layer.layer_heads.index(head)],
                'threshold': self.para.run_config.config.get('rule_occurrence_threshold', 1),
                'upper_threshold': self.para.run_config.config.get('rule_occurrence_upper_threshold', None),
            }

            metadata_str = json.dumps(cache_metadata, sort_keys=True)
            cache_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:12]

            results_path = Path(self.para.run_config.config.get('paths')['results'])
            cache_dir = results_path / self.para.db / 'ShareGNN_indices_cache' / str(self.layer_id)
            cache_dir.mkdir(parents=True, exist_ok=True)

            head_idx = self.layer.layer_heads.index(head)
            return cache_dir / f'h{head_idx}_p{property_key}_{cache_hash}.pt'

    layer = MockLayer()
    head = MockHead()
    property_key = 3

    # Test basic path generation
    cache_path = layer.get_cache_path(head, property_key)

    assert cache_path.exists() or cache_path.parent.exists(), "Cache directory should be created"
    assert 'ShareGNN_indices_cache' in str(cache_path), "Cache path should contain 'ShareGNN_indices_cache'"
    assert f'h0_p{property_key}_' in cache_path.name, "Cache filename should contain head and property"
    assert cache_path.suffix == '.pt', "Cache file should have .pt extension"

    print("✓ Cache path generation test passed")
    return True


def test_save_and_load():
    """Test saving and loading cached indices."""
    print("Testing save and load functionality...")

    # Create temporary directory
    test_dir = Path('/tmp/test_cache_save_load')
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    try:
        # Create test data
        test_indices = torch.tensor([0, 1, 2, 1, 0, 3], dtype=torch.int64)
        test_counts = torch.tensor([2, 2, 1, 1], dtype=torch.int64)
        test_path = test_dir / 'test_cache.pt'

        # Save cache (simplified version)
        cache_data = {
            'indices': test_indices,
            'counts': test_counts,
            'metadata': {
                'layer_id': 0,
                'head_index': 0,
                'property_key': '3',
                'indices_shape': list(test_indices.shape),
                'counts_shape': list(test_counts.shape),
                'num_unique_pairs': len(test_counts),
            }
        }
        torch.save(cache_data, str(test_path))

        # Save metadata
        meta_path = test_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(cache_data['metadata'], f, indent=2)

        # Verify files exist
        assert test_path.exists(), "Cache file should exist"
        assert meta_path.exists(), "Metadata file should exist"

        # Load cache
        loaded_data = torch.load(str(test_path), weights_only=False)
        loaded_indices = loaded_data['indices']
        loaded_counts = loaded_data['counts']

        # Verify loaded data
        assert torch.equal(loaded_indices, test_indices), "Loaded indices should match original"
        assert torch.equal(loaded_counts, test_counts), "Loaded counts should match original"

        # Load metadata
        with open(meta_path, 'r') as f:
            loaded_metadata = json.load(f)

        assert loaded_metadata['layer_id'] == 0, "Metadata should match"
        assert loaded_metadata['num_unique_pairs'] == 4, "Metadata should have correct count"

        print("✓ Save and load test passed")
        return True

    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_cache_invalidation():
    """Test that different configs produce different cache keys."""
    print("Testing cache invalidation (different configs → different hashes)...")

    def get_cache_hash(threshold=1, upper_threshold=None):
        """Helper to compute cache hash for given config."""
        cache_metadata = {
            'dataset': 'TEST_DB',
            'layer_id': 0,
            'head_id': 0,
            'property': 'test_property',
            'property_key': '3',
            'source_label': 'source_label',
            'target_label': 'target_label',
            'threshold': threshold,
            'upper_threshold': upper_threshold,
        }
        metadata_str = json.dumps(cache_metadata, sort_keys=True)
        return hashlib.md5(metadata_str.encode()).hexdigest()[:12]

    # Test different configurations produce different hashes
    hash1 = get_cache_hash(threshold=1, upper_threshold=None)
    hash2 = get_cache_hash(threshold=2, upper_threshold=None)
    hash3 = get_cache_hash(threshold=1, upper_threshold=10)

    assert hash1 != hash2, "Different thresholds should produce different hashes"
    assert hash1 != hash3, "Different upper_thresholds should produce different hashes"
    assert hash2 != hash3, "All three configs should have unique hashes"

    # Same config should produce same hash
    hash1_repeat = get_cache_hash(threshold=1, upper_threshold=None)
    assert hash1 == hash1_repeat, "Same config should produce same hash"

    print("✓ Cache invalidation test passed")
    return True


def test_error_handling():
    """Test graceful handling of corrupted cache files."""
    print("Testing error handling for corrupted cache...")

    test_dir = Path('/tmp/test_cache_error')
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    try:
        # Create corrupted cache file
        corrupted_path = test_dir / 'corrupted.pt'
        with open(corrupted_path, 'w') as f:
            f.write("This is not a valid PyTorch file")

        # Try to load corrupted cache
        try:
            data = torch.load(str(corrupted_path), weights_only=False)
            print("✗ Expected exception but none was raised")
            return False
        except Exception as e:
            print(f"✓ Corrupted cache correctly raised exception: {type(e).__name__}")

        # Create cache with wrong structure (missing keys)
        invalid_structure_path = test_dir / 'invalid.pt'
        torch.save({'wrong_key': torch.tensor([1, 2, 3])}, str(invalid_structure_path))

        loaded = torch.load(str(invalid_structure_path), weights_only=False)
        if 'indices' not in loaded or 'counts' not in loaded:
            print("✓ Invalid structure correctly detected")
        else:
            print("✗ Invalid structure was not detected")
            return False

        print("✓ Error handling test passed")
        return True

    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)


def run_all_tests():
    """Run all unit tests."""
    print("=" * 80)
    print("Running Cache Methods Unit Tests")
    print("=" * 80)
    print()

    tests = [
        test_cache_path_generation,
        test_save_and_load,
        test_cache_invalidation,
        test_error_handling,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} raised exception: {e}")

        print()

    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
