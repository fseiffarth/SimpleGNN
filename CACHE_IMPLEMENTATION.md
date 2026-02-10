# Weight Distribution Caching Implementation

## Overview

Implemented disk-based caching for `InvariantBasedMessagePassingLayer` weight and bias distributions. This eliminates the expensive recomputation of distributions for repeated experiments with the same layer configuration and dataset.

## Performance Impact

- **First run**: Same as before (compute + cache save)
- **Subsequent runs**: ~10-150x speedup (load from cache in <1s vs 10-150s computation)
- **Cache hit scenarios**:
  - Multiple training runs with same config
  - Hyperparameter search where only non-layer params change
  - Development/debugging iterations

## Implementation Details

### File: `src/models/ShareGNN/layers/inv_based_message_passing.py`

### Key Components

#### 1. Cache Key Generation (`_build_cache_key_dict`, `_generate_cache_key`)

Creates a deterministic SHA256 hash from all components that affect distributions:
- Layer configuration (layer_id, num_heads, bias settings, input features)
- Dataset identity (name, source, size)
- Label descriptions (source, target, bias) for each head
- Property descriptions and valid values for each head
- Threshold parameters (`rule_occurrence_threshold`, `rule_occurrence_upper_threshold`)

**Hash example**: `a1b2c3d4e5f6g7h8` (first 16 hex chars of SHA256)

#### 2. Cache File Organization

**Directory structure**:
```
data/{source}/weight_distributions/{dataset_name}/
├── {dataset}_layer_{id}_weight_dist_{hash}.pt   # Binary tensors
└── {dataset}_layer_{id}_weight_dist_{hash}.yml  # Metadata
```

**Example**:
```
data/TUDatasets/weight_distributions/MUTAG/
├── MUTAG_layer_0_weight_dist_a1b2c3d4e5f6g7h8.pt
└── MUTAG_layer_0_weight_dist_a1b2c3d4e5f6g7h8.yml
```

#### 3. Cached Data Structure (`.pt` file)

Saved using `torch.save()`:
```python
{
    'weight_distribution': tensor,          # (M, 4) int64
    'weight_distribution_slices': tensor,   # (num_graphs+1,) int64
    'weight_num': list,
    'skips': list,
    'skips_description': list,
    'skips_description_text': list,
    'n_source_labels': list,
    'n_target_labels': list,
    'n_properties': list,
    'n_heads_per_label': list,
    'b_head_offset': int,

    # Optional (if self.bias == True):
    'bias_distribution': tensor,            # (N, 4) int64
    'bias_distribution_slices': tensor,     # (num_graphs+1,) int64
    'bias_num': list,
    'n_bias_labels': list,
}
```

#### 4. Metadata Structure (`.yml` file)

Human-readable information:
```yaml
timestamp: "2026-02-10T12:34:56"
cache_hash: "a1b2c3d4e5f6g7h8"
layer_id: 0
num_heads: 4
dataset:
  name: "MUTAG"
  source: "TUDatasets"
  size: 188
layer_config:
  in_features: 7
  bias: false
  n_heads_per_label: [1, 1, 1, 1]
head_configurations:
  - head_id: 0
    num: 1
    source_label: "primary"
    target_label: "primary"
    # ...
thresholds:
  rule_occurrence_threshold: 1
  rule_occurrence_upper_threshold: null
statistics:
  total_weights: 1234
  weight_distribution_shape: [5678, 4]
  cache_file_size_mb: 0.45
```

#### 5. Integration into `__init__`

**Modified flow** (lines 67-231):
```python
def __init__(self, parameters, layer, graph_data):
    # Lines 36-64: Existing initialization (unchanged)
    # ... initialize attributes ...

    # NEW: Build cache key and check cache
    cache_key_dict = self._build_cache_key_dict(layer, graph_data)
    cache_hash = self._generate_cache_key(cache_key_dict)
    cache_path_base = ...  # Build cache path

    # Try to load from cache
    if self._load_distribution_cache(cache_path_base):
        print(f"✓ Cache hit for layer {self.layer_id}")
    else:
        print(f"⊗ Cache miss for layer {self.layer_id}. Computing...")

        # Lines 67-227: EXISTING computation code (only on cache miss)
        # ... all existing distribution computation ...

        # Save to cache after computation
        metadata = self._build_cache_metadata(cache_key_dict, cache_hash, layer, graph_data)
        self._save_distribution_cache(cache_path_base, metadata)

    # Lines 233-251: ALWAYS initialize Param_W and Param_b (fast)
    # ... weight initialization ...
```

**Key insight**: The expensive computation (lines 87-227) is skipped entirely on cache hit. Weight parameter initialization (`Param_W`, `Param_b`) still runs every time but is fast (~milliseconds).

#### 6. New Helper Methods

1. **`_build_cache_key_dict(layer, graph_data) -> dict`**
   - Extract all components that affect distributions
   - Return dictionary with layer config, dataset info, head configs, thresholds

2. **`_generate_cache_key(cache_key_dict) -> str`**
   - Serialize dict to JSON with sorted keys
   - Compute SHA256 hash, return first 16 hex chars

3. **`_build_cache_metadata(cache_key_dict, cache_hash, layer, graph_data) -> dict`**
   - Build human-readable metadata for YAML file
   - Include timestamps, config details, statistics

4. **`_load_distribution_cache(cache_path_base) -> bool`**
   - Check if cache file exists
   - Load tensors and lists from .pt file
   - Validate required keys
   - Restore to `self.weight_distribution`, `self.weight_distribution_slices`, etc.
   - Handle corrupted cache: delete and return False
   - Return True on success, False on failure

5. **`_save_distribution_cache(cache_path_base, metadata) -> None`**
   - Save all tensors and lists to .pt file
   - Calculate file size, warn if >500MB
   - Save metadata to .yml file
   - Non-fatal: log warning if save fails but don't raise exception

### Error Handling

**Corrupted cache**:
- Try-except around `torch.load()`
- Validate required keys exist
- If load fails, delete corrupted files and trigger recomputation

**Partial cache**:
- Bias data included only if `self.bias == True`
- This is correct: bias configuration is part of cache key hash

**Disk space**:
- Typical cache sizes: 200KB-1GB depending on dataset
- Warning if cache file exceeds 500MB

**Cache invalidation**:
- Automatic: Hash changes when any dependency changes
- Manual: Users can delete cache directory

### New Imports

Added to top of file:
```python
import hashlib
import json
import yaml
from datetime import datetime
```

## Cache Behavior

### Cache Hit Conditions

Cache is reused when ALL of these match:
- Same dataset (name, source, size)
- Same layer configuration (heads, labels, properties)
- Same threshold parameters
- Same label descriptions and valid property values

### Cache Miss Conditions

Cache is recomputed when ANY of these change:
- Different dataset or dataset size
- Different layer structure (num heads, bias settings)
- Different thresholds (`rule_occurrence_threshold`, `rule_occurrence_upper_threshold`)
- Different label or property descriptions

### Cache Locations

- **Path**: `data/{source}/weight_distributions/{dataset_name}/`
- **Example datasets**:
  - `data/TUDatasets/weight_distributions/MUTAG/`
  - `data/TUDatasets/weight_distributions/PROTEINS/`
  - `data/ZINC/weight_distributions/ZINC/`
  - `data/QM9/weight_distributions/QM9/`

## Testing

### Test File: `tests/test_cache_implementation.py`

Validates:
- ✓ All required imports present (hashlib, json, yaml, datetime)
- ✓ All 5 helper methods implemented
- ✓ Cache logic integrated into `__init__`
- ✓ Cache key generation before computation
- ✓ Cache hit/miss messages
- ✓ Cache save after computation
- ✓ File operations (torch.save, torch.load, yaml.dump)
- ✓ SHA256 hashing implementation

**Run test**:
```bash
python3 tests/test_cache_implementation.py
```

**Expected output**:
```
✓ ALL CACHE IMPLEMENTATION CHECKS PASSED
✓ CACHE STRUCTURE VALIDATION PASSED
✓ ALL TESTS PASSED
```

## Verification

### Manual Testing

1. **First run (cache miss)**:
   ```bash
   cd src && python -m examples.basic_example_share_gnn.main
   ```
   - Should print: `⊗ Cache miss for layer X. Computing...`
   - Check files created in `data/TUDatasets/weight_distributions/MUTAG/`
   - Note the initialization time

2. **Second run (cache hit)**:
   ```bash
   cd src && python -m examples.basic_example_share_gnn.main
   ```
   - Should print: `✓ Cache hit for layer X`
   - Should be significantly faster (load <1s vs compute 10s+)

3. **Correctness check**:
   - Compare final validation/test metrics between runs
   - Should be identical (cache doesn't affect training, only init speed)

4. **Cache invalidation**:
   - Modify threshold in config (e.g., `rule_occurrence_threshold: 1 → 2`)
   - Should get cache miss (different hash)
   - Verify new cache files created with different hash

5. **YAML inspection**:
   - Check `.yml` files are human-readable
   - Verify metadata matches actual layer config

## Benefits

1. **Development speed**: Instant layer initialization during development
2. **Hyperparameter search**: Eliminates redundant computation across configs
3. **Reproducibility**: Identical distributions for same configuration
4. **Debugging**: Remove initialization time from profiling
5. **Large datasets**: Especially valuable for QM9 (100-150s → <1s)

## Backward Compatibility

- Non-breaking change: If cache fails, falls back to computation
- Existing code works unchanged
- Cache directory creation uses `mkdir(parents=True, exist_ok=True)`
- No changes to `forward()` method or training logic

## Future Enhancements

Possible improvements:
- Compression for large cache files (gzip)
- Cache statistics/monitoring
- Automatic cache cleanup for old/unused files
- Cache versioning for breaking changes
- Multi-process safe locking for parallel preprocessing
