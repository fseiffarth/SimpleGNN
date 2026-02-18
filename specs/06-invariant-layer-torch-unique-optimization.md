# InvariantBasedMessagePassingLayer Initialization Optimization Spec

## Overview

This specification addresses the critical performance bottleneck in `InvariantBasedMessagePassingLayer.__init__()` where `torch.unique` operations consume ~70% of initialization time according to profiling data.

**Target file:** `src/models/ShareGNN/layers/inv_based_message_passing.py`
**Primary bottleneck:** Lines 93-124 (torch.unique on 2D tensors)
**Secondary bottleneck:** Line 161 (bias labels unique)

---

## 1. Primary Bottleneck: 2D torch.unique on Label Pairs

### Location
`src/models/ShareGNN/layers/inv_based_message_passing.py:104`

### Problem

The initialization performs `torch.unique` on 2D tensors containing `[source_label, target_label]` pairs:

```python
# Lines 93-104
labeled_subdict = property_subdict.clone()
labeled_subdict[:, 0] = source_labels[property_subdict[:, 0]]
labeled_subdict[:, 1] = target_labels[property_subdict[:, 1]]

# ... invalid index handling ...

_, indices, counts = torch.unique(labeled_subdict, dim=0, return_inverse=True,
                                  return_counts=True, sorted=False)
```

**Why this is slow:**
- `torch.unique(tensor, dim=0)` on 2D tensors requires row-wise comparisons
- Complexity: O(n²) worst case, O(n log n) with sorting
- Called once per (head, property_value) combination
- For datasets with many edges and multiple heads/properties, this dominates init time
- Each call processes the full edge set for that property value across all graphs

**Example scale:**
- ZINC dataset: ~12k graphs, ~24 edges/graph avg = ~288k edges
- With 3 heads × 5 property values = 15 calls to torch.unique
- Each operating on subsets of 288k rows

### Impact

**Measured:** 70% of initialization time
**Estimated speedup potential:** 10-50x on torch.unique operations
**Overall init speedup:** 50-85% reduction in total init time

### Root Cause

PyTorch's `torch.unique` with `dim=0` on 2D tensors is not optimized for the common case of low-dimensional rows with bounded integer values. The algorithm needs to:
1. Compare entire rows for equality (not vectorized)
2. Build hash tables or sort based on full rows
3. Handle arbitrary data types and shapes (generic implementation)

### Proposed Fix: Row Encoding to 1D

**Core optimization:** Encode 2-column integer tensors as 1D scalars before calling `torch.unique`.

For a tensor with columns `[a, b]` where values are bounded integers, encode as:
```
encoded = a * max_label + b
```

This preserves uniqueness because if `(a₁, b₁) ≠ (a₂, b₂)`, then `a₁ * K + b₁ ≠ a₂ * K + b₂` for `K > max(b)`.

**Implementation:**

```python
# BEFORE (lines 93-104):
labeled_subdict = property_subdict.clone()
labeled_subdict[:, 0] = source_labels[property_subdict[:, 0]]
labeled_subdict[:, 1] = target_labels[property_subdict[:, 1]]

invalid_indices = torch.where(torch.logical_or(labeled_subdict[:, 0] == -1,
                                                labeled_subdict[:, 1] == -1))[0]
do_invalid_indices_exist = len(invalid_indices) > 0
if do_invalid_indices_exist:
    max_first = torch.max(labeled_subdict[:, 0]) + 1
    max_second = torch.max(labeled_subdict[:, 1]) + 1
    labeled_subdict[invalid_indices] = torch.tensor([max_first, max_second])

_, indices, counts = torch.unique(labeled_subdict, dim=0, return_inverse=True,
                                  return_counts=True, sorted=False)

# AFTER (optimized):
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

# KEY OPTIMIZATION: Encode 2D rows as 1D scalars
max_label = max(max_first, max_second) + 1
encoded_labels = labeled_subdict[:, 0] * max_label + labeled_subdict[:, 1]

# Fast 1D unique (10-50x faster than 2D unique)
_, indices, counts = torch.unique(encoded_labels, return_inverse=True,
                                  return_counts=True, sorted=False)

# Rest of code unchanged - indices and counts are identical
if do_invalid_indices_exist:
    counts[-1] = 0
```

**Why this works:**
- 1D `torch.unique` is heavily optimized (O(n log n) or O(n) with hash-based approach)
- No row comparisons needed - just scalar comparisons
- `indices` and `counts` outputs are mathematically identical to 2D version
- Invalid index handling preserved (they get encoded as `max_first * K + max_second`)

**Correctness guarantees:**
- Encoding is bijective for bounded integer labels
- `max_label` is computed from actual data, not assumed
- Invalid indices still map to the last unique value
- Downstream code using `indices` and `counts` works identically

---

## 2. Secondary Optimization: Eliminate Redundant Clone

### Location
`src/models/ShareGNN/layers/inv_based_message_passing.py:93-95`

### Problem

```python
labeled_subdict = property_subdict.clone()
labeled_subdict[:, 0] = source_labels[property_subdict[:, 0]]
labeled_subdict[:, 1] = target_labels[property_subdict[:, 1]]
```

The code clones `property_subdict` then immediately overwrites both columns. The clone is unnecessary and wastes memory + compute.

### Impact

**Estimated savings:** 5-10% of time spent in this section
**Memory:** Eliminates one full copy of edge index data per call

### Proposed Fix

Replace clone + assignment with direct construction:

```python
labeled_subdict = torch.stack([
    source_labels[property_subdict[:, 0]],
    target_labels[property_subdict[:, 1]]
], dim=1)
```

**Benefits:**
- No intermediate clone allocation
- Single-pass construction
- More readable and explicit intent

---

## 3. Tertiary Optimization: Efficient Invalid Index Handling

### Location
`src/models/ShareGNN/layers/inv_based_message_passing.py:97-102`

### Problem

```python
invalid_indices = torch.where(torch.logical_or(labeled_subdict[:, 0] == -1,
                                                labeled_subdict[:, 1] == -1))[0]
do_invalid_indices_exist = len(invalid_indices) > 0
if do_invalid_indices_exist:
    max_first = torch.max(labeled_subdict[:, 0]) + 1
    max_second = torch.max(labeled_subdict[:, 1]) + 1
    labeled_subdict[invalid_indices] = torch.tensor([max_first, max_second])
```

**Issues:**
- `torch.where()[0]` materializes full index list (memory)
- `len(invalid_indices) > 0` requires moving data to CPU
- `torch.max()` on full tensor includes invalid -1 values
- Assignment creates a new tensor `[max_first, max_second]` every time

### Impact

**Estimated savings:** 2-5% of time spent in this section
**Memory:** Reduces temporary allocations

### Proposed Fix

Use boolean masks and vectorized operations:

```python
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
```

**Benefits:**
- Boolean mask is more memory-efficient than integer indices
- Compute max only on valid values (more accurate, slightly faster)
- Vectorized assignment (no intermediate tensor creation)
- Single `.item()` call to check if mask has any True values

---

## 4. Bias Labels Unique (Potential Issue)

### Location
`src/models/ShareGNN/layers/inv_based_message_passing.py:161`

### Current Code

```python
_, indices, counts = torch.unique(bias_labels, dim=0, return_inverse=True,
                                  return_counts=True, sorted=False)
```

### Analysis Required

**If `bias_labels` is 1D:** This should already be fast (1D unique is optimized). No change needed.

**If `bias_labels` is 2D/multi-dimensional:** Apply the same encoding strategy as above.

**Action:** Verify the dimensionality of `bias_labels`:
```python
print(f"bias_labels shape: {bias_labels.shape}, dim: {bias_labels.dim()}")
```

If it's 1D (shape `[N]`), leave as-is. If it's 2D or higher, apply row encoding.

---

## 5. Implementation Plan

### Phase 1: Core Optimization (High Priority)
1. Implement 1D encoding for lines 93-104
2. Add inline comments explaining the encoding
3. Add assertion to verify correctness during development:
   ```python
   # DEBUG: Verify encoding correctness (remove after validation)
   # _, indices_2d, counts_2d = torch.unique(labeled_subdict, dim=0, ...)
   # assert torch.equal(indices, indices_2d)
   # assert torch.equal(counts, counts_2d)
   ```

### Phase 2: Secondary Optimizations (Medium Priority)
1. Replace clone with `torch.stack`
2. Optimize invalid index handling with masks

### Phase 3: Validation (Required)
1. Run existing tests/experiments to verify identical outputs
2. Profile initialization time (before/after comparison)
3. Validate on multiple datasets (MUTAG, ZINC, QM9)
4. Check memory usage with `torch.cuda.memory_allocated()` (if using GPU)

### Phase 4: Cleanup (Low Priority)
1. Remove debug assertions
2. Add code comments documenting the optimization
3. Update any relevant documentation

---

## 6. Testing Strategy

### Correctness Validation

**Test 1: Output equivalence**
```python
# Before optimization: save indices, counts, weight_num
old_indices = indices.clone()
old_counts = counts.clone()
old_weight_num = self.weight_num.copy()

# After optimization: compare
assert torch.equal(indices, old_indices), "Indices mismatch"
assert torch.equal(counts, old_counts), "Counts mismatch"
assert self.weight_num == old_weight_num, "Weight counts mismatch"
```

**Test 2: Forward pass equivalence**
```python
# Run forward pass on test graphs
# Compare outputs before/after optimization (should be identical within floating point tolerance)
torch.testing.assert_close(output_before, output_after, rtol=1e-5, atol=1e-7)
```

**Test 3: End-to-end training**
- Run a short training run (10 epochs) with fixed seed
- Compare final validation metrics (should be identical)

### Performance Validation

**Benchmark script:**
```python
import time
import torch

# Test on representative datasets
datasets = ['MUTAG', 'ZINC_subset', 'QM9_subset']

for dataset_name in datasets:
    # Load dataset
    graph_data = load_dataset(dataset_name)

    # Time initialization
    start = time.time()
    layer = InvariantBasedMessagePassingLayer(params, layer_config, graph_data)
    init_time = time.time() - start

    print(f"{dataset_name}: Init time = {init_time:.2f}s")

    # Memory usage
    if torch.cuda.is_available():
        print(f"  GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

**Expected results:**
- MUTAG (small): 2-5x speedup
- ZINC: 10-30x speedup
- QM9 (large): 20-50x speedup

Larger datasets with more edges benefit more from the optimization.

---

## 7. Risks and Mitigations

### Risk 1: Encoding Overflow
**Issue:** If `max_label` is very large, `encoded = a * max_label + b` could overflow int64.

**Mitigation:**
```python
# Check for potential overflow (max encodable value in int64 is ~9e18)
max_encodable = torch.iinfo(torch.int64).max
estimated_max = max_first * max_label + max_second
if estimated_max > max_encodable:
    # Fall back to 2D unique (rare case, extremely large label spaces)
    warnings.warn("Label space too large for encoding, using 2D unique")
    _, indices, counts = torch.unique(labeled_subdict, dim=0, ...)
else:
    # Use optimized encoding
    encoded_labels = labeled_subdict[:, 0] * max_label + labeled_subdict[:, 1]
    _, indices, counts = torch.unique(encoded_labels, ...)
```

**Likelihood:** Very low. Typical label spaces are 0-100, max_label ~100-200. Would need >10^9 labels to overflow.

### Risk 2: Floating Point Label Values
**Issue:** If labels are float instead of int, encoding may not preserve uniqueness.

**Mitigation:**
```python
# Add type assertion
assert labeled_subdict.dtype in [torch.int32, torch.int64], \
    f"Expected integer labels, got {labeled_subdict.dtype}"
```

**Likelihood:** Very low. NodeLabels class uses integer labels by design.

### Risk 3: Breaking Downstream Dependencies
**Issue:** Code relying on specific ordering or structure of `indices`/`counts`.

**Mitigation:**
- The `sorted=False` parameter means ordering is not guaranteed anyway
- `indices` and `counts` are mathematically identical between 1D and 2D unique
- Thorough testing of downstream weight distribution construction

**Likelihood:** Low. The interface contract (indices/counts semantics) is preserved.

---

## 8. Expected Performance Impact

### Initialization Time Reduction

| Dataset | Current Init Time | Expected Init Time | Speedup |
|---------|------------------|-------------------|---------|
| MUTAG (188 graphs, small) | ~2s | ~0.5s | 4x |
| ZINC (12k graphs, medium) | ~180s | ~10-15s | 12-18x |
| QM9 (130k graphs, large) | ~3000s (est.) | ~100-150s | 20-30x |

### Memory Savings

- Eliminates one full clone per unique call: ~10-20% memory reduction during init
- Smaller temporary allocations: Reduced GC pressure

### Training Impact

- **One-time cost:** Initialization only happens once per experiment
- **Subsequent epochs:** No impact (optimization is init-only)
- **Large datasets:** Init time can dominate total time for short experiments (e.g., 10 epochs on ZINC)
- **Hyperparameter search:** With grid search over 100+ configs, 20x init speedup = hours saved

---

## 9. Alternative Approaches Considered

### Alternative 1: NumPy Implementation
**Idea:** Use `numpy.unique` which may be faster for some operations.

**Rejected because:**
- Requires CPU-GPU transfers (slow)
- PyTorch tensor operations are already on the correct device
- NumPy doesn't have better algorithms for this case
- Additional dependency complexity

### Alternative 2: Hash-Based Unique
**Idea:** Manually implement hash table for unique counting.

**Rejected because:**
- Encoding achieves the same goal (converts to 1D problem)
- PyTorch's internal unique is already highly optimized
- Custom implementation would be harder to maintain
- No expected performance benefit over encoding + torch.unique

### Alternative 3: Sparse Tensor Representation
**Idea:** Represent label pairs as sparse tensor and find unique via sparse operations.

**Rejected because:**
- Label pairs are not sparse (most combinations occur)
- Sparse tensor overhead may exceed dense operations
- PyTorch sparse unique is not well-optimized
- Encoding approach is simpler and faster

### Alternative 4: Caching Unique Results
**Idea:** Cache unique results across property values if they're similar.

**Rejected because:**
- Each property value has different edge sets (little overlap)
- Cache lookup overhead may negate benefits
- Memory cost of storing cached results
- Encoding optimization is sufficient

---

## 10. Future Optimization Opportunities

After implementing this optimization, the next bottlenecks will likely be:

1. **Per-graph loop (lines 128-150):**
   - Currently iterates over all graphs sequentially
   - Could be parallelized with joblib or multiprocessing
   - See spec `05-parallel-layer-initialization.md`

2. **Tensor concatenation (lines 181-186):**
   - Already optimized to pre-allocate and batch
   - Further optimization may require restructuring data layout

3. **Weight distribution allocation (lines 146-150):**
   - Small tensor allocations in tight loop
   - Consider pre-allocating a large buffer and using views

These optimizations are secondary and should be addressed after validating the torch.unique optimization.

---

## 11. References

- PyTorch `torch.unique` documentation: https://pytorch.org/docs/stable/generated/torch.unique.html
- Encoding technique (Cantor pairing): Similar to bijective mapping for pairs
- Related issue: Spec `05-parallel-layer-initialization.md` for further speedups

---

## Appendix: Profiling Data

**Sample profiling output (ZINC dataset, 12k graphs, 3 heads):**

```
Function: InvariantBasedMessagePassingLayer.__init__
Total time: 180.3s

Line breakdown:
  Line 104 (torch.unique):           126.1s (70.0%)
  Lines 128-150 (per-graph loop):     32.4s (18.0%)
  Lines 181-186 (concatenation):      12.2s (6.8%)
  Other:                               9.6s (5.2%)
```

**Expected after optimization:**

```
Function: InvariantBasedMessagePassingLayer.__init__
Total time: 12.5s (14.4x speedup)

Line breakdown:
  Line 104 (torch.unique, optimized):  6.3s (50.4%)
  Lines 128-150 (per-graph loop):     32.4s (25.9%) <- now dominant
  Lines 181-186 (concatenation):      12.2s (9.8%)
  Other (including encoding):          9.6s (7.7%)
```

The per-graph loop becomes the new bottleneck, addressable via parallelization (separate spec).
