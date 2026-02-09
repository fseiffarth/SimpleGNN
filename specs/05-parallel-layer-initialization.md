# Parallel Layer Initialization Spec

**Created:** 2026-02-09
**Priority:** HIGH
**Effort:** 2-4 hours
**Impact:** 5-20x faster initialization for multi-head ShareGNN models

---

## Problem Statement

ShareGNN layer initialization processes heads sequentially in a single-threaded Python loop, making initialization time scale linearly with the number of heads. For models with many heads (8-64 heads common in practice), this creates a significant bottleneck during model setup.

### Location

**File:** `src/models/ShareGNN/layers/inv_based_message_passing.py:78-177`

**Critical loop:**
```python
# Line 78-177: Sequential processing of all heads
for head_id, head in enumerate(self.layer.layer_heads):
    # ... ~100 lines of processing per head ...
    # Line 128-150: Process all graphs for this head
    for idx in range(len(graph_data)):
        # Build weight distributions for current head and graph
        # ...
```

### Current Performance Characteristics

For a typical ShareGNN model with:
- 32 heads across multiple layers
- 10,000 graphs in dataset
- Each head requiring ~5-30 seconds of processing

**Sequential initialization time:** 32 heads × 15s avg = **8 minutes**

With parallelization across 8 cores: **1-2 minutes** (5-8x speedup)

---

## Root Cause Analysis

1. **Sequential head processing:** Each head's weight/bias distribution is computed one at a time
2. **Independent computation:** Each head's processing is completely independent - no shared state or data dependencies
3. **CPU-bound workload:** Processing involves:
   - PyTorch tensor operations (CPU)
   - Dictionary lookups
   - Index manipulation
   - Memory allocation
4. **Already has progress printing:** Large datasets (>5000 graphs) show progress, indicating long runtimes

---

## Proposed Solution

### Implementation Strategy

Parallelize the outer loop over `head_id, head` using `joblib.Parallel` with configurable worker count.

### Key Design Points

1. **Parallelization scope:**
   - Parallelize over the outer `for head_id, head in enumerate(self.layer.layer_heads)` loop
   - Keep inner graph loop (`for idx in range(len(graph_data))`) sequential within each worker
   - Each worker processes one complete head (all graphs for that head)

2. **Configuration parameter:**
   - Add `parallel_loading` parameter to model YAML configuration
   - Default: `-1` (use all available CPU cores)
   - Values: `-1` (all cores), `1` (sequential), `2+` (specific number of workers)
   - Location: Model config layer definition

3. **Data handling:**
   - Extract head processing into a separate function `_process_head()`
   - Return per-head results (weight_distribution_chunks, bias_distribution_chunks, metadata)
   - Merge results after parallel execution

4. **Thread safety:**
   - No shared state modification during parallel execution
   - Each worker operates on independent head data
   - Results accumulated via return values, not mutation

---

## Detailed Implementation Plan

### Step 1: Extract Head Processing Function

Create a static/module-level function to process a single head:

```python
def _process_single_head(
    head_id: int,
    head,
    layer,
    graph_data: GraphDataset,
    para: Parameters,
    layer_id: int,
    in_features: int,
    num_heads: int,
    device,
    precision,
    # ... other necessary parameters
) -> Tuple[List, List, Dict]:
    """
    Process a single head to build weight and bias distributions.

    Returns:
        - weight_distribution_chunks: List of tensors per graph
        - bias_distribution_chunks: List of tensors per graph
        - metadata: Dict with weight_num, bias_num, skips info
    """
    # Move lines 78-177 logic here
    # Return computed distributions and metadata instead of mutating self
    pass
```

### Step 2: Add Configuration Parameter

**In YAML model configuration:**
```yaml
- layer_type: 'invariant_based_convolution'
  parallel_loading: -1  # -1 = all cores, 1 = sequential, N = N workers
  # ... other layer config ...
```

**In Parameters class (`src/framework/utils/parameters.py`):**
```python
# Add to layer configuration parsing
self.parallel_loading = layer_config.get('parallel_loading', -1)
```

### Step 3: Implement Parallel Execution

**In `InvariantBasedMessagePassingLayer.__init__`:**

```python
# Line ~78: Replace sequential loop with parallel execution
if self.para.parallel_loading == 1:
    # Sequential execution (original behavior)
    results = [
        _process_single_head(head_id, head, ...)
        for head_id, head in enumerate(self.layer.layer_heads)
    ]
else:
    # Parallel execution
    from joblib import Parallel, delayed
    n_jobs = self.para.parallel_loading if self.para.parallel_loading > 0 else -1

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_process_single_head)(head_id, head, ...)
        for head_id, head in enumerate(self.layer.layer_heads)
    )
```

### Step 4: Merge Results

```python
# Accumulate results from all heads
weight_distribution_chunks = [[] for _ in range(len(graph_data))]
bias_distribution_chunks = [[] for _ in range(len(graph_data))]

for head_id, (weight_chunks, bias_chunks, metadata) in enumerate(results):
    # Merge per-head results into global structures
    for idx in range(len(graph_data)):
        weight_distribution_chunks[idx].extend(weight_chunks[idx])
        if self.bias:
            bias_distribution_chunks[idx].extend(bias_chunks[idx])

    # Accumulate metadata
    self.weight_num.extend(metadata['weight_num'])
    self.bias_num.extend(metadata['bias_num'])
    # ...
```

### Step 5: Handle Progress Reporting

**Challenge:** Progress printing from parallel workers can interleave and clutter output.

**Solutions:**
1. **Disable progress in parallel mode:** Only show high-level "Processing N heads in parallel..."
2. **Aggregate progress:** Collect completion events and show overall progress bar
3. **Per-head summary:** Print "Head X/Y completed (Z seconds)" after each worker finishes

**Recommended:** Option 3 - post-completion summaries

```python
# In _process_single_head, return timing info
start_time = time.time()
# ... processing ...
elapsed = time.time() - start_time

# In main loop after Parallel returns:
for head_id, (weight_chunks, bias_chunks, metadata) in enumerate(results):
    print(f"Head {head_id+1}/{len(results)} completed in {metadata['elapsed']:.2f}s")
```

---

## Backward Compatibility

### Default Behavior
- Default `parallel_loading: -1` enables parallelization automatically
- Existing configs without this parameter will get the default (parallel)

### Opt-out
- Set `parallel_loading: 1` in model config to use sequential processing
- Useful for debugging or when memory constraints exist

---

## Expected Performance Gains

### Benchmark Estimates

| Scenario | Sequential Time | Parallel Time (8 cores) | Speedup |
|----------|----------------|------------------------|---------|
| 16 heads, 5K graphs | 4 min | 40 sec | 6x |
| 32 heads, 10K graphs | 16 min | 2.5 min | 6.4x |
| 64 heads, 20K graphs | 64 min | 10 min | 6.4x |

**Note:** Speedup is typically 0.7-0.8 × N_cores due to:
- Joblib overhead
- Non-parallelizable merge step
- Memory bandwidth contention

### Bottleneck Shifts

After this optimization, the bottleneck will shift to:
1. **Tensor concatenation** (line 181-186, 189-194) - Already optimized in current code
2. **Parameter initialization** (line 200) - Negligible time
3. **Graph label preprocessing** - Upstream (outside this layer)

---

## Testing & Validation

### Correctness Tests
1. **Determinism check:** Run same model with `parallel_loading: 1` and `parallel_loading: -1`, compare weight distributions
2. **Output validation:** Ensure forward pass produces identical results
3. **Edge cases:**
   - Single head (parallelization overhead should be minimal)
   - Empty graphs
   - Graphs with no valid properties

### Performance Tests
1. **Scaling test:** Measure initialization time vs. number of heads (4, 8, 16, 32, 64)
2. **Core utilization:** Profile CPU usage during parallel loading
3. **Memory usage:** Ensure memory doesn't spike (joblib workers may duplicate data)

---

## Risk Assessment

### Low Risk
- ✅ No changes to forward pass logic
- ✅ No changes to computed distributions (pure refactor)
- ✅ Easy rollback (set `parallel_loading: 1`)

### Potential Issues
- ⚠️ **Memory usage:** Each worker may duplicate read-only data (graph_data, labels)
  - **Mitigation:** Use `backend='loky'` with copy-on-write semantics
  - **Mitigation:** Monitor memory and reduce n_jobs if needed
- ⚠️ **Progress reporting:** Parallel output may interleave
  - **Mitigation:** Disable or aggregate progress updates
- ⚠️ **Determinism:** Random number generation if any (none found in current code)
  - **Mitigation:** Ensure any RNG is seeded per-head

---

## Alternative Approaches Considered

### 1. Parallelize Inner Graph Loop Instead
- **Pros:** Finer-grained parallelism
- **Cons:** Higher overhead (more joblib calls), shared state issues, progress reporting harder
- **Verdict:** ❌ Rejected - Outer loop parallelization is cleaner

### 2. Use Multiprocessing Instead of Joblib
- **Pros:** Standard library, no dependency
- **Cons:** More boilerplate, worse error handling, no automatic backend selection
- **Verdict:** ❌ Rejected - Joblib already used elsewhere in codebase

### 3. Use Threading Instead of Multiprocessing
- **Pros:** Lower memory overhead
- **Cons:** GIL contention - Python loops won't parallelize well
- **Verdict:** ❌ Rejected - CPU-bound workload needs true parallelism

---

## Success Criteria

### Functional Requirements
- ✅ Produces identical weight/bias distributions as sequential version
- ✅ Configurable via `parallel_loading` parameter in model YAML
- ✅ Works with existing model configurations (default enabled)

### Performance Requirements
- ✅ 4x+ speedup on 8-core machine for models with 16+ heads
- ✅ Initialization time < 5 minutes for 64-head models on 10K graphs

### Quality Requirements
- ✅ No new memory leaks
- ✅ Graceful handling of worker failures
- ✅ Clear error messages for configuration issues

---

## Related Specs

- **01-sharegnn-optimizations.md Section 3:** Weight distribution pre-allocation (orthogonal optimization)
- **02-training-pipeline-optimizations.md Section 8:** Joblib serialization (shared infrastructure)

---

## References

- Current implementation: `src/models/ShareGNN/layers/inv_based_message_passing.py:78-177`
- Similar pattern exists in: `src/models/ShareGNN/layers/inv_based_pooling.py:54-68`
- Joblib usage example: `src/framework/core.py:100-105`

---

## Next Steps After Implementation

1. Apply same pattern to `InvariantBasedAggregationLayer` (pooling layer)
2. Profile to identify next bottleneck (likely graph label preprocessing)
3. Consider GPU acceleration for tensor operations if CPU parallelism saturates
