# ShareGNN Optimization Spec

## 1. Per-Graph Forward Pass (No Batching)

**Location:** `src/framework/model_configuration.py:798-807` (training), `:872-881` (evaluation)

**Problem:** ShareGNN models cannot use batch processing. The framework falls back to a Python loop that calls `forward()` on each graph individually:

```python
# Line 804-807
for j, graph_id in enumerate(batch_ids, 0):
    outputs[j] = self.net(self.graph_data[graph_id], pos=graph_id)
```

For a batch of 32 graphs, this executes 32 sequential forward passes instead of 1 vectorized pass.

**Impact:** Estimated 10-50x slowdown vs. batched execution.

**Root cause:** ShareGNN layers use per-graph precomputed weight distributions indexed by `pos=graph_id`. The weight matrix `current_W` is reconstructed from scratch for each graph in `set_weights()`.

**Proposed fix:**
- Option A: Batch graphs into block-diagonal sparse matrices (like PyG's `Batch` for standard GNNs), precompute batched weight distributions.
- Option B: Vectorize the weight reconstruction to handle multiple graphs simultaneously using padded tensors.
- Option C (quickest): Pre-build all per-graph weight matrices once, store as a list, index in forward pass without reconstruction.

---

## 2. Dense Matrix Allocation in Forward Pass

**Location:** `src/models/ShareGNN/layers/inv_based_message_passing.py:237` (set_weights), `:254` (set_bias)

**Problem:** Every forward pass allocates a dense `(num_heads, N, N)` zero tensor and fills it via sparse indexing:

```python
# set_weights (called every forward pass)
self.current_W = torch.zeros((self.num_heads, input_size, input_size), dtype=self.precision).to(self.device)
# ... then fill via:
self.current_W[matrix_indices[0], matrix_indices[1], matrix_indices[2]] = torch.take(self.Param_W, weight_indices)
```

For a graph with N=100 nodes and 4 heads, this allocates 4 x 100 x 100 = 40,000 floats every forward call.

**Impact:** Unnecessary memory allocation + GC pressure. For large graphs (N>500), the dense matrix becomes prohibitive.

**Proposed fix:**
- Use `torch.sparse_coo_tensor` or `torch.sparse_csr_tensor` to represent the weight matrix.
- Replace `einsum('hij,jf->hif', current_W, node_representation)` with sparse matrix multiplication `torch.sparse.mm()`.
- Pre-allocate the dense tensor once and zero it in-place with `self.current_W.zero_()` instead of reallocating.

---

## 3. Quadratic Weight Distribution Construction

**Location:** `src/models/ShareGNN/layers/inv_based_message_passing.py:117-141`

**Problem:** The weight distribution is built via nested Python loops over all graphs, with `torch.cat` inside the loop:

```python
for idx in range(len(graph_data)):           # O(num_graphs)
    # per head, per property...
    new_weight_distribution = torch.zeros(...)
    if self.weight_distribution[idx] is None:
        self.weight_distribution[idx] = new_weight_distribution.detach().clone()
    else:
        self.weight_distribution[idx] = torch.cat(
            (self.weight_distribution[idx], new_weight_distribution), dim=0)
```

Each `torch.cat` copies the existing tensor + appends, giving O(n^2) total allocation for n elements.

**Impact:** For ZINC (12k graphs), initialization takes significant time. The same pattern appears in bias distribution (lines 151-161) with O(graphs x features) complexity.

**Proposed fix:**
- Pre-compute total sizes per graph, allocate once, fill via slicing.
- Replace the list-of-tensors + cat pattern with a single pre-allocated tensor and offset tracking.
- Example:
  ```python
  # Compute total entries per graph upfront
  total_entries = sum(len(valid_indices) for idx in range(len(graph_data)))
  all_distributions = torch.zeros((total_entries, 4), dtype=torch.int64)
  offset = 0
  for idx in range(len(graph_data)):
      n = len(valid_indices)
      all_distributions[offset:offset+n] = new_distribution
      offset += n
  ```

---

## 4. Repeated Config Lookups in Forward Pass

**Location:** `src/models/ShareGNN/layers/inv_based_message_passing.py:304-308`

**Problem:** Dictionary lookups on `self.para.run_config.config` happen inside `forward()`:

```python
if self.para.run_config.config.get('degree_matrix', False):
    ...
elif self.para.run_config.config.get('use_in_degrees', False):
    ...
```

These are constant after initialization but checked every forward call.

**Proposed fix:** Cache as boolean attributes in `__init__`:
```python
self.use_degree_matrix = self.para.run_config.config.get('degree_matrix', False)
self.use_in_degrees = self.para.run_config.config.get('use_in_degrees', False)
```

---

## 5. Unnecessary Clone in Label Relabeling

**Location:** `src/models/ShareGNN/layers/inv_based_message_passing.py:84`

**Problem:** `labeled_subdict = property_subdict.detach().clone()` creates a full copy. The clone is then modified in-place (lines 85-86), but the original `property_subdict` is not used afterward in many code paths.

**Proposed fix:** Use the tensor directly if it's safe to modify, or use `clone()` without `detach()` (the tensor is already not in the computation graph at this point).

---

## 6. Pooling Layer Inefficiency

**Location:** `src/models/ShareGNN/layers/inv_based_pooling.py:54-68`

**Problem:** Triple nested loop: O(num_heads x num_graphs x output_dim) with tensor allocations inside.

Same quadratic `torch.cat` pattern as message passing (section 3).

**Proposed fix:** Same pre-allocation strategy as section 3. Additionally, `torch.unique()` (line 57) is called per head but could be cached if label sets don't change between heads.

---

## 7. Pooling Output Shape

**Location:** `src/models/ShareGNN/layers/inv_based_pooling.py:175`

**Problem:** `node_representation.flatten().unsqueeze(0)` produces shape `(1, heads*output_dim*features)`. This flattens all structure into a single vector, losing head/feature organization.

**Impact:** Subsequent linear layers must know the exact flattened size. Makes the architecture brittle to changes in num_heads or output_dim.

**Proposed fix:** Output `(1, heads * output_dim, features)` or `(1, output_dim, heads * features)` to preserve some structure for downstream layers.

---

## 8. ShareGNN Preprocessing eval() Security Risk

**Location:** `src/models/ShareGNN/preprocessing/preprocessing.py:153`

**Problem:** `eval(experiment_configuration['subgraphs'][layer['id']])` executes arbitrary Python code from config.

**Proposed fix:** Use `ast.literal_eval()` or `json.loads()`.

---

## 9. Empty Positional Encoding File

**Location:** `src/models/ShareGNN/layers/inv_based_positional_encoding.py`

**Status:** File is completely empty (0 lines of implementation). Layer type `INVARIANT_BASED_POSITIONAL_ENCODING` is registered in `LayerTypes` enum but has no implementation.

**Action:** Either implement or remove from the enum to avoid confusion.

---

## 10. Expensive Label Computation for ShareGNN

**Location:** `src/simplegnn/datasets/utils/node_labeling.py`

**Problem:** Several label types used by ShareGNN are computationally expensive:

| Label Type | Complexity | Typical Time (1000 graphs) |
|-----------|-----------|---------------------------|
| trivial/index/degree | O(nodes + edges) | < 1 second |
| wl_k (depth 3) | O(k x graph_size) | 10-60 seconds |
| simple_cycles (len 6) | O(exponential) | 30+ seconds |
| induced_cycles (len 6) | O(exponential) | 60+ seconds |
| cliques | NP-hard | 2-5 minutes |
| subgraph | NP-complete | hours for large patterns |

Labels are cached to disk after first computation, but there's no parallel computation across graphs.

**Proposed fix:**
- Parallelize label computation across graphs using `joblib` or `multiprocessing`.
- Add progress bars (currently only prints every 10% for cycles).
- Consider approximate algorithms for expensive label types.

---

## Priority Order for Implementation

1. **Section 4** - Cache config lookups (5 min, immediate forward pass speedup)
2. **Section 3** - Pre-allocate weight distributions (1-2 hours, fixes initialization bottleneck)
3. **Section 2** - Sparse or pre-allocated weight matrices (2-4 hours, major forward pass improvement)
4. **Section 1** - Batch processing support (1-2 days, largest performance win but most complex)
5. **Section 8** - Replace eval() (15 min, security fix)
6. **Section 10** - Parallelize label computation (2-4 hours, preprocessing speedup)
