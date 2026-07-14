# Spec 08: Invariant-Based Message Passing Layer Optimization (Fable Optimization Plan)

**Status:** Phases 0–4c IMPLEMENTED (2026-07-14); Phase 4d (batching) remains open
**Created:** 2026-07-14 (multi-agent analysis: 5 Explorer agents + 1 Architect agent)

## Implementation Notes (deviations from the design below)

- **Forward mode defaults to `auto`, not `sparse`.** End-to-end measurement on
  MUTAG showed the sparse path 3.4x SLOWER than dense+matmul (N≈18: sparse
  construction overhead dominates); a microbenchmark shows parity around
  N≈128–512 and sparse winning ~6x at N=1024. `share_gnn_forward.mode: auto`
  picks `sparse` when the dataset's largest graph has ≥ 256 nodes, `dense`
  otherwise; both can be forced explicitly.
- **`Module.type(precision)` → `Module.to(precision)`** in
  `model.py get_model_layer` for the invariant layers: `.type()` casts ALL
  buffers including the new int64 index buffers (breaking `torch.take`);
  `.to(dtype)` only casts floating-point tensors. This was discovered by the
  weight-update integration test, exactly as planned.
- **Phase 4a (per-graph x view cache) intentionally skipped:** the Phase-0
  check confirmed the installed PyG `InMemoryDataset.get` already memoizes
  `separate()` results in `_data_list`; repeat access costs one shallow copy.
- **Phase 4c partially applied:** `zero_grad(set_to_none=True)` done; the
  per-batch `loss.item()` accumulation was NOT deferred to per-epoch because
  `evaluate_results` consumes the running loss per batch — restructuring that
  belongs with Phase 4d.
- **Bias `torch.unique(dim=0)` kept verbatim** (bitwise-safety); the win came
  from hoisting it out of the per-head loop via a per-description cache.
- Bitwise equality of the vectorized assembly vs. the old per-graph loops was
  verified on real MUTAG for the message-passing weight distribution, bias
  distribution, slices, and the pooling weight distribution.
- Test coverage lives in `tests/test_fable_optimization.py` (T3–T9) plus the
  pre-existing `tests/test_share_gnn_mutag_integration.py` gates.
**Primary files:**
- `src/simplegnn/models/ShareGNN/layers/inv_based_message_passing.py`
- `src/simplegnn/models/ShareGNN/layers/inv_based_pooling.py`
- `src/simplegnn/models/ShareGNN/layers/inv_based.py`
- `src/simplegnn/framework/model_configuration.py`
- `src/simplegnn/datasets/graph_dataset.py`
- `src/simplegnn/models/model.py`

**Relation to other specs:**
- Builds on `06-invariant-layer-torch-unique-optimization.md` (implemented: 1D-encoded unique, mask-based invalid handling).
- Orthogonal to `05-parallel-layer-initialization.md` (designed, **not** implemented — no joblib in the layer; can compose with Phase 3 later, but vectorization + coarse caching should land first).
- Supersedes the stale forward-pass items in `01-sharegnn-optimizations.md` / `next_steps.md` (see status corrections at the end).

---

## 0. Motivation and Problem Statement

The `InvariantBasedMessagePassingLayer` rebuilds a dense `(num_heads, N, N)` weight matrix for every graph on every forward pass via `torch.take(Param_W, ...)` + scatter-assignment, and its `__init__` runs expensive per-graph Python loops that are re-paid for every fold × run × grid-config combination.

Two **training-correctness bugs** motivated this analysis and are root-caused below:

1. **On GPU, layer weights are not updated by backprop** (CPU training works).
2. **A previous attempt to precompute the weight matrices in a preprocessing step and copy them to the device froze training even on CPU** (weights not updated).

Both are fully explained and fixed by Phases 1 and 2 of this plan.

---

## 1. Root-Cause Analyses (verified against the live tree)

### 1.1 GPU bug: `.to(device)` demotes `nn.Parameter` to a plain tensor

`inv_based_message_passing.py:402,405` and `inv_based_pooling.py:68`:

```python
self.Param_b = self.init_weights(np.sum(self.bias_num), init_type='convolution_bias').to(self.device)
self.Param_W = self.init_weights(np.sum(self.weight_num), init_type='convolution').to(self.device)
```

`init_weights` returns a genuine `nn.Parameter` (`inv_based_message_passing.py:590`). But `.to(device)` on a Parameter returns a **plain non-leaf Tensor whenever the device actually changes**. `nn.Module.__setattr__` only registers values in `module._parameters` when the assigned object *is* an `nn.Parameter`, so on CUDA the attribute is stored as an ordinary attribute — invisible to `net.parameters()` and therefore to the optimizer created at `model_configuration.py:607`:

```python
self.optimizer = opt(self.net.parameters(), lr=..., weight_decay=...)
```

On CPU, `.to('cpu')` is an identity operation returning the *same* Parameter object, so registration survives — **which is exactly why CPU trains and GPU does not**. Gradients may still be computed into the graph on GPU, but no optimizer step ever applies them.

Asymmetry worth noting: `inv_based_pooling.py:64` (`Param_W`, no `.to()`) is already correct — on GPU the pooling layer trains its weights but not its bias, while the message-passing layer trains neither.

The correct pattern already exists in the repo: `src/simplegnn/models/layers/nn_standard/linear.py:33-42` creates Parameters **without** `.to()`, relying on the whole-model move `self.net.to(self.device)` at `model_configuration.py:539` to relocate registered parameters.

**Latent second bug (currently masked):** the graph data is never moved to the device — both `self.graph_data.to(self.device)` calls are commented out (`model_configuration.py:369` and `:479`). Today this stays consistent because the broken parameters also remain on CPU. Once the parameter fix lands, CPU inputs meeting CUDA parameters raise device-mismatch `RuntimeError`s. Both fixes must land together (Phase 1).

### 1.2 Precomputed value matrices detach the autograd chain

The live autograd chain in `set_weights` / `forward` is:

```
Param_W (leaf, requires_grad)
  → torch.take(Param_W, weight_indices)        # inv_based_message_passing.py:639 — differentiable gather
  → scatter-assign into current_W               # in-place index_put, tracked by autograd
  → einsum('hij,jf->hif', current_W, x)         # :771
  → ... → loss
```

The **only** differentiable link from the loss back to `Param_W` is the gather on line 639. If weight matrices are instead precomputed (in preprocessing or at init) with their *values* baked in, they are detached leaf constants with no `grad_fn` pointing at `Param_W`: `einsum` produces a gradient *for the matrix*, but there is no edge back to `Param_W`, so `Param_W.grad` stays `None` and the optimizer never moves it. This is a graph-topology problem, **independent of device** — hence the observed failure on CPU as well.

Reusing a stored `current_W` across iterations also fails: after `loss.backward()` the graph is freed (`retain_graph=False` default), so the next iteration either raises "backward through the graph a second time" or, if detached, silently feeds stale values.

**Consequence (design invariant for every phase):** only the **index structure** (which cells get which parameter) may be precomputed. The value gather `Param_W[indices]` must execute inside every forward pass.

The regression gate already exists: `tests/test_share_gnn_mutag_integration.py:81-142` (`test_invariant_layer_weights_are_updated_by_training`) asserts `Param_W.grad` is non-None/non-zero after `backward()` and that `Param_W` changes after `optimizer.step()`.

### 1.3 Why sparse tensors fit naturally

`weight_distribution` rows are `(head, i, j, param_idx)` (`inv_based_message_passing.py:346-351`) with per-graph offsets in `weight_distribution_slices` (`:388`) — this **is** a COO representation; the dense build in `set_weights` (`:632-639`) is just a scatter of it. Verified properties:

- **No duplicate `(head, i, j)` cells occur**: property generators (`models/ShareGNN/preprocessing/properties.py`) emit each ordered node pair exactly once per property bucket (a pair has exactly one shortest-path distance), and head replicas write disjoint channel slices. Therefore sparse coalesce-sum semantics ≡ dense overwrite semantics. (Weight *sharing* — many cells gathering the same `param_idx` — is expected and works identically in both.)
- torch 2.10 is installed (`venv/`, currently the CPU build; torch is unpinned in `pyproject.toml`). `torch.sparse.mm` supports autograd w.r.t. the sparse tensor's `values` on CPU and CUDA — exactly the gradient path needed.
- The degree-matrix branches in forward (`:766-769`, using `self.D` and `self.in_edges`) are **dead code**: those attributes are never assigned anywhere in the repo; enabling `degree_matrix`/`use_in_degrees` in a config would raise `AttributeError` today.

### 1.4 Init assembly cost structure

- The per-graph weight loop (`:329-351`) and the bias triple loop (`:363-374`, graphs × head-replicas × in_features) are the remaining hotspots after spec 06. **Both run in full even on a disk-cache hit** — the existing cache only skips the (now cheap) `torch.unique` call.
- Layer init is re-paid `n_folds × num_runs × num_grid_configs` times (cartesian run loop at `core.py:293`, model constructed fresh in `model_configuration.py:536`), although the produced `weight_distribution` iterates the **full** dataset (`:329`) and is therefore identical across folds and runs.
- **Latent cache-hit correctness bug:** `do_invalid_indices_exist` is initialized `False` (`:270`) and only set on a cache **miss** (`:285`). On a cache hit with default `rule_occurrence_threshold == 1` and no upper threshold, the filtering block (`:310-325`) is skipped entirely, so the invalid (`-1`-label) bucket becomes a real trained weight and `num_weights` is off by one — cache-hit runs and cache-miss runs produce **different models**. The flag must be persisted in the cache.
- The fine cache key (`get_cache_path`, `:433-440`) correctly omits thresholds because it stores the *pre-filter* `(indices, counts)` and filtering is re-applied after load. A coarse cache of *post-filter* results (Phase 3d) **must** include thresholds in its key.

---

## 2. Phase Plan Overview

| Phase | Content | Type | Depends on |
|---|---|---|---|
| 0 | Baseline benchmarks + test scaffolding | infra | — |
| 1 | Correctness: GPU parameter registration, buffer/device strategy, graph-data device move, cache-hit invalid-bucket fix, precision-default unification | bug fixes | — |
| 2 | Forward pass: per-graph index precompute + sparse COO forward (dense fallback flag), dead-branch removal, micro-fixes | perf | 1 |
| 3 | Init assembly: vectorize weight/bias/pooling distribution loops, coarse per-layer disk cache | perf | 1c |
| 4 | Training loop: per-graph input caching, eval realloc fix, sync reduction; (4d, stretch) block-diagonal batched forward | perf | 1, 2 |

**Ordering rationale.** Phase 1 first: nothing later can be validated on GPU while `Param_W` is silently detached from the optimizer. Phase 2 before Phase 3: the sparse forward defines which precomputed index structures init must produce. Phase 3's coarse cache must land after the 1c invalid-bucket fix so the bug is not baked into cached artifacts. Phase 4 last: batching sits on top of a correct device story (P1) and the sparse block mechanism (P2).

**Dense-intermediate vs straight-to-sparse: go straight to sparse, keep dense as a fallback flag.** A separate "dense with cached indices" phase is not worth it — ~80% of its work (per-graph folded index tensors built once at init) is exactly the precompute the sparse path needs. Phase 2 delivers both paths from one precompute: sparse as the default, dense (today's scatter + `matmul`) behind a config flag as (a) an equivalence oracle in tests and (b) a fallback for tiny graphs or sparse-op regressions.

---

## Phase 0 — Baseline and Scaffolding (0.5 day)

No production code changes.

1. Record baselines on the MUTAG integration fixture (`tests/fixtures/`) and, if available locally, ZINC: (a) layer `__init__` wall time, (b) per-epoch train time, (c) `forward_step_time` totals, (d) final validation accuracy for a fixed seed. Record the numbers in this spec.
2. Add a pytest GPU marker/fixture: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")`.
3. Confirm the installed PyG `InMemoryDataset._data_list` memoization behavior (affects Phase 4a sizing — `__getitem__` may already memoize `separate()` results and only pay a shallow copy per access).

---

## Phase 1 — Correctness Fixes (0.5–1 day)

### 1a. Register parameters properly (fixes the GPU-frozen-weights bug)

- `inv_based_message_passing.py:405`:
  `self.Param_W = self.init_weights(...).to(self.device)` → `self.Param_W = self.init_weights(int(np.sum(self.weight_num)), init_type='convolution')`
- `inv_based_message_passing.py:402`: same for `Param_b`.
- `inv_based_pooling.py:68`: same for `Param_b` (`:64` `Param_W` is already correct).

Devices are handled solely by `self.net.to(self.device)` (`model_configuration.py:539`) — the `linear.py:33-42` pattern.

### 1b. Device strategy: buffers + graph-data move

**Decision:** parameters are created on CPU and moved by `net.to(device)`; large derived index tensors become **non-persistent registered buffers** so `net.to` moves them too; graph data is moved **once**, immediately after `net.to`.

1. **Buffers.** Replace plain assignments at `inv_based_message_passing.py:389/:397` (and the pooling `weight_distribution`, `inv_based_pooling.py:50`) with:
   ```python
   self.register_buffer('weight_distribution', torch.cat(merged_weight_distributions, dim=0), persistent=False)
   ```
   `persistent=False` keeps these large, deterministic, dataset-derived tensors out of `state_dict` (avoids bloating checkpoints in `results/<db>/Models/`; old checkpoints stay loadable).
   **Implementation trap:** `InvariantBasedLayer.__init__` (`inv_based.py:24-32`) pre-assigns `self.weight_distribution = None` etc.; `register_buffer` raises "attribute already exists". Remove the `None` pre-assignments (keep the docstring).
2. **Slices stay on CPU.** `weight_distribution_slices` / `bias_distribution_slices` remain plain CPU tensors — they are used in Python slice expressions, and 0-dim CUDA tensors there force a host sync per forward. Additionally materialize `self._num_nodes_list: list[int]` at init to eliminate the per-forward `num_nodes[pos].item()` syncs (`:631`, `:648`).
3. **Graph data to device.** Add `GraphDataset.to(self, device)` in `graph_dataset.py`: `self._data = self._data.to(device)`, invalidate PyG's separate-cache (`self._data_list = None`), keep `self.slices` and a copy of `num_nodes` on CPU. Call it in `ModelConfiguration` right after `self.net.to(self.device)` (`model_configuration.py:539`), replacing the commented-out calls at `:369`/`:479`. Ensure `graph_data.y` lands on device too (used at `:1291/:1294`; currently only the one-hot branch moves it).
4. **Residual-device audit.** In `set_weights`/`set_bias`, derive the allocation device from `self.Param_W.device` rather than the `self.device` string, then run the GPU integration test and fix any remaining mismatch.

**Risk:** none for checkpoints (these keys were never in `state_dict` before). `persistent=False` means a fresh layer always rebuilds/loads distributions before `load_state_dict` — already the case.

### 1c. Cache-hit invalid-bucket fix

- `_save_cached_indices` (`:492`): store `'do_invalid_indices_exist': bool(...)` in the cache dict (pass it in as a parameter).
- `_load_cached_indices` (`:454`): return `(indices, counts, do_invalid_indices_exist)`; if the key is absent (old cache file), **raise** → treated as a cache miss, recomputed and re-saved. No key change; old files self-heal.
- Update call sites at `:277` and `:308`.

### 1d. Precision default unification

`model.py:215` defaults `config.get('precision', 'float')` while `model.py:306` and `model_configuration.py:1367` default `'double'`. A config without an explicit `precision` key generates random-variation noise in double while the model runs in float. Unify all reads to default `'float'` via a single helper (e.g. `RunConfig.precision_dtype`) so the default lives in one place. **Risk:** silently changes numerics for configs relying on the implicit double default in the node-task path — call out in the changelog; verify shipped configs set precision explicitly.

### Phase 1 verification

- Existing CPU tests stay green with identical fixed-seed accuracy trajectory (Phase 1 is CPU-behavior-preserving except 1c/1d).
- **New GPU variant of `test_invariant_layer_weights_are_updated_by_training`**: `net.to('cuda')`, `graph_data.to('cuda')`; assert `Param_W.is_cuda`, grads non-zero, post-step change, and that `sum(p.numel() for p in net.parameters())` matches the CPU build (catches de-registration).
- New buffer test: after `net.to('cuda')`, `layer.weight_distribution.is_cuda` is True and `'weight_distribution' not in net.state_dict()`.
- New invalid-bucket regression test (T5 in the matrix).

---

## Phase 2 — Forward Pass: Index Precompute + Sparse COO (1–2 days)

### 2a. Sparse design decision

**Fold heads into a single block-structured `(H·N, N)` COO matrix per graph; one `torch.sparse.mm` per forward.**
Rejected alternative — per-head loop of `(N, N)` sparse mms: identical nnz and index memory (2×nnz int64 either way) but H Python iterations and H kernel launches per layer per graph; on MUTAG-sized graphs (N≈18) launch overhead dominates. Autograd support is identical.

`weight_distribution` **layout stays untouched** — `draw()` (`:902-1025`), `get_graph_weights` (`:788`), and the Phase 3 coarse cache keep consuming the `(head, i, j, param_idx)` rows. The forward-only structures below are *derived* from it.

### 2b. One-time index precompute (end of `__init__`, after `:397`)

```python
# Derived, forward-only structures (registered buffers, persistent=False)
rows = wd[:, 0] * n_of_graph + wd[:, 1]          # fold head into the row index (graph-local N)
cols = wd[:, 2]
order = <per-graph stable sort by (rows, cols)>  # makes each graph's block coalesced
self.register_buffer('_fwd_indices', torch.stack([rows, cols])[:, order], persistent=False)
self.register_buffer('_fwd_param_idx', wd[order, 3].contiguous(), persistent=False)
self._fwd_slices = <CPU int list, per-graph offsets — same partition as weight_distribution_slices>
```

The per-graph sort makes `is_coalesced=True` construction legal, skipping runtime `.coalesce()` (safe because no duplicate cells exist — §1.3; add a debug/test-only assertion that the `(row·N + col)` encoding is strictly increasing per graph).

Bias analog: precompute `_bias_scatter_flat` (flattened `(H, N, F)` positions) and `_bias_param_idx`; add a fast path when every head has bias (full coverage) → `current_B = Param_b[_bias_param_idx[s:e]].view(H, n, F)` with no zero-fill.

### 2c. Forward rewrite

```python
def _sparse_W(self, pos):
    s, e = self._fwd_slices[pos], self._fwd_slices[pos + 1]
    n = self._num_nodes_list[pos]
    values = self.Param_W[self._fwd_param_idx[s:e]]      # differentiable gather — EVERY forward (see §1.2)
    return torch.sparse_coo_tensor(self._fwd_indices[:, s:e], values,
                                   (self.num_heads * n, n), is_coalesced=True)

# in forward(), replacing set_weights + einsum (:765-771):
W = self._sparse_W(pos)
out = torch.sparse.mm(W, node_representation).view(self.num_heads, n, -1)
```

Config flag `share_gnn_forward: {mode: sparse|dense}` (default `sparse`). Dense mode keeps today's scatter but driven by the same precomputed structures, and replaces `torch.einsum('hij,jf->hif', ...)` with `torch.matmul(self.current_W, node_representation)` (broadcasted bmm — same result, less dispatch overhead).

Output contract preserved exactly: `(N, H·F)` with `(N, F, H)` flatten order via the unchanged `.permute(1, 2, 0).flatten(start_dim=1)`; single-graph `pos` semantics unchanged.

Micro-fixes folded into this phase:
- **Delete the dead degree branches** (`:766-769`); raise `ValueError` in `__init__` if `degree_matrix`/`use_in_degrees` is set in the config (they crash today anyway; reviving is a feature request — §6). If ever revived on sparse: `diag(D) @ W @ diag(D)` ≡ scaling COO values by `D[i]·D[j]`.
- Gate the `time.time()` timing (`:763`, `:779`; pooling `:157/:164`) behind `config.get('profile_layers', False)` — pure per-call overhead and misleading on CUDA without synchronize.
- Remove the dead pre-allocation in pooling `set_weights` (`inv_based_pooling.py:119` allocates zeros immediately overwritten by the `torch.take` rebinding at `:121`).

### Risks

- Sparse-mm numerics differ at ULP level → equivalence tests use `torch.allclose` with dtype-dependent tolerances (float: rtol 1e-5 / atol 1e-6; double: 1e-10).
- `is_coalesced=True` with unsorted indices is undefined behavior → the precompute **must** sort per graph block (assertion above).
- Graphs with empty weight distribution (`len == 0`, handled at `:634` today): nnz-0 sparse tensors are valid; add a unit test.

### Phase 2 verification

- **Dense-vs-sparse equivalence** (new): same seed, both modes, all 188 MUTAG fixture graphs — `allclose` on outputs *and* on `Param_W.grad` after one backward.
- **CPU-vs-GPU equivalence** (new, GPU-marked): same model/seed, forward + backward on both devices.
- Existing weight-update test (CPU + Phase-1 GPU variant) must pass in sparse mode — the autograd gate.
- Timing benchmark: MUTAG (expect neutral-to-modest — tiny graphs) and ZINC (expect 1.5–4× forward speedup on CPU; more on GPU: no per-forward `(H,N,N)` zeros + scatter, fewer launches).

---

## Phase 3 — Init Assembly: Vectorization + Coarse Per-Layer Cache (2–3 days)

### 3a. Vectorize the per-graph weight loop (`:329-351`)

Replace `for idx in range(len(graph_data))` with one vectorized pass per `(head_id, property_key)`:

```python
row_graph = torch.repeat_interleave(torch.arange(G), property_subdict_slices.diff())   # graph id per row
keep = valid_indices_bool if filtering else slice(None)                                 # global bool mask
w_idx = indices[keep]
p_idx = property_subdict[keep] - graph_data.slices['x'][row_graph[keep]].unsqueeze(1)   # global → local node ids
# head replicas by broadcasting: col0 = current_head_id + arange(n)[:, None],
#                                col3 = w_idx + weight_offset + arange(n)[:, None] * num_weights
```

There is no genuine cross-graph data dependency — graphs are only a partitioning of rows.

**Hard requirement: bitwise-identical output.** The merged `weight_distribution` row order today is per graph, chunks appended in `(head_id → property_key → head-replica n)` order. Reproduce it exactly: tag each vectorized block with a monotone chunk-sequence id, one stable `argsort` over `(graph_id, chunk_seq)` at merge time; per-graph slices via `bincount(graph_id).cumsum(0)`. Temporary test builds old and new paths on the MUTAG fixture and asserts `torch.equal` (keep the old path in the test file, not production).

### 3b. Vectorize the bias triple loop (`:363-374`)

Loop order today: `idx → n → feature_id → node` (rows per graph = heads·F·N_g). Rebuild with broadcasting honoring the same nesting order. Replace the 2-D `torch.unique(bias_labels, dim=0, ...)` at `:362` with the spec-06 1D encoding (bias label rows are bounded ints), and hoist it out of the head loop when the bias-label description is unchanged between heads. Note: the bias block sits inside the `for head_id` loop and uses the current head's `bias_labels`/`n_heads_per_label[head_id]` — this per-head placement is **correct** (per-head bias configs), preserve semantics verbatim; flag anything accidental-looking with a `# NOTE`, don't change behavior in a perf spec.

### 3c. Vectorize the pooling init loop (`inv_based_pooling.py:54-60`)

The inner per-graph loop is a no-op partition of a contiguous global assignment (`indices + h_num * n_node_labels[head_id] + base_offset` for all rows at once). Collapse to a broadcast over the head offsets. Bitwise-equality test against the old output.

### 3d. Coarse per-layer disk cache

Today the full assembly (both loops) re-runs on every model construction — `n_folds × num_runs × num_grid_configs` times (`core.py:293`) — although the result is fold- and run-invariant (§1.4).

- **Cached artifact (one `.pt` per layer):** `weight_distribution`, `weight_distribution_slices`, `bias_distribution`, `bias_distribution_slices`, `weight_num`, `weight_offset`, `bias_num`, `b_head_offset`, `weight_offset_description(_text)`, per-(head,property) `do_invalid_indices_exist` flags, `format_version: 1`.
- **Key (MD5 of sorted JSON, same style as `get_cache_path` `:433-440`):** dataset name + `len(graph_data)`, `layer_id`, `in_features`, per head: source/target/bias label strings, property string, full `valid_property_values` list, `head.num`, `head.bias`; plus `rule_occurrence_threshold`, `rule_occurrence_upper_threshold`, `format_version`. (Thresholds and valid_property_values must be in this key — the coarse cache stores *post-filter* results, unlike the fine cache.)
- **Placement:** checked at the top of `__init__` after metadata extraction; hit → skip the entire head/property loop; miss → run the vectorized assembly, save. Files `layerdist_<hash>.pt` in `data/caches/` alongside the fine-cache files, with a `.json` metadata sibling (as `_save_cached_indices` does).
- **Invalidation:** key-hash based; bump `format_version` on layout changes. Missing/corrupt file → fall through to the fine-cache path (unchanged), which still amortizes `torch.unique` on coarse misses.
- **Invalid-bucket interplay:** the coarse artifact bakes in filtered results, so the 1c fix must be live before this lands; store the flags in the coarse dict as well for auditability.
- The Phase 2 `_fwd_*` structures are **recomputed** from the cached `weight_distribution` (cheap; keeps the cache format independent of the forward mode).

### Phase 3 verification

- Bitwise-equality tests old-vs-new assembly (weights, bias, pooling) on the MUTAG fixture.
- Coarse-cache round-trip: cold build → save → fresh layer warm → `torch.equal` on every cached tensor; then the weight-update test on the warm-cache layer (proves the invalid-bucket bug cannot reappear via the coarse layer).
- Existing `test_cache*.py` suites updated for the 3-tuple fine-cache API and kept green.
- Benchmark: layer `__init__` cold and warm, MUTAG + ZINC. Expected: 10–100× on the loop portion for large datasets; warm coarse-cache init ≈ I/O-bound.

---

## Phase 4 — Training Loop and Batching (1 day for 4a–c; 4d stretch: 3–5 days)

### 4a. Per-graph input caching

`train_graph_task` (`model_configuration.py:1282`) and `evaluate_graph_task` (`:1356`) call `self.net(self.graph_data[graph_id], pos=graph_id)` — a PyG `separate()`/Data reconstruction per graph per epoch, though the ShareGNN path consumes only `.x` (layers read everything else from `self.graph_data` + `pos`). After the Phase-1 device move, cache per-graph `x` views once (`self._graph_x = [x[s:e] ...]`, zero-copy views) and pass a trivial `x`-attribute holder in the invariant-layer branch (`model.forward` reads `batch_data.x`). Classical-GNN path untouched. (First check the Phase-0 PyG memoization finding to size the win.)

### 4b. `evaluate_graph_task` output reallocation (`:1341-1344`)

`outputs = torch.zeros(...)` sits **inside** the batch sizing loop — reallocated per batch, only the last allocation survives. Hoist to a single `torch.zeros((len(graph_ids), num_classes), ...)`. Also note the ShareGNN branch never uses the `loader`/`batches` it builds — remove the dead setup.

### 4c. Sync and overhead reduction

- `optimizer.zero_grad()` (`:1269`, `:1362`) → `zero_grad(set_to_none=True)`.
- Per-batch `.item()` syncs (`loss.item()` `:1317`; accuracy math in `evaluate_results` `:847`; validation criterion `:906`): accumulate on-device, sync once per epoch at write-out. Cheap on CPU today; stall points on GPU.

### 4d. Block-diagonal batched forward (stretch — split into spec 09 if it grows)

The mechanism for `next_steps.md` items #19/#24: `weight_distribution` cols 1–2 are **graph-local** (offset subtracted at `:344`), so a batch `g1..gk` folds into one sparse `(H·ΣN, ΣN)` block-diagonal matrix by re-adding per-graph node offsets to `_fwd_indices` (a single add of a `repeat_interleave`d offset vector; blocks stay disjoint and per-block order is preserved — no re-sort). Per batch: one gather from `Param_W`, one `sparse.mm`, one batched bias gather; the pooling layer becomes an `index_add_`/segment reduction over the batch vector. Requires threading a `batch` vector through `GraphModel.forward` instead of `pos` and turning the per-graph loop (`:1280-1283`) into a per-batch call. Keep the single-graph path as the reference implementation.

### Phase 4 verification

- 4a–c: existing integration test green; fixed-seed accuracy trajectory unchanged (pure plumbing).
- 4d: batched-vs-loop equivalence (outputs + `Param_W.grad` allclose over a MUTAG batch); per-epoch benchmarks CPU + GPU.

---

## 5. What NOT To Do (and why)

1. **Do not precompute per-graph value matrices.** See §1.2 — the per-forward gather is the only differentiable link to `Param_W`; baking values in freezes training silently on any device. Only index structure may be precomputed. `test_invariant_layer_weights_are_updated_by_training` exists precisely to catch this.
2. **Do not revive the degree-matrix paths as part of this work.** `self.D`/`self.in_edges` are never assigned; the branches are unreachable-or-crashing. Delete with an init-time config guard; reimplementation is a feature request.
3. **Do not apply `torch.compile` before Phase 4d.** Per-graph forwards with variable `N` cause per-shape recompiles and graph breaks on sparse construction; payoff is negative until batching amortizes shapes.
4. **Do not switch to CSR.** COO with precomputed-coalesced indices suffices; autograd for values through CSR mm is less uniformly supported across devices/dtypes in torch 2.x.
5. **Do not change the `weight_distribution` row layout** `(head, i, j, param_idx)` — `draw()`, `get_graph_weights`, the coarse cache format, and the bitwise-equality test strategy all depend on it.

---

## 6. Effort and Expected Impact

| Phase | Effort | Expected impact |
|---|---|---|
| 0 | 0.5 d | Baselines; none |
| 1 | 0.5–1 d | **Enables GPU training at all** (params currently frozen on CUDA); fixes cache-hit model divergence; no CPU perf change |
| 2 | 1–2 d | Forward: ~neutral on MUTAG-size graphs (dense fallback available); est. 1.5–4× on ZINC-size CPU, more on GPU (no per-forward `(H,N,N)` zeros + scatter, fewer launches); removes per-forward syncs |
| 3 | 2–3 d | Cold init: est. 10–100× on the loop portion for large datasets; warm coarse cache: init amortized to ~I/O across the fold × run × config product |
| 4a–c | 1 d | 5–15% per-epoch wall time; eval allocation bug fixed |
| 4d | 3–5 d | est. 2–10× per-epoch on GPU (batch-level parallelism); modest on CPU |

---

## 7. Test / Verification Matrix

| ID | Test | Type | Phase gate | Device |
|---|---|---|---|---|
| T1 | `test_share_gnn_trains_and_evaluates_on_mutag` (existing) | integration | every phase | CPU |
| T2 | `test_invariant_layer_weights_are_updated_by_training` (existing) | integration | 1, 2, 3 | CPU |
| T3 | **New** GPU variant of T2 (params on CUDA, grads non-zero, step changes weights, `p.is_cuda`, param count matches CPU build) | integration | 1, 2 | GPU (skipif) |
| T4 | **New** buffer/device test: `net.to('cuda')` moves `weight_distribution`; not in `state_dict` | unit | 1 | GPU |
| T5 | **New** invalid-bucket persistence: dataset with `-1` labels; `num_weights` and distribution identical cold vs warm fine cache and warm coarse cache | unit | 1c, 3d | CPU |
| T6 | **New** dense-vs-sparse equivalence: outputs + `Param_W.grad` allclose, all fixture graphs incl. empty-distribution graph | unit | 2 | CPU |
| T7 | **New** CPU-vs-GPU equivalence (sparse mode), forward + grads | unit | 2 | GPU |
| T8 | **New** bitwise equality old-vs-vectorized assembly (weights, bias, pooling) | unit | 3a–c | CPU |
| T9 | **New** coarse-cache round-trip: `torch.equal` on all cached tensors; T2 on warm-cache model | unit | 3d | CPU |
| T10 | Existing `test_cache*.py`, `test_torch_unique_optimization.py` (updated for 3-tuple fine-cache API) | unit | 1c, 3 | CPU |
| T11 | **New** batched-vs-loop equivalence (outputs + grads) | unit | 4d | CPU+GPU |
| T12 | Timing benchmarks: layer init (cold/warm), per-epoch train, MUTAG + ZINC, vs Phase-0 baseline | bench | 2, 3, 4 | both |

Fixed-seed accuracy trajectories on the MUTAG fixture are the cross-phase behavioral regression check for phases claiming behavior preservation (1a/1b, 2 dense mode, 3, 4a–c).

---

## 8. Status Corrections to Existing Specs (applied to next_steps.md)

- Item 4 (cache config lookups in forward): **already implemented** at `inv_based_message_passing.py:419-420`.
- Items 6/7 (pre-allocate weight/bias distributions vs quadratic `torch.cat`): **already implemented** at `:384-397`; the pattern only remains in the pooling init loop (`inv_based_pooling.py:54-60`) — covered by Phase 3c here.
- Items 9/13 (per-forward dense allocation): still valid; real sites are `set_weights` `:632` and `set_bias` `:649` (old line refs stale) — covered by Phase 2 here.
- Items 19/24 (batch processing): real per-graph loops are `model_configuration.py:1280-1282` (train) and `:1355-1356` (eval); the enabling fact is that `weight_distribution` node indices are graph-local (`:344`) — covered by Phase 4d here.
