# 07 — Batched ShareGNN forward (CPU/GPU)

## Problem

ShareGNN training processes one graph per forward call
(`model_configuration.py`: `outputs[j] = self.net(self.graph_data[graph_id], pos=graph_id)`).
Every graph pays the full Python/dispatch/kernel-launch overhead, and on GPU each
graph is a separate tiny kernel plus (potentially) a separate host→device transfer.
Goal: process **all graphs of a batch jointly** — one feature tensor, one device
transfer, one sparse matmul per layer — and compare against the unbatched path on
MUTAG (CPU here; the code path is device-agnostic and works on CUDA unchanged).

## Key insight

The invariant message-passing layer already computes, per graph, a sparse block-COO
matrix of shape `(H·N, N)` (head folded into rows) multiplied with the node features
`(N, F)`. For a batch, the graphs are independent, so the batch computation is a
**block-diagonal** sparse matrix over the concatenated nodes:

- node offsets `o_g = cumsum(N_g)`, `N_total = Σ N_g`
- entry `(h, i, j)` of graph `g` → row `(o_g + i)·H + h`, col `o_g + j`
- one `torch.sparse.mm((N_total·H, N_total), (N_total, F))` → view `(N_total, H, F)`

Row layout is **node-major** (`row = i·H + h`, not `h·N + i`) so that concatenated
per-graph blocks have strictly increasing row ranges → the batch COO is sorted and
can be built with `is_coalesced=True` (no runtime coalesce). This requires a second
precomputed ordering of the per-graph index buffers (built once at init, next to the
existing per-graph `_fwd_indices`).

Output equivalence with the per-graph path:
`(H,N,F).permute(1,2,0).flatten(1)` ≡ `(N,H,F).permute(0,2,1).flatten(1)` → `(N, F·H)`.

## Changes

1. **`inv_based_message_passing.py`**
   - `_build_forward_index_structures`: additionally precompute node-major buffers
     `_fwd_b_rows` (`i·H + h`), `_fwd_b_cols` (`j`), `_fwd_b_param_idx`, sorted per
     graph by `(row, col)`; same per-graph slices `_fwd_slices`. Store bias slices
     as ints (`_bias_slices`).
   - `forward`: if `pos` is a sequence (list/ndarray/tensor) → `_forward_batched`.
   - `_forward_batched(x, positions)` dispatches to one of two implementations
     (`_use_dense_batch`), which produce bit-identical results:
     - `_batched_sparse_messages`: concatenate per-graph index slices, add
       row/col offsets via `repeat_interleave`, gather `Param_W`
       (differentiable), one block-diagonal sparse mm.
     - `_batched_dense_messages`: pad the batch to `(B, H, N_max, N_max)` and
       run one `matmul` against the padded `(B, 1, N_max, F)` features, then
       gather the real nodes back (padding rows stay zero and are dropped, so
       nothing leaks between graphs). Reuses the raw `weight_distribution`
       rows — no extra index buffers.
   - Bias (shared): scatter `Param_b` into `(N_total, H, F)` via advanced
     indexing (same uniqueness assumption as per-graph `set_bias`).
   - Duplicated graph ids in one batch are fine: each occurrence gets its own node
     block (needed for `training_data_sampling: random/balanced`).

   **Choosing the batched implementation** (`share_gnn_forward.mode`): unlike the
   per-graph case the winner depends on the *device*, so `auto` resolves per
   forward. Measured on one NCI1 conv layer (64 graphs, ≤ 93 nodes, fwd+bwd):

   | implementation       | CPU     | iGPU    |
   |----------------------|---------|---------|
   | per-graph dense      | 11.3 ms | 15.8 ms |
   | batched sparse       | 40.8 ms |  6.2 ms |
   | batched padded-dense |  3.1 ms |  7.6 ms |

   `auto` → padded dense on CPU, block-diagonal sparse on CUDA. On CPU the padded
   matmul goes through BLAS and beats both alternatives; on GPU the sparse mm wins
   because it does not pay for the padding, and the kernel-launch overhead that
   batching removes is already gone. Since dense padding costs `B·H·N_max²`
   elements, `auto` falls back to sparse above `dense_batch_max_nodes` (256) or
   `dense_batch_max_bytes` (128 MB) of padded weights. An explicit
   `mode: sparse|dense` overrides the choice.

2. **`inv_based_pooling.py`**
   - `_forward_batched`: gather `Param_W[weight_distribution[rows]]` → `(N_total, H)`,
     `contrib = W.unsqueeze(-1) * x.unsqueeze(1)` → `(N_total, H, F)`,
     `out.index_add_(0, graph_of_node, contrib)` → `(B, H, F)` → `+ Param_b` →
     flatten `(B, H·F)`. Matches per-graph `(H,F).flatten().unsqueeze(0)` layout.

3. **`reshape.py`**: batched `pos` + default shape `[-1]` → reshape to `(B, -1)`.

4. **`model_configuration.py`**
   - Config opt-in: `share_gnn_forward: { batched: true }`. (Since 2026-07: batched
     is the default; opt out with `share_gnn_forward: { batched: false }`.)
   - `_assemble_share_gnn_batch(batch_ids)`: concatenate `x` rows of all batch
     graphs (single `.to(device)` if data is not already on the device → the whole
     batch is loaded to the GPU together), return `(SimpleNamespace(x=...), positions)`.
   - `train_graph_task` / `evaluate_graph_task`: batched branch replacing the
     per-graph loop (eval uses `eval_batch_size` chunks).

Standard layers in between (linear `aggr_features`, `layer_norm`, activation,
dropout) are row-wise → batch-transparent, no changes.

## Verification

- `tests/test_batched_share_gnn.py` (uses `share_gnn_setup` fixture, real MUTAG):
  - forward equivalence: per-graph vs batched outputs `allclose` (double
    precision), for each `share_gnn_forward.mode` (`auto`/`dense`/`sparse`), i.e.
    both batched implementations
  - gradient equivalence: `Param_W.grad` matches, again for all three modes
  - duplicate graph ids in a batch
  - full config-driven pipeline through `ModelConfiguration`
- `tests/test_speed_share_gnn_nci1.py` (marker `speed`, excluded from the default
  `pytest tests` run): benchmarks one NCI1 configuration per (device, mode) and
  asserts that batching does not change the resulting accuracies.
- `examples/share_gnn_basic/benchmark_batched.py`: times per-graph vs batched
  training epochs on MUTAG with the full example model (cycles + betweenness heads),
  `--device cpu|cuda`, reports speedup and checks loss agreement.

## Results — MUTAG (2026-07-14)

MUTAG, full `examples/share_gnn_basic` model (cycles + betweenness heads, double
precision), 188 graphs, batch size 32, timed epochs. GPU = Radeon 890M iGPU
(ROCm 6.4, `venv-rocm/`, `HSA_OVERRIDE_GFX_VERSION=11.0.0`). Both modes run the
same batches with the same seed; max epoch-loss difference 2.2e-16, max
final-output difference < 5e-13 — identical training trajectories.

| device | mode      | train epoch | inference (188 graphs) |
|--------|-----------|-------------|------------------------|
| CPU    | unbatched |  79.9 ms    | 28.7 ms                |
| CPU    | batched   |  52.2 ms  (**1.53x**) | 14.5 ms (**1.98x**) |
| iGPU   | unbatched | 300.0 ms    | 73.6 ms                |
| iGPU   | batched   |  42.8 ms  (**7.01x**) | 12.1 ms (**6.07x**) |

Takeaways:
- Unbatched GPU is *slower* than unbatched CPU — per-graph kernel-launch
  overhead dominates tiny MUTAG graphs. Batching is what makes the GPU pay off.
- Batch size barely matters on GPU; the win comes from eliminating per-graph
  dispatch, not from batch size tuning.
- CPU epoch times are noisy on this machine (occasional outlier epochs inflate
  the mean by 2-4x); the numbers above are from a run without outliers.
  The earlier sparse-only batched implementation reached 1.11x (train) / 1.48x
  (inference) on CPU — the padded-dense batch raises that to ~1.5x / ~2x.

Reproduce: `python examples/share_gnn_basic/benchmark_batched.py --device cpu`
and `HSA_OVERRIDE_GFX_VERSION=11.0.0 venv-rocm/bin/python
examples/share_gnn_basic/benchmark_batched.py --device cuda`.

## Results on NCI1 (2026-07-14)

One full configuration (`tests/fixtures/share_gnn_nci1/`, 4110 graphs, batch size
64, 10 epochs, double precision, split 0, seed 42) run through the production
path (`FrameworkMain.run_configuration` → `ModelConfiguration.train_configuration`)
for each device/mode combination. Measured by `tests/test_speed_share_gnn_nci1.py`
(mean of epochs 2-10; epoch 1 is warmup). Same iGPU as above.

| device | mode      | s/epoch | speedup   | train acc | val acc |
|--------|-----------|---------|-----------|-----------|---------|
| CPU    | unbatched | 2.19    | 1.00x     | 81.51     | 78.10   |
| CPU    | batched   | 1.43    | **1.53x** | 81.51     | 78.10   |
| iGPU   | unbatched | 3.88    | 1.00x     | 81.51     | 78.10   |
| iGPU   | batched   | **1.31**| **2.97x** | 81.51     | 78.10   |

- **Accuracies are identical** across all four runs (0.0000 percentage points
  difference, same final loss 23.3750): the batched forward changes only the
  summation order, not the trained model. Note the standard NCI1 splits are a
  10-fold CV over train/validation only — their `test` lists are empty, so there
  is no test accuracy to compare.
- **On the GPU batching is what makes the device worth using**: unbatched CUDA
  (3.88 s/epoch) is *slower* than unbatched CPU (2.19 s/epoch) — 4110 tiny
  per-graph kernel launches per epoch — while batched CUDA is the fastest
  configuration overall.
- Batched CPU used to be a *pessimization* here (3.52 s/epoch, 0.59x) when the
  batched path was sparse-only: NCI1's graphs are small (≤ 111 nodes), so the
  per-graph path runs the dense BLAS matmul, which beat the block-diagonal sparse
  mm. The padded-dense batch (see `_use_dense_batch` above) fixes that — batching
  is now a win on both devices.

Takeaway for users: enable `share_gnn_forward: {batched: true}` on both CPU and
GPU; `mode: auto` picks the right batched implementation per device.

Reproduce: `pytest tests/test_speed_share_gnn_nci1.py -m speed -s`, or as a script
(the ROCm venv has no pytest):
`HSA_OVERRIDE_GFX_VERSION=11.0.0 venv-rocm/bin/python tests/test_speed_share_gnn_nci1.py --device cuda`

## Follow-up: ZINC profiling (2026-07-14)

Profiling ZINC (12000 graphs, `network_ZINC.yml`: 20-head conv + 140-head
aggregation, double precision, batch 128, CPU) uncovered two problems that
MUTAG/NCI1 were too small to show:

1. **16 GB peak RSS at model init, ~10 GB retained** — in both batched and
   unbatched mode. `_build_forward_index_structures` eagerly built *two*
   sorted index orderings (head-major `_fwd_indices` for the per-graph sparse
   forward, node-major `_fwd_b_*` for the batched sparse forward) over the
   ~92M-row conv `weight_distribution`. On CPU neither is used: the per-graph
   path picks `forward_mode: dense` (graphs < 256 nodes) and the batched path
   picks padded dense. Fix: both orderings are now built lazily on first use
   (`_ensure_per_graph_fwd_indices` / `_ensure_batched_sparse_indices`), so
   runs that never enter a sparse path never pay for them (also removes two
   ~92M-element argsorts from init).

2. **Aggregation `_forward_batched` materialized (total_nodes, H, F)** — the
   segment-reduction contributions tensor is ~1.3 GB per 512-graph eval chunk
   with H=140, F=100, and its mul-backward dominated training. Fix: one padded
   `bmm` — `(B, H, N_max) @ (B, N_max, F)`, padded rows contribute zero — is
   ~7x faster and peaks at ~50 MB instead of ~1.4 GB. The segment reduction is
   kept as fallback when `B * N_max > 4 * total_nodes` (pathologically skewed
   graph sizes in one batch).

ZINC, 2 epochs, batch 128, CPU, same seed (losses identical to ~1e-13):

| mode      | epoch time before | after     | peak RSS before | after   |
|-----------|-------------------|-----------|-----------------|---------|
| unbatched | 40.6-55.2 s       | ~58 s     | 16.0 GB         | 5.8 GB  |
| batched   | 17.5-18.0 s       | **5.7 s** | 16.0 GB         | 6.1 GB  |

(unbatched epoch time is unchanged — its forward paths were not touched; the
batched CPU forward is now ~10x faster per epoch than unbatched.)

The remaining ~5 GB retained memory is the raw `weight_distribution`
(92M x 4 int64 = 3.0 GB) and `bias_distribution` (1.8 GB) of the conv layer —
shrinking those (int32 columns, or slicing per batch from disk) is a possible
next step.

## Non-goals

- Node-classification / mixed models (batching only for graph-level ShareGNN tasks).
