# Node Classification / Regression: Fixes and Optimization

**Status: implemented** (2026-07-17)

The node-level task path of ShareGNN (`task: node_classification` /
`node_regression`) had been broken since the model forward signature was
refactored from `net(x, pos)` to `net(batch_data, pos=...)`: the node-task
training and evaluation code was never migrated, and the node-capable dataset
sources (Planetoid) no longer produced datasets that passed validation. This
spec documents the bugs that were fixed, the new benchmark, and the runtime
optimizations that went in alongside.

## Bugs fixed

1. **`train_node_task` was called with a missing argument**
   (`framework/model_configuration.py`). `train_configuration()` did not pass
   `random_variation_bool`, so every node-task run died with a `TypeError`.
   The parameter was removed entirely: random input variation is applied
   inside `GraphModel.forward` nowadays.

2. **Old forward signature in the node path.** `train_node_task` and
   `evaluate_node_task` called `self.net(self.graph_data[0].x, 0)`;
   `GraphModel.forward` reads `batch_data.x`, so a raw tensor crashed with
   `AttributeError`. Both now call `self.net(self.graph_data[0], pos=0)`.

3. **Random input variation leaked into evaluation.**
   `GraphModel.forward` added Gaussian noise regardless of `self.training`.
   It is now gated on training mode (this also removes eval noise for graph
   tasks that use `random_variation`).

4. **Planetoid datasets failed validation** (`datasets/graph_dataset.py`).
   The `process()` branch for `planetoid`/`Planetoid` produced data without
   `primary_node_labels`, `node_attributes`, `primary_edge_labels`,
   `edge_attributes` (all required by `validate_dataset_object`), and built a
   wrong `edge_index` slice (`[0, 2]` — `shape[0]` of the (2, E) tensor —
   instead of `[0, E]`). The branch now populates all required keys and
   correct slices. Primary node labels are set to zeros: **the class targets
   live in `y` and must not become structural labels**, otherwise the
   invariant layers' weight sharing would leak test labels.

5. **`num_classes` for node classification** was `sizes['num_node_labels']`,
   which conflates structural labels with targets. It is now computed from
   `torch.unique(data['y'])`.

6. **`node_regression` was not a recognized task** anywhere. It is now
   routed like `node_classification` for training/evaluation and like
   `graph_regression` for metrics (MAE/MAE-std), CSV headers, and best-epoch
   selection. `GraphDataset` derives the target dimension from `y`.

7. **`create_splits` did not exist** although
   `Preprocessing.create_split_file` called it (latent `NameError`). It is
   now implemented in `framework/utils/preprocessing.py`; for node tasks it
   splits node indices of the single graph and uses the dataset's
   train/val/test masks (Planetoid standard split) when available.

8. **`preprocessing_from_config` crashed for `base_labels` configs**: the
   first `layer_to_labels(layer)` call passed a single positional argument.

9. **New input transformation `normalize_rows`** for
   `input_features: {name: node_features}`: scales each node's feature vector
   to sum 1 (the standard bag-of-words normalization for citation networks).
   Without it, unnormalized neighborhood sums made Cora underperform badly
   (~42% vs ~69% validation accuracy).

## Runtime optimizations

- **Sparse index memoization in `InvariantBasedMessagePassingLayer`**
  (`_sparse_row_cache`): the per-graph sparse forward re-assembled the
  (head, i, j, param) rows and re-sorted them (argsort over all nonzeros) on
  *every* forward. The index structure depends only on the graph, not the
  weights, so it is now memoized for the last `pos`. Single-graph node tasks
  hit the cache on every forward; only the differentiable `Param_W` gather and
  the sparse matmul remain per step. Epoch time on Cora dropped ~2.4x
  (~5s → ~2.1s; ~1.1s with the final model).

- **Full-graph evaluation cache for node tasks**
  (`ModelConfiguration._node_eval_outputs`): validation and test evaluation
  of the same epoch now share one full-graph forward. The cache is
  invalidated after every optimizer step and on (re-)initialization of the
  model.

## Benchmark

`examples/node_classification/` runs ShareGNN on Cora (standard Planetoid
split, 140/500/1000 nodes): dropout → invariant convolution (2 heads, degree
labels `wl_0`, distances 0–2) → dropout → linear readout. Input features are
row-normalized bag-of-words plus masked one-hot train labels
(`one_hot_train_labels: True` — non-train rows are uniform, so no target
leakage). The convolution weights start as a uniform aggregator
(`constant 0.3`) and learn per-(degree, degree, distance) deviations.

Reference results (CPU, seeds 42–44, best-model evaluation): 67.0% mean
validation / 65.9% mean test accuracy with the original learned unnormalized
convolution. With the degree normalization added later (specs/16) the example
now ships a normalized frozen-convolution configuration reaching 78.3% mean
validation / 78.1% mean test accuracy — close to GCN (~81%).

Regression coverage: `tests/test_share_gnn_cora_node_classification.py`
(integration; downloads Cora on first run, ~15s cached).

## Possible follow-ups

- ~~Degree-normalized aggregation~~ — implemented, see
  specs/16-invariant-conv-degree-normalization.md (lifts Cora to ~78%
  validation).
- Batched multi-graph node tasks (currently node tasks assume graph 0).
- `node_regression` benchmark (the path is implemented and unit-covered via
  the shared code, but no example exists).
