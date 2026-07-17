# 12 — Layer Drawing: Bug Fixes and Runtime Improvements

Analysis of the ShareGNN layer visualization code (`draw()` in
`inv_based_message_passing.py` / `inv_based_pooling.py`, `graph_drawing.py`,
callers in `experiments/base_paper/src/latex_plots.py` and `plot_zinc.py`).

## Findings

### Bugs — InvariantBasedMessagePassingLayer.draw (the actively used path)

1. **Wrong node-label indexing in the second circle-layout block**
   (`inv_based_message_passing.py:1507-1508`): uses
   `node_labels['primary'].node_labels[graph_id][node]`, but `NodeLabels.node_labels`
   is a flat 1-D tensor over all dataset nodes. Indexing `[graph_id]` returns a 0-d
   tensor; `[node]` then raises IndexError. The first block (line 1365) uses the
   correct flat indexing `slices['x'][graph_id] + node`. Also contains a stray debug
   `print`.
2. **`filter_weights` bound computation**:
   - `sorted_weights[int(len * percentage) - 1]` → index `-1` (the *maximum*) when
     the product truncates to 0, inverting the filter.
   - `sorted_weights[int(len * (1 - percentage))]` → IndexError when the index
     equals `len`.
   - Neither `percentage` nor `absolute` present → `lower_bound_weight` unbound
     (NameError).
   - `upper_weights + lower_weights` **double-counts** weights that satisfy both
     bounds (whenever bounds overlap, e.g. `absolute > len/2` or many equal
     weights) → drawn edge widths/colors are 2x wrong.
3. **Division by zero / NaN colors** when all graph weights are equal
   (`weight_max - weight_min == 0`), when `weight_max_abs == 0`, and analogously
   for the bias normalization.
4. **Per-head bias ignored**: node bias coloring indexes the flat `Param_b` with the
   bias *label* only, i.e. always head 0 / feature 0's block. The correct parameter
   for head `h` is `Param_b[_b_off[h, 0] + label]`.
5. **Infinite loop risk in the circle layout walk** when the root node is not found
   (`pos[None]`) or a node has no unvisited next neighbor (non-ring graphs).

### Bugs — InvariantBasedAggregationLayer.draw (broken since before the factorization)

6. `self.graph_data.graphs[graph_id]` — `GraphDataset` has no `.graphs` attribute
   (that is legacy `GraphData`) → AttributeError, always.
7. Treats `_weight_index_matrix()` output — an `(N, num_heads)` Param_W index
   matrix — as legacy rows `[head, out_dim, node_idx, param_idx]`: `[:, 3]` picks
   head 3's indices (or crashes when `num_heads < 4`), and the node loop reads
   meaningless columns.
8. Same wrong `node_labels[graph_id][node]` indexing as (1), in three places.
9. `get_bias()` unconditionally reads `Param_b`, which only exists when
   `self.bias` is true → AttributeError for bias-free configs.

### Bugs — callers

10. `latex_plots.py:290` (single-graph branch): all filterings draw onto the same
    axis `axs[2+i]`, while titles are set on `axs[1+column_for_invariants+i*len(filtering)+j]`
    → filtered plots overwrite each other and appear under wrong/blank titles.
11. `latex_plots.py:354`: `axs.axis('off')` on a numpy array of Axes → AttributeError.
12. `latex_plots.py:419` (`plot_specific_graphs_from_db`): `bbox_inches`/`backend`
    kwargs are passed to `Path.joinpath` instead of `plt.savefig` → TypeError.

### Runtime

- Layout code (circle walk + kawai/shell/bfs/graphviz dispatch + pos-file
  load/save) is duplicated 4x across the two draw methods.
- Edge assembly is a Python loop over all assembled rows with per-element tensor
  `.item()` calls — O(K) interpreter overhead; vectorizable with one head mask.
- `sorted(set(...))` on floats → `np.unique`.
- Pooling `get_weights`/`get_bias` build Python lists via per-element `.item()` —
  replace with a single `.detach().cpu().numpy()` (only used by draw).

## Plan

1. **`graph_drawing.py`**: add shared helpers
   `load_positions(pos_path)`, `save_positions(pos, pos_path)`,
   `compute_positions(graph, draw_type, root_node=None)` (safe circle walk with
   dead-end guard), and `resolve_positions(...)` that chains load → compute → save.
   Add `filter_weight_bounds(graph_weights, filter_weights)` implementing the
   corrected keep-mask (clamped indices, OR of the two bounds, ValueError on
   unknown keys).
2. **Message-passing draw**: use the helpers; fix label indexing; guard zero
   ranges; per-head bias via `_b_off`; vectorized edge construction; drop debug
   print and unused `matrix_indices`.
3. **Pooling draw**: rewrite the weight part for the factored storage — node
   param indices `self._weight_index_matrix(node_range)[:, head]`, node
   color/size from the head's pooling weight; `out_dimension` parameter kept for
   API compatibility but ignored (no out-dim in the factored layout); bias
   guarded by `self.bias`; shared layout helpers; flat label indexing.
4. **latex_plots.py**: fix (10)-(12) minimally.
5. **Tests**: new `tests/test_layer_drawing.py` (Agg backend) — smoke-draws both
   layers of the MUTAG fixture model (graph_only, full weights, absolute filter,
   each head), pos-file save/load roundtrip, and unit tests for
   `filter_weight_bounds` / `compute_positions` edge cases.

## Implemented (2026-07-17)

All of the above landed:

- `graph_drawing.py`: `load_positions` / `save_positions` / `compute_positions` /
  `resolve_positions` and `filter_weight_bounds` (shared by both layers; removes
  the 4x duplicated layout block).
- `InvariantBasedMessagePassingLayer.draw` rewritten: fixes (1)-(5), vectorized
  edge assembly, per-head bias parameters via `_b_off[head, 0] + bias_label`.
  New `_primary_node_labels()` helper dedupes the NodeLabels/Tensor branches.
  `get_weights`/`get_bias` return `.numpy()` directly (no numpy-2 deprecation).
- `InvariantBasedAggregationLayer.draw` rewritten for the factored storage:
  per-node pooling weight of the drawn head colors/sizes the nodes
  (`_weight_index_matrix(node_range)[:, head]`); gains `pos_path` caching;
  `out_dimension` kept but ignored (no out-dim axis in the factored layout).
  Also removed an accidental `pandas.core.array_algos` internal import.
- `latex_plots.py`: fixed the single-graph filter-axis indexing, the
  `axs.axis('off')` crash, and the `savefig`-kwargs-into-`joinpath` bug.
- `tests/test_layer_drawing.py`: 21 tests, all passing.

Benchmark (MUTAG fixture, 50 graphs, kawai): edge assembly old per-row tensor
loop 38 ms vs vectorized 3 ms (~13x; grows with graph size since the old loop
paid per-row `.item()` costs). Remaining draw cost is ~57% kamada-kawai layout
(scipy, cached across calls via `pos_path`) and ~40% matplotlib FancyArrowPatch
creation — both outside this code.
