# 09 — ZINC: Shrinking the Readout + Architecture Improvements

**Status**: Implemented (code + configs). **No training run yet** — the accuracy claims below are still
predictions. See [§6 Implementation](#6-implementation-what-was-built) for what exists and how to run it.
**Target config**: `experiments/base_paper/regression/ZINC/configs/network_ZINC.yml` + `parameters_ZINC.yml`
**Baseline**: ~0.10 MAE on ZINC, reached after ~30 epochs (competitive: GIN ~0.25, PNA ~0.14, GSN ~0.10,
CIN ~0.08)

---

## Context

The ShareGNN ZINC model performs well, but a single dense linear layer holds ~80% of its weights. This
document locates those weights exactly, explains why the layer has the wrong shape, and proposes changes —
config first, code second. It also collects architecture improvements (new labels, edge-feature
integration, hyperparameter tuning) that are available today without new code.

---

## 1. Where the parameters actually are

Layer stack of `network_ZINC.yml` with resolved shapes:

| # | Layer | Shape / dims | Params |
|---|-------|--------------|--------|
| 1 | `linear aggr_features` | 21→10 (one-hot atom types) | ~220 |
| 2 | `layer_norm` | | 20 |
| 3 | `invariant_based_convolution` | 20 heads (`num:1` each) → out = 10×20 = **200** | ~10⁴–10⁵ |
| 4 | `linear aggr_channels` | 200→100 | 20,100 |
| 5 | `layer_norm` | | 200 |
| 6 | `invariant_based_aggregation` | 14 groups × `num:10` = **140 heads**, F=100 → out = 100×140 = **14,000** | W ~10³–10⁴, bias 14,000 |
| 7 | `reshape` | → (14000,) | 0 |
| 8 | **`linear aggr_features`** | **14,000 → 100** | **1,400,100** ← the 80% |
| 9 | `layer_norm` | | 200 |
| 10 | `linear` | 100→100 | 10,100 |
| 11 | `linear` | 100→1 | 101 |

### Why the invariant layers are cheap and the readout is not

ShareGNN **shares weights across the feature dimension**. In
`src/simplegnn/models/ShareGNN/layers/inv_based_message_passing.py:358`, the per-head weight count is the
number of distinct `(source_label, target_label)` pairs per property value — it is *not* multiplied by
`in_features`. Only the bias is per-feature (`bias_num = in_features × n_bias_labels`, line 414). The same
holds in `inv_based_pooling.py:52`: `weight_num = Σ_i n_node_labels[i] × n_heads_per_label[i]`.

So the "Share" in ShareGNN keeps the interesting layers small, and **all the parameter mass ends up in the
one layer that has no sharing at all.**

### Why layer 8 has the wrong shape

`InvariantBasedAggregationLayer` sets `out_features = in_features × num_heads` and `out_channels = 1`
(`inv_based_pooling.py:35-38`) — it *flattens* the (H=140, F=100) graph embedding into a 14,000-vector.
Layer 8 then learns a free parameter for every `(head, feature, output)` triple: 140 × 100 × 100. Head and
feature are semantically independent axes and the dense map is blind to that: it is a rank-unconstrained
map over an outer-product space, sitting *after* pooling, trained on ~10k graphs.

Two consequences:

- **Overfitting surface.** 1.4M post-pooling params on 10k training graphs, with `weight_decay` currently
  **0.0** — it defaults to `[0.0]` in `src/simplegnn/framework/run_configuration.py:143` and is never set
  in `parameters_ZINC.yml`.
- **Benchmark budget.** The standard ZINC-subset protocol (Dwivedi et al.) caps models at **~100k params**.
  At ~1.75M total, 0.10 MAE is not an apples-to-apples comparison with the GIN/PNA/GSN numbers it would
  normally be placed against. Shrinking the readout is a *scientific* fix as well as a generalization one.

Note also: the 10 replicas within each aggregation group share the same label description and differ only
by initialization. The embedding has a block structure the dense readout never exploits.

---

## 2. Fixing the readout

### Stage A — config only, do this first

The readout cost is `H × F × out`, and both `H` and `F` are YAML choices. No code change:

- Layer 4 `out_features: 100` → **32** (shrinks F, and the aggregation bias with it)
- Aggregation `num: 10` → **4** (H: 140 → 56)

Readout becomes 56 × 32 = 1,792 → 100 = **179,200 params (~8× smaller)**. Pushing further to `num: 2`,
`F=32` gives 28 × 32 = 896 → 100 = **89,600**, landing the whole model near the 100k benchmark budget.

Do this first because it *tests whether the capacity was ever needed*. Expectation: MAE holds or improves.
If it holds, the problem is solved with a YAML edit and no new code.

### Stage B — factorized readout (only if Stage A costs accuracy)

Restore the (H, F) structure and factorize. Needs a small code change, because aggregation currently
flattens unconditionally:

1. Add a `flatten: False` option to `InvariantBasedAggregationLayer` (`inv_based_pooling.py`) returning
   `(B, H, F)`, with `out_channels = num_heads`, `out_features = in_features`.
2. Propagate `in_channels → num_heads` for `LinearLayer` in `src/simplegnn/models/model.py`. Today a plain
   linear always has `num_heads = 1` (it has no `heads` key), so `channel_wise` is unreachable after
   pooling.
3. Then either:
   - **`mode: channel_wise`** (already implemented, `nn_standard/linear.py:40`): per-head F-map 100→8 gives
     140×100×8 = 112,000, then reshape → 1,120 → 100 = 112,000. ~224k total.
   - **CP/Tucker readout** (new mode, best): `W[h,f,o] ≈ Σ_r A[h,r]·B[f,r]·C[r,o]`. At rank 32:
     140×32 + 100×32 + 32×100 ≈ **11k params** — ~125× reduction, and it encodes exactly the head/feature
     independence the data actually has.

### Stage C — regularize what remains

- **`weight_decay`** is supported and grid-searchable but currently 0. Most likely free win. Grid
  `[0.0, 1e-6, 1e-5, 1e-4]` and switch `optimizer: Adam` → **`AdamW`** (supported,
  `framework/model_configuration.py:596`) so decay is decoupled.
- **Dropout** exists as a standalone layer (`layer_type: dropout`, key `p` —
  `nn_standard/dropout.py:10`). Note that the `dropout` key on a `linear` layer is **silently ignored**:
  `LinearLayer.forward` only does matmul + bias + activation. Insert an explicit
  `- {layer_type: dropout, p: 0.1}` before the readout.

---

## 3. Architecture improvements

### New labels (all supported today; no new code)

Catalogue in `src/simplegnn/models/ShareGNN/preprocessing/preprocessing.py:49-174`: `primary`, `trivial`,
`index`, `degree`, `wl`, `wl_labeled`, `wl_labeled_edges`, `simple_cycles`, `induced_cycles`, `subgraph`,
`cliques`, `betweenness_centrality`.

- **`wl_labeled` is only ever used at `depth: 0`** — and depth 0 short-circuits to
  `save_labeled_degree_labels` (`preprocessing.py:100`), i.e. just *(atom type, degree)*. Proper WL
  refinement (`depth: 1`, `depth: 2`) is **never used in the ZINC conv layer**. This is the most standard,
  most powerful invariant in the catalogue and it is switched off. Add depth-1/2 heads, capped with
  `max_labels`. Highest expected value of anything in this section.
- **Combined labels in conv heads.** `label_type` accepts a *list* and `layer_to_labels` combines them
  recursively (`preprocessing.py:22-30`). Used in the aggregation heads (`[induced_cycles, primary]`) but
  **never in the convolution heads**. Try `head: {label_type: [primary, degree]}` or
  `[primary, induced_cycles]` for far more discriminative message-passing rules.
- **`betweenness_centrality`** with `num_bins` — a cheap global positional signal, currently unused. ZINC
  targets (constrained solubility) are globally sensitive, so this is worth a head.
- **`subgraph`** — arbitrary motif counts via the `subgraphs:` config key (e.g. `nx.star_graph(3)` for
  branch points). Rings are already covered; branch/functional-group motifs are not.
- **`cliques`** — low value here; ZINC is dominated by rings ≥5, not triangles.

Caveat: bigger label vocabularies grow `weight_num` in the conv layer. Control with `max_labels` and by
grid-searching `rule_occurrence_threshold` (currently 2) over `[2, 5, 10]`.

### Edge features — currently the weakest part

ZINC bond types (single/double/triple) enter in only two thin ways: `wl_labeled_edges` labels, and the
`edge_label_distances` property with **`cutoff: 1`**.

That cutoff is the problem. `write_distance_edge_properties`
(`ShareGNN/preprocessing/properties.py:152`) builds a property key of
`(distance, number_of_paths, multiset_of_bond_labels_along_all_shortest_paths)` — a genuinely rich,
bond-aware path descriptor. **With `cutoff: 1` all of that is discarded and only "adjacent + bond type"
survives.**

- **Raise `edge_label_distances` cutoff to 2–3.** This turns on path-composition-aware rules — arguably
  ShareGNN's most distinctive capability, currently disabled. Zero new code.
- Pair it with `wl_labeled_edges depth: 1` head/tail labels.
- Cost: the property-key vocabulary grows fast, and preprocessing is `all_pairs_all_shortest_paths` +
  `deepcopy`, O(n²·paths) — fine at ZINC's ~23 nodes/graph, but the *weight* count will jump. Raise
  `rule_occurrence_threshold` to compensate.

### Hyperparameters

- **`weight_decay`**: currently 0. See Stage C. Biggest free win.
- **`min_lr: 0.0001`** is a high floor. Standard ZINC recipes anneal to 1e-5/1e-6. Lower it, and raise
  `epochs` from 150 → 300–500 (ZINC benchmarks train long; early stopping is disabled anyway).
- **`precision: double`** — float64 is ~2× slower on GPU with no accuracy benefit for this regression. The
  invariant layers are index/`torch.take`-based, so precision is not load-bearing there. Switching to
  `float` buys compute for longer training and more HPO.
- **Loss**: MAE matches the ZINC metric (keep for eval), but training with Huber/smooth-L1 and *evaluating*
  MAE sometimes converges better.
- **Duplicate heads**: `primary/distances`, `edge_label_distances`, `wl_labeled_edges d0` and `d1` each
  appear **twice** with `num: 1` in `network_ZINC.yml`. These are identical rules with different init —
  equivalent to `num: 2`. Not a bug, but it is ensembling, not expressivity. Worth knowing before tuning.
- If the model stays large, consider **ZINC-full** (250k graphs) rather than the 12k subset — 1.75M params
  on 10k training graphs is the real mismatch.

---

## 4. Suggested order

1. Set `weight_decay` + `AdamW`, lower `min_lr`, raise `epochs`. Pure `parameters_ZINC.yml` edit; no
   architecture change. Establishes a regularized baseline.
2. Stage A readout shrink (`F=32`, `num=4`). Confirm MAE holds at ~180k params.
3. Turn on `wl_labeled depth: 1-2` conv heads and raise `edge_label_distances` cutoff to 2.
4. Only if step 2 lost accuracy: build the factorized readout (Stage B).

---

## 5. Verification

- **Parameter count**: `python -m simplegnn.utils.param_count --config <main_config>` prints the per-layer
  breakdown (this module was added for exactly this).
- **Baseline reproduction**: `python experiments/base_paper/regression/ZINC/main_ZINC.py` on the current
  config to re-establish 0.10 MAE as the reference.
- **One change at a time**: the framework grid-searches list-valued params natively, so
  `weight_decay: [0.0, 1e-6, 1e-5, 1e-4]` sweeps via `run_configurations()`, and `evaluate_results()` picks
  the best on validation.
- **Regression guard**: `pytest tests -q` after any change to `inv_based_pooling.py`, `linear.py`, or
  `model.py`.

---

## 6. Implementation (what was built)

### Measured parameter counts

`python -m simplegnn.utils.param_count --config ...`, per architecture:

| Config | Readout | Total | Note |
|--------|---------|-------|------|
| `network_ZINC.yml` (baseline) | 1,400,100 (70%) | 1,992,824 | was 3,423,445 — see the dead-weights bug below |
| `network_ZINC_small.yml` | 179,300 (27%) | 660,522 | F=32, H=56 |
| `network_ZINC_labels.yml` | 204,900 (28%) | 722,896 | small + the extra invariants |
| `network_ZINC_factorized.yml` | **10,980 (1.8%)** | 603,704 | full capacity (F=100, H=140), rank-32 CP |

**Two corrections to the analysis above.** (a) §1's table underestimates the message-passing layer: it is
**405,493 params**, not 10⁴–10⁵. Once the readout is fixed it becomes the dominant cost by far (60–67%), so
the "~100k benchmark budget" of §2 Stage A is *not* reachable by shrinking the readout alone — the conv
weight count has to come down too (`rule_occurrence_threshold`, `max_labels`, fewer heads). (b) The readout
was carrying twice the weights the table claims, because of a bug (below).

### Bugs found and fixed while implementing

1. **Dead duplicate weights in every linear layer** (`nn_standard/linear.py`). `LinearLayer.__init__` built
   a `torch.nn.Linear(in_features, out_features)` that `forward` never used — its parameters were
   nonetheless registered, so they sat in `net.parameters()`, in the optimizer, and in every checkpoint.
   On ZINC that was **1,430,621 dead parameters, 42% of the model** (3.42M → 1.99M after removal). They
   never received gradients, so removing them changes no numerics — but any parameter count taken from
   `net.parameters()` before this was inflated. *Old checkpoints in `results/` will no longer load with
   `strict=True`; they are regenerable.*

2. **`layer_norm` silently skipped in the per-graph forward** (`nn_standard/layer_normalization.py`).
   It handled 2D and 3D tensors only. The per-graph forward hands the graph embedding to the layer_norm
   after `reshape` as a **1D** tensor, which fell through *unnormalized*; the batched forward hands it a 2D
   `(B, F)` batch, which *was* normalized. So the two forwards computed **different functions** for the ZINC
   architecture — which has exactly this tail and trains with `share_gnn_forward: {batched: True}`. Fixed by
   normalizing over the last axis for 1D/2D, and (for the new unflattened aggregation) over the feature axis
   for batched 3D. Regression test:
   `tests/test_batched_share_gnn.py::test_layer_norm_on_graph_embedding_matches_per_graph`.

### New code

- `InvariantBasedAggregationLayer` — **`flatten: False`** (`inv_based_pooling.py`) keeps the head axis:
  `out_channels = H`, `out_features = F`, emitting `(H, 1, F)` per graph and `(B, H, F)` batched instead of
  folding the heads into a flat `H*F` vector.
- `LinearLayer` — new **`mode: factorized`** with a `rank` key: the CP readout of §2 Stage B,
  `W[h, f, o] = Σ_r A[h, r]·B[f, r]·C[r, o]`. Also fixed `channel_wise` and `aggr_channels` for the batched
  graph-level shape.
- `GraphModel.get_model_layer` — propagates `in_channels → num_heads` for the channel-aware linear modes.
  Without this a linear layer always had `num_heads = 1` (it has no `heads` key), which is why
  `channel_wise` was unreachable after pooling.
- `simplegnn.utils.param_count` — the per-layer parameter table used above.

### New configs (the baseline files are untouched)

Run with `python experiments/base_paper/regression/ZINC/main_ZINC_v2.py --variant <name>`:

| Variant | Content |
|---------|---------|
| `small` | Stage A + C: F 100→32, aggregation `num` 10→4, dropout before the readout |
| `labels` | `small` + WL depth 1/2 conv heads, combined conv labels, `edge_label_distances` cutoff 2, betweenness |
| `factorized` | Stage B: `flatten: False` + rank-32 CP readout at full baseline capacity |

All three share `parameters_ZINC_v2.yml`: **AdamW**, `weight_decay: [0.0, 1e-6, 1e-5, 1e-4]` (grid),
`min_lr` 1e-4 → 1e-6. `epochs` stays at **150** — contrary to §3, the baseline already reaches ~0.10 MAE
after ~30 epochs, so the epoch budget is not the constraint. `precision` stays `double` so that a comparison
against the baseline isolates the architecture.

Verified for all four configs: the model builds, and the batched forward matches the per-graph forward
(max abs diff < 1e-15) with every parameter receiving a gradient. **Accuracy is unmeasured.**
