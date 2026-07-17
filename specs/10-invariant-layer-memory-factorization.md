# 10 — Invariant-Layer Memory Factorization (ZINC-full OOM fix)

**Status**: implemented
**Scope**: `InvariantBasedMessagePassingLayer`, `InvariantBasedAggregationLayer`
**Trigger**: ZINC-full (`experiments/base_paper/regression/ZINC/main_ZINC_full.py`) was
OOM-killed (kernel log 2026-07-16 22:11:52, `anon-rss` ≈ 111.5 GiB) during model
construction on a 122 GB machine.

---

## 1. Root cause (verified, not estimated)

ZINC-full: **249,456 graphs, 5,775,257 nodes, 133,002,620 node-pairs** with
shortest-path distance 1–23 (measured from
`data/ZINC/properties/ZINC-full/ZINC-full_properties_distances.pt`).

`InvariantBasedMessagePassingLayer._build_distributions()` materializes, per head,
one `(head, i, j, param_idx)` int64 row for **every matching node-pair in the whole
dataset**. Replaying the real build on the real label/property files with the real
`rule_occurrence_threshold: 10` for the ZINC network
(`configs/network_ZINC.yml`, conv layer = 16 head-configs / 20 heads):

| Structure | Rows | int64 size |
|---|---|---|
| 10 cycle heads × all 133M pairs (dist 1–23) | 1.33 B | 42.6 GB |
| primary×distances (×2 replicas), wl heads, edge-label heads | 0.59 B | 18.9 GB |
| **`weight_distribution` rows total** | **1.92 B** | **61.5 GB** |
| row→graph-id side tensors (`weight_chunk_graph_ids`) | 1.92 B | 15.4 GB |
| `bias_distribution`: nodes × 10 features × 20 heads | 1.16 B | 37.0 GB |
| **Live at end of build phase** | | **≈ 114 GB** |

The kill fired at 111.5 GiB RSS: the build had just finished and
`_merge_distribution_chunks()` called `torch.cat` on the 61.5 GB chunk list, which
needs a **second full copy** while the chunks stay alive (then an argsort + gather
for a third partial copy). Total demand ≈ 190 GB.

Threshold filtering removes ~0.01% of rows (cycle labels have only 2–8 unique
values, so every label-pair combo is frequent) — it does not help.

### Three levels of redundancy

1. **The same 133M `(i, j)` pairs are stored ~14×** (once per head-config covering
   them). They already exist once, shared, in the `Properties` object (2.2 GB).
   Only the `param_idx` column is head-specific — and even that is a pure function
   `param_idx = f(label_i, label_j, property_value)` per head-config.
2. **`bias_distribution` is a 37 GB broadcast of ~23 MB of information**: every
   head uses `bias: primary`, and the `(head, node, feature, param)` enumeration
   repeats the same per-node label index across 20 heads × 10 features. The
   forward only needs `Param_b[offset(h, f) + bias_label[node]]`.
3. **The merge transiently duplicates everything again** (cat + stable argsort +
   reorder while the chunk list is never freed).

Not duplicated (checked): the dataset between `preprocessing()` and
`run_configurations()`; label files (loaded once per run into the shared
`graph_data`).

The earlier working-tree mitigations (int32 rows, `precision: float`) are
insufficient: `precision` is irrelevant to integer index tensors, and int32 still
leaves a ~95–110 GB init peak plus ~56 GB held for all 150 epochs.

---

## 2. New design

### 2.1 Message-passing layer: factored, description-consolidated storage

Per **property description** (e.g. `distances`), built once and shared by all
head-configs (and all invariant layers) via the `Properties` object:

- `lp` — local pair list, `(P, 2) int32`, **graph-major** (all pairs of graph *g*
  contiguous, across all property values), node ids local to their graph.
- `key_id` — `(P,) int16`, the property-value index of each pair row.
- `slices` — `(G+1,) int64`, per-graph row ranges into `lp`.

Per **head-config** *h* (16 for ZINC), stored on the layer:

- `pv[h]` — `(P,) int32`, the **absolute `Param_W` index** of each pair row for
  replica 0, or **-1** if the row does not exist for this config (property value
  not in the config's list, label combo below `rule_occurrence_threshold`, or
  invalid `-1` label).
- `nw[h]` — `(num_property_values,) int64` small table: `num_weights` per property
  value, used to shift replicas (`replica n ⇒ pv + n · nw[key_id]`).
- Bookkeeping (`weight_num`, `weight_offset`, …) unchanged — **parameter layout
  and semantics are identical to the old implementation**, so training results
  are unchanged.

Sizes for ZINC-full: `lp` 1.06 GB + `key_id` 0.27 GB (shared) + `pv` 15 configs ×
133M × 4 B ≈ 8.0 GB ⇒ **≈ 9.3 GB total** (was 61.5 + 15.4 = 77 GB) and **no merge,
no dataset-wide sort** — each `pv` segment is written directly into a preallocated
tensor (suggestion 3 is realized by eliminating the merge entirely).

**Batch assembly** (new `_assemble_rows(positions)`): per description, one
vectorized range-gather builds the batch's pair rows + graph slots; per
head-config, one gather of `pv` + mask + replica/head-column arithmetic emits the
same `(head, i, j, param_idx)` rows the old code precomputed. Consumers
(`_batched_dense_messages`, `_batched_sparse_messages`, per-graph dense/sparse
forward, `get_graph_weights`, `draw`) receive identical row sets; dense scatters
are bit-identical, sparse paths coalesce (sort) per call instead of using
dataset-wide presorted buffers (`_ensure_per_graph_fwd_indices`,
`_ensure_batched_sparse_indices`, `_fwd_*` buffers: **removed**).

The fine-grained on-disk `(indices, counts)` cache is reused as-is (same files,
same format): on a hit, `pv = mapping[indices]` — the threshold relabeling is
unchanged.

### 2.2 Bias factorization

Per unique bias-label description: `bias_idx` — `(total_nodes,) int32` unique
inverse (23 MB for ZINC-full), shared across heads. Plus a precomputed
`(num_heads, in_features) int64` offset table `b_off` with
`b_off[h, f] = head_base + n·F·n_bias + f·n_bias` (exactly the old enumeration's
offsets). Forward:

```
current_B[node, h, f] = Param_b[b_off[h, f] + bias_idx[node]]
```

`bias_distribution` (37 GB) is **deleted**. This is also faster: the old batched
bias path read ~19 MB of enumerated rows per 128-graph batch.

### 2.3 Aggregation layer (suggestion 5)

`weight_distribution` was `(total_nodes, num_heads) int64` — 5.78M × 140 × 8 B =
6.5 GB for ZINC-full, with replica columns differing only by a constant offset.
Replaced by per-head-config `(total_nodes,) int32` unique-inverse vectors
(14 × 5.78M × 4 B = 324 MB) plus a `(num_heads,) int64` per-column offset vector;
`(N, H)` weight indices are reconstructed by gather + add at forward time
(same output, neutral runtime).

### 2.4 Optional materialized-row cache (`precompute_rows`)

The factored assembly costs ~1 ms extra per 64-graph step, which matters on
small datasets where the old dataset-wide rows were affordable anyway. The
layer therefore supports materializing the assembled rows once (graph-major,
4 int32 columns + per-graph slices) and serving batches by a single
range-gather — the old representation's per-step speed without its build
transients:

```yaml
share_gnn_forward:
  precompute_rows: auto   # default; True / False to force
  precompute_rows_max_bytes: 4294967296  # auto threshold (4 GiB)
```

`auto` materializes only when the rows fit under the threshold AND under 25%
of currently available RAM: NCI1 (~125 MB) materializes, ZINC-full (~31 GB)
stays factored. `True` on an oversized dataset raises a MemoryError with the
actual size instead of building it. The factored structures remain the ground
truth either way (equivalence pinned by
`test_precomputed_rows_match_factored_assembly`).

### 2.5 Fail-fast memory guard (suggestion 7)

Before allocating, both layers estimate their index-structure footprint from the
property/label slice metadata (O(1), no big allocations) and compare against
available RAM (`/proc/meminfo` `MemAvailable`, graceful fallback if unreadable).
If the estimate exceeds ~80% of available memory the layer raises a `MemoryError`
with the estimate and actionable hints instead of letting the OOM killer take the
machine down (only 2 GB swap ⇒ no warning today). Estimates > 1 GB are printed.

---

## 3. Expected memory & runtime (ZINC-full, this machine)

| | old (int64) | old (int32 patch) | new |
|---|---|---|---|
| init peak | ≈ 190 GB (OOM) | ≈ 95–110 GB (marginal) | ≈ 12 GB |
| steady state during 150 epochs | ≈ 105 GB | ≈ 56 GB | ≈ 10 GB |

Runtime:
- **Init**: much faster — no 1.9 B-row cat/argsort, no 37 GB bias enumeration.
- **Per batch (conv)**: assembly goes from one contiguous slice-copy (~1–2 ms per
  128-graph batch) to consolidated gathers (~3–5 ms). Steps are dominated by
  matmuls/backward ⇒ expected ≈ +5–10% step time, partly offset by the faster
  bias path. Measured with `tests/test_speed_share_gnn_nci1.py` before/after
  (numbers recorded in §5).
- **Aggregation**: neutral (same gather volume).

Alternative considered and deferred: computing `param_idx` per batch from
per-(config, key) label-combo lookup tables (no `pv` vectors at all, ~50 MB
total). Saves a further ~8 GB but adds an encode+searchsorted per pair per batch
and a bigger rewrite; revisit if 9 GB matters on some target machine.

---

## 4. Compatibility notes

- Parameter count, ordering, and initialization are unchanged ⇒ identical
  training trajectories (dense paths bit-identical; sparse paths differ only in
  fp summation order, as they already did between modes).
- The opt-in per-layer distribution cache (`cache: {layer_distributions: True}`)
  format version bumps (v2 ⇒ v3) and now stores the factored structures.
- The fine-grained `(indices, counts)` cache is format-compatible; existing files
  keep working.
- `set_weights` / `set_bias` / `get_graph_weights` / `draw` keep their behavior,
  backed by on-demand single-graph assembly.
- On CUDA the factored buffers move with `net.to(device)` as before
  (non-persistent buffers). A future knob could keep assembly on CPU and move
  only the per-batch rows.

## 5. Test & verification plan

1. Full `pytest tests -q` (MUTAG/NCI1 integration, batched-vs-pergraph
   equivalence, cache tests) must pass.
2. New unit test: factored assembly reproduces the legacy row set exactly on the
   MUTAG fixture (reference rows built by a straightforward per-graph loop).
3. NCI1 speed test before/after (record numbers here).
4. ZINC-full smoke: layer init on the real dataset stays under a few GB and
   completes (the previously-OOMing step).

### Measured results (filled in after implementation)

- pytest: PASS (see summary in session log)
- NCI1 speed before/after: see §6 addendum
- ZINC-full conv-layer init: see §6 addendum

## 6. Addendum — measured results

**Bit-exact equivalence** (golden capture from the pre-refactor code, replayed
against the factored implementation):

- MUTAG default fixture (1 head-config, primary labels): 18,298 weight rows +
  3,371 bias rows identical across all 188 graphs; aggregation index matrix
  identical.
- MUTAG adversarial fixture (3 head-configs incl. `num: 2` replicas,
  `rule_occurrence_threshold: 3`, wl_labeled with invalid labels,
  in_features 3): 106,016 weight rows + 40,452 bias rows identical.
- NCI1 speed fixture: weight_num / weight_offset / bias bookkeeping and
  per-graph rows identical. (A 194-parameter difference observed in early
  benchmarks was traced to *unrelated* uncommitted layer_norm/linear changes
  in the working tree, and a stale pre-bugfix `layerdist_*.pt` cache from
  2026-07-14 that HEAD silently loads — renamed to `.stale`.)
- Full pytest suite: 178 passed (same count as before the refactor).

**Layer-level microbenchmark** (NCI1 fixture, CPU, double precision, only the
invariant conv layer, quiet machine — earlier end-to-end comparisons were
invalidated by concurrently running stale benchmarks and by the unrelated
working-tree changes):

| | old | new (factored) | new (`precompute_rows` auto-materialized) |
|---|---|---|---|
| layer init | 0.72 s | 0.41 s | 0.73 s |
| batched 64-graph forward+backward | 1.71 ms | 2.68 ms | **1.33 ms** |
| per-graph forward | 0.063 ms | 0.182 ms | 0.099 ms |

With the default `precompute_rows: auto`, small datasets materialize and are
as fast as (batched: faster than) the old code; large datasets stay factored
and pay ~+1 ms per 64-graph step (≈16% of a full training step) — versus not
fitting in memory at all.

**ZINC-full smoke** (the previously OOM-killed step, 249,456 graphs,
`main_config_ZINC_full.yml`, batched CPU):

- data + labels + properties loaded: 8.7 s, RSS 6.0 GiB
- memory guard estimate printed: 9.2 GiB (151M shared pair rows, 2.09B pv
  entries across 16 head-configs) — passes on this 122 GiB machine
- model built: 120 s, **peak RSS 17.1 GiB** (old code: OOM-killed at
  ~117 GiB RSS with ~190 GiB total demand)
- forward+backward, 128-graph batch: 0.06 s, sane initial MAE loss
- steady state: **53 ms/step ⇒ ~1.7 min/epoch** (≈ 4.3 h train time for the
  150-epoch config, plus per-epoch evaluation), RSS flat at 17.1 GiB

Final suite: 201 passed, 1 skipped (includes the new coverage/assembly tests).
