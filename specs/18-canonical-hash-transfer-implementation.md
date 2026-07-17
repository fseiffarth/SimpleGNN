# Canonical Label Hashes + Hash-Keyed Transfer — Full Implementation Plan

**Created:** 2026-07-17
**Status:** Plan (approved direction; supersedes the sketch in
[`17-hashed-node-label-ids.md`](17-hashed-node-label-ids.md) with full per-occurrence detail).
**Sequencing rationale:** hash first (Part A) — it is cheap, durable, independently useful, and makes
the cross-dataset overlap measurement (Part B0, the go/no-go gate for the transfer engine) a trivial
vocabulary intersection. Transfer engine + config + example (Part B) follows.
**Relation to other specs:** replaces spec 11's "original label value" keys with canonical hashes
(`match: hashes`); the transfer *strategies* (finetune/linear-probe/freeze) and provenance ideas of
specs 07/07a are folded in here in minimal form — this plan is self-contained and does not require
implementing 07/07a first.

---

## Part A — Canonical label hashes

### A0. Hashing helper module (new: `src/simplegnn/datasets/utils/label_hashing.py`)

One small module, no torch dependency in the core helpers:

```python
HASH_SCHEMA_VERSION = 1          # bump on ANY encoding change → old vocabularies invalid
RESERVED_INVALID = -2**63        # hash slot for the -1 invalid label
RESERVED_CAPPED  = -2**63 + 1    # hash slot for the max_labels "other" bucket

def stable_hash(*parts) -> int:
    """blake2b(digest_size=8) over a length-prefixed, versioned byte encoding of parts
    (ints → 8-byte LE, str → utf-8, bytes passthrough, tuple/list → recursive with
    type+length prefix). Returns the digest as a signed int64 bit-pattern.
    NEVER Python hash() (salted), never repr() of unsorted containers."""

def hash_vocabulary(signatures: dict[int, tuple], label_kind: str, params: tuple) -> "np.ndarray":
    """original-label-id → int64 hash; every hash = stable_hash(HASH_SCHEMA_VERSION,
    label_kind, params, signature). Raises on within-vocabulary collision (two distinct
    signatures, same hash) — exact detection, signatures are in hand here."""
```

Every hash commits to `(schema_version, label_kind, params, signature)` so `wl_3` can never alias
`wl_4`, `closed_walks_2_6` never aliases `closed_walks_2_8`, and an encoding change invalidates
loudly. Unit-test vectors pinned in the test file (hard-coded expected int64s) so a refactor that
silently changes hashes fails CI.

### A1. Label file format v2 + `NodeLabels` + loader

**Current format** (`save_labels_to_file`, `node_labeling.py:1050–1069`): a 3-tuple
`(dataset_name, label_name, relabel_node_labels(labels, max_labels))` where the tensor is `(N, 2)`
(col 0 original ids, col 1 frequency-sorted ids). Loaded by `load_labels` (`:21–51`) →
`NodeLabels` (`NodeLabels.py`), whose `.node_labels` (col 1) is what all model code consumes.

**Format v2**: a dict (still `weights_only=True`-loadable):

```python
{'dataset_name': str, 'label_name': str,
 'node_labels': Tensor(N, 2),                      # unchanged content
 'label_hashes': Tensor(num_unique, ) int64,       # freq-sorted-id -> hash (index = col-1 id)
 'hash_meta': {'schema': 1, 'canonical': bool,     # canonical=False => dataset-relative labeling
               'kind': str, 'params': tuple,
               'capped': bool}}                     # True when max_labels merged a bucket
```

Changes:

- `save_labels_to_file(file, dataset_name, label_name, graph_node_labels, max_labels, label_hashes=None, hash_meta=None)`
  (`node_labeling.py:1050`): accepts an optional original-id→hash array. It must apply **the same
  permutation** `relabel_node_labels` applies: factor `relabel_node_labels` (`:1631–1655`) so it also
  returns `sorted_indices` (the original→freq-sorted map already computed at `:1646`); reorder
  `label_hashes` with it. Ids merged by `max_labels` capping (`:1650–1651`) collapse to
  `RESERVED_CAPPED` and set `hash_meta['capped']=True` (unless exactly one original id maps there —
  then its true hash survives). Invalid `-1` → `RESERVED_INVALID`. When `label_hashes is None`
  (labeling not yet hash-enabled, or non-canonical), write v2 with `label_hashes=None`-equivalent
  (omit key) so the file format is uniform going forward.
- `load_labels` (`:21`): accept both the legacy 3-tuple and the v2 dict.
- `NodeLabels.__init__` (`NodeLabels.py:5`): new optional args `label_hashes`, `hash_meta`; expose
  `self.label_hashes` (or `None`) and `self.has_canonical_hashes` (True iff hashes present, schema
  matches `HASH_SCHEMA_VERSION`, and `canonical` flag set). **No behavior change** for
  `.node_labels` / `.num_unique_node_labels` — all 8 consumer sites (listed in A4) keep working
  untouched.
- `combine_node_labels` (`node_labeling.py:53–141`): after `torch.unique(stacked, dim=0)` (`:128`),
  `unique_labels` rows are `(id_a, id_b)` pairs — combined hash per unique row =
  `stable_hash(schema, 'combined', (), (hash_a[id_a], hash_b[id_b]))`; canonical iff both inputs
  canonical. The artificial invalid row (`:126–130`) → `RESERVED_INVALID`.

### A2. Occurrence-by-occurrence: every label producer

Dispatcher: `models/ShareGNN/preprocessing/preprocessing.py:layer_to_labels` (`:14–196`) routes each
`label_dict` to a `save_*` function; the combined-label branch (`:22–44`) goes through
`combine_node_labels` (covered in A1). Each producer below gets a `label_hashes` array threaded into
its `save_labels_to_file` call. The `NodeLabelingBase` class hierarchy
(`node_labeling.py:377–974`) is a parallel, partially dead path (its `save_labels_to_file` ends in
`raise NotImplementedError`, `:604`, and the dispatcher never calls it) — **out of scope**; add a
comment pointing here.

| # | Producer (file:line) | Signature that exists today | Hash derivation | Canonical? |
|---|---|---|---|---|
| 1 | `save_primary_labels` (`node_labeling.py:1071`) | raw primary value per node (`graph_data.node_labels['primary']`) | `H('primary', (), raw_value)` per unique raw value | **flagged per-dataset**: TU datasets enumerate atom types arbitrarily (`canonical: False` default; config override `primary_labels_canonical: True` for datasets known to share coding, e.g. ZINC/QM9 atomic numbers) |
| 2 | `save_degree_labels` (`:1096`) | degree int (`:1112–1115`) | `H('degree', (), d)` | yes |
| 3 | `save_labeled_degree_labels` (`:1128`) | string `f'{own}\|{sorted_neighbors}'` already built at `:1151–1154` | `H('wl_labeled_0', (), identifier)`; inherits primary-label caveat (identifier is built from primary values) | iff primary canonical |
| 4 | `save_trivial_labels` (`:1175`) | constant 0 | `H('trivial', (), 0)` | yes (degenerate) |
| 5 | `save_index_labels` (`:1198`) | node index | none — **excluded** (`canonical: False`, no hashes) | no |
| 6 | `save_wl_labels` (`:1234`) → `weisfeiler_lehman_node_labeling` (`node_labeling_functions.py:132`) | per-dataset color ids from `_wl_color_refinement` | recursive per-class WL hash — **§A3** | yes |
| 7 | `save_wl_labeled_labels` (`:1260`) — same WL path, `labeled=True` | colors seeded from base labels (`node_labeling_functions.py:151–165`) | §A3 with round-0 hashes = base-label hashes (base vocabulary loaded from the base label file, which the dispatcher generates first — `preprocessing.py:248–249` already orders this) | iff base labels canonical |
| 8 | `save_wl_labeled_edges_labels` (`:1290`) → `_weisfeiler_lehman_node_labeling_nx` (`node_labeling_functions.py:174`) | nx blake2b hex digests **already computed** then discarded (`:199–227`) | keep the final-iteration digest per node before the `hash_dict` compaction: `H('wl_edges', (depth,), final_hex_digest)`. Near-free. | iff primary node+edge labels canonical (nx seeds from them) |
| 9 | `save_cycle_labels` (`:1319`) | `_canonical_count_string(count_dict)` (`:1345,1354`) | `H(kind, (min_len, max_len), canon_string)`; the "no cycles" fallback label (`:1357`) → `H(kind, params, 'none')` | yes |
| 10 | `save_in_circle_labels` (`:1371`) (not dispatched, keep for completeness) | binary 0/1 | `H('in_cycle', (bound,), b)` | yes |
| 11 | `save_subgraph_labels` (`:1409`) | `_canonical_count_string` over pattern-id counts (`:1441,1450`) | `H('subgraph', (canonical form of the subgraph list,), canon_string)` — the pattern list comes from config (`preprocessing.py:154`); hash the patterns via sorted edge lists so `id` renumbering across configs doesn't break identity | yes, given identical pattern graphs |
| 12 | `save_clique_labels` (`:1467`) | `_canonical_count_string` (`:1485,1494`) | `H('cliques', (max_clique,), canon_string)`; no-clique fallback (`:1497`) → `'none'` | yes |
| 13 | `save_betweenness_centrality_labels` (`:1511`) → `BetweennessCentralityNodeLabeling.generate` (`:832–877`) | percentile bin index — bins from **dataset** percentiles (`:858–859`) | `H('betweenness', (num_bins,), bin)` but `canonical: False` — bin edges are dataset-relative | **no** (flagged) |
| 14 | `save_closed_walk_labels` (`:1567`) → `ClosedWalkNodeLabeling.generate` (`:943–974`) | profile tuple `((A^l)_ii …)` (`:969`) | `H('closed_walks', (min_len, max_len), profile)` | yes |

Implementation pattern shared by 1–5, 8–14: each producer already builds a
`signature → original_id` dict (e.g. `label_dict` at `:1346`, `profile_to_label` at `:973`,
`unique_neighbor_label_dict` at `:1160`); invert it, hash each signature via `hash_vocabulary`, pass
the array to `save_labels_to_file`. **No labeling algorithm changes, ids stay bit-identical** —
existing experiments reproduce exactly (aside from regenerated files; see A5 migration).

### A3. Canonical WL hashes on the vectorized path (occurrences 6–7)

`weisfeiler_lehman_node_labeling` (`node_labeling_functions.py:132–171`) already builds the
disjoint-union `edge_src/edge_dst` arrays — exactly what per-class hashing needs.

1. `_wl_color_refinement` (`:65–110`) gains `return_rounds=False`; when True it returns the list of
   per-round color arrays (`colors` before each refinement step plus the final one). Memory:
   `rounds × N` int64, freed after hashing.
2. New `_canonical_wl_hashes(edge_src, edge_dst, round_colors, seed_hashes) -> (final_colors_unique → int64 hash)`:
   - `canon[0][c] = seed_hashes[c]` (constant `H('wl', (depth,), 'init')` for unlabeled; base-label
     hashes for `wl_labeled`).
   - Round `r`: for each unique color `c` in `round_colors[r]`, pick the first node `v` with that
     color (`np.unique(..., return_index=True)`), gather
     `sorted([canon[r-1][color_{r-1}(u)] for u in N(v)])` from the CSR-style neighbor layout the
     refinement already constructs (`deg/start/order`, `:81–85`), and hash
     `canon[r][c] = stable_hash(canon[r-1][color_{r-1}(v)], neighbor_hash_multiset)`.
   - One representative per class is sound because same-color nodes have identical
     (own-color, neighbor-color-multiset) by construction of the refinement.
3. `_wl_labels_to_output` (`:113–129`) gains an optional `class_hashes` pass-through so
   `weisfeiler_lehman_node_labeling` can return `(graph_node_labels, unique, db_unique, label_hashes)`
   (4th element optional to keep old callers working); `save_wl_labels`/`save_wl_labeled_labels`
   forward it.
4. **Depth-convergence caveat:** the refinement early-exits when the partition stabilizes (`:107–108`).
   Hashing must keep iterating the *hash* chain to the full `rounds` (identity hashes of a stable
   partition still change per round) **or** — simpler and recommended — define the canonical identity
   as `H('wl', (depth,), canon[R_actual][c], R_actual)` only if we accept that two datasets may
   stabilize at different rounds for the same structure. They can't for the same structural
   neighborhood at the same depth parameter, but to be safe: **iterate hashes for exactly `rounds`
   rounds, no early exit in the hash chain** (cost is trivial — per class, not per node).
5. **Parity test:** for a set of small graphs, assert equality of the induced partition with
   `nx.weisfeiler_lehman_subgraph_hashes` and — the key transfer property — that two *disjoint
   datasets* containing isomorphic neighborhoods produce identical hashes (generate one graph set,
   split it two ways, compare vocabularies).

### A4. Consumers — verified no-change surface

All model/framework consumers read only `.node_labels` / `.num_unique_node_labels` and are untouched:
`framework/utils/preprocessing.py:742–749` (loading), `inv_based_message_passing.py:255–261, 341–343`
(counts + label vectors), `inv_based_pooling.py:51–89`, `inv_based_positional_encoding.py:52–89`,
`graph_dataset.py:1248–1325` (one-hot input features), drawing code
(`inv_based_message_passing.py:1533+`), legacy `properties.py:123`. The only new consumer of
`label_hashes` is Part B. Property keys need **no hashing**: distance keys are ints, edge-label
distance keys are canonical tuples (`properties.py:218`) — serialized as-is (canonical iff primary
edge labels are; recorded per-manifest as a flag).

### A5. Part A tests + migration

- Unit: pinned hash vectors; vocabulary collision detection; permutation correctness under
  `relabel_node_labels` (hash follows its label through frequency sorting); capping →
  `RESERVED_CAPPED` + `capped` flag; combined-label hashes; v1-file load compatibility.
- WL parity tests (A3.5).
- Regression: `pytest tests -q`; one example run (`examples/share_gnn_basic/main.py`) after deleting
  `data/TUDatasets/labels/MUTAG/` to confirm regenerated v2 files train identically.
- Migration: none needed for training (v1 files keep working). Export/transfer requires v2 →
  the Part B error message says "delete `<labels dir>` and rerun `preprocessing()`". No silent
  recompute.

**Effort: ~2 days.** Runtime cost recap (unchanged from spec 17 §3): zero on train/inference; label
generation +seconds (per-class hashing); +8 B per unique label on disk.

---

## Part B — Hash-keyed transfer: engine, config, example

### B0. Overlap measurement (go/no-go gate, ~0.5 day, ships first)

New `src/simplegnn/framework/utils/transfer.py`:

```python
def measure_label_overlap(source_label_file, target_label_file) -> OverlapReport
def measure_pair_overlap(source_labels, source_props, target_labels, target_props,
                         property_values) -> OverlapReport   # (hash, hash, value) triple mass
```

Loads two independently preprocessed datasets' v2 label files + property files, intersects
vocabularies, and reports: unique-label overlap, and **occurrence-weighted** coverage of the target's
`(src_hash, tgt_hash, property_value)` pairs by the source (the quantity that upper-bounds what §B4
can transfer). Plus a tiny driver `examples/transfer_learning/measure_overlap.py` printing a table
per label type (wl_2, wl_3, closed_walks, induced_cycles) for NCI1→DHFR and ZINC→QM9. **Decision
point: if weighted coverage is single-digit % for every usable labeling, stop here.**

### B1. Slot→key reconstruction inside the invariant layers

The conv layer discards the `torch.unique` values it would need to name each weight slot
(`inv_based_message_passing.py:424`: `_, indices, counts = torch.unique(encoded_labels, ...)`).
Changes (all in `inv_based_message_passing.py`):

- `_save_cached_indices` (`:1004`) / `_load_cached_indices` (`:958`): additionally store/load
  `uniques` (the sorted unique encoded values, pre-threshold) and the `max_label` encoder base
  (`:417`). Backward compatible: old cache files load fine, `uniques=None`.
- `_build_distributions` (`:295`): retain per (head_id, property_key):
  `(uniques, max_label, num_weights, base_offset)` in a new `self._slot_keys` list — built on both
  the miss path (values in hand) and the hit path (from the extended cache; if the cache predates
  `uniques`, recompute `encoded_labels` — one gather per key, same as the miss path minus unique).
  Memory: `num_weights` int64 per key ≈ size of `Param_W` — acceptable, and only materialized when
  `save_best_model` or a `transfer` block is active (lazy flag).
- New `export_weight_keys(self) -> dict`: walks `weight_offset_description` (`:464`) /
  `self._slot_keys`; decodes each unique value as `(src_id, tgt_id) = divmod(u, max_label)`, applies
  the same threshold mask that produced the final indices (`:434–445` — rows with mapped index ≥ 0),
  maps ids → hashes via `graph_data.node_labels[desc].label_hashes`, and emits per head-config:
  `{'src_hash': int64[num_weights], 'tgt_hash': int64[num_weights], 'property_key': …,
  'replica': …, 'param_slice': (offset, offset+num_weights)}` — replicas `n>0` reuse the same key
  rows shifted by `n*num_weights` (`:462–463` layout). Bias table: per bias slot
  `(bias_hash, feature, head_col) → Param_b index` from `_bias_slot`/`_b_off` (`:480–496`); the
  per-label unique inverse at `:482` needs its unique *values* kept too (same pattern, tiny).
- `inv_based_pooling.py`: `export_weight_keys` from the `torch.unique` at `:74` (keep values) +
  `col_offset` layout (`:76–83`): `(label_hash, replica) → Param_W index`; bias `Param_b` is
  `(num_heads, in_features)` — transferred positionally (config-shaped, not dataset-shaped).
- `inv_based_positional_encoding.py`: `(label_hash, entry k) → Param_W index` from `_pe_idx_{h}`
  uniques + `offset + i*num + k` layout (`:61–76`).
- Layers whose labels lack canonical hashes (`has_canonical_hashes == False`): the head is exported
  with `'canonical': False` and skipped on import (logged), unless
  `invariant_transfer.allow_non_canonical: True`.

### B2. Portable checkpoint sidecar

`model_configuration.py` saves best models at `:1217–1218` and `:1247–1248`
(`torch.save(self.net.state_dict(), final_path)`). Add, guarded by
`save_transfer_keys: True` (hyperparameter config, default True when any invariant layer is present
and hashes are available):

```
model_<config>_run_<r>_val_step_<k>.pt          # unchanged
model_<config>_run_<r>_val_step_<k>.keys.pt     # NEW sidecar
```

Sidecar content: `{'schema': 1, 'layers': {state_dict_prefix: export_weight_keys() output}}` plus a
human-readable summary block (label/property description strings per head, counts). Binary `.pt`
(hash tables with up to millions of rows; JSON rejected). Assembled by a `GraphModel` method
`export_transfer_keys()` iterating `net_layers` and calling each invariant layer's
`export_weight_keys` (non-invariant layers contribute nothing — they transfer by name+shape).

### B3. Import: the remap engine (`framework/utils/transfer.py`)

```python
def apply_transfer(target_net, source_state_dict, source_keys, cfg) -> TransferReport
```

1. **Standard layers** (`linear`, `layer_norm`, `batch_norm`): copy by state-dict name when shapes
   match — except the **head**: the last `linear` layer (and anything after the final aggregation if
   `head.reinit: always`) keeps its fresh init. With `input_features: {name: constant}` (the
   transferable configuration; warn otherwise — one-hot width is dataset-dependent,
   `graph_dataset.py:1248`), intermediate linears always shape-match.
2. **Invariant layers**: call the target layer's `export_weight_keys()` (its own dataset's build just
   ran in `GraphModel.__init__`), then per head-config a vectorized join: encode
   `(src_hash, tgt_hash)` rows of source and target into a sort-merge (`torch.unique`-style
   `searchsorted` on a packed 128-bit key stored as two sorted int64 columns, or a Python dict at
   `num_weights ≤ 10^7` — benchmark, dict is likely fine at seconds), restricted to equal
   `property_key` and `replica`; copy matched source weights into the fresh `Param_W`, leave the
   rest on their initialization (`on_missing: reinit`) or zero them (`zero`). Same for aggregation /
   positional-encoding tables and hash-keyed bias slots.
3. **Report**: per layer `matched/total` slot counts and occurrence-weighted coverage, appended to
   the run's result JSON and printed; warn under `min_overlap_warn`.
4. Wire-up in `model_configuration.initialize_model` (`:652–679`): today the tuple path does
   `self.net = pretrained_network` verbatim (only same-dataset-safe). New: when the hyperparameter
   config has a `transfer:` block, build the target net normally, resolve the source checkpoint +
   sidecar, and call `apply_transfer`. The old in-memory tuple path stays as-is for same-dataset
   use. `core.py:753–773` passes the config through unchanged.
5. **Freezing / strategies** (minimal, from 07): `strategy: finetune` (default, nothing special) |
   `linear_probe` (freeze every layer that received transferred weights; requires param groups in
   `set_optimizer`, `model_configuration.py:588–617`). `freeze:` glob list on state-dict prefixes.

### B4. Config schema (hyperparameter tier) + validation

```yaml
# parameters.yml of the FINETUNE experiment
transfer:
  source:
    results_path: results/pretrain/ShareGNN_NCI1/   # results dir of the pretraining run
    dataset: NCI1                                    # db name of the source checkpoint
    select: best                                     # best | {config_id: 0, run_id: 0, validation_id: 0}
  strategy: finetune                                 # finetune | linear_probe
  head:
    reinit: always                                   # always | never (default: always)
  invariant_transfer:
    match: hashes                                    # hashes | none (reinit all invariant layers)
    on_missing: reinit                               # reinit | zero
    allow_non_canonical: False                       # also match betweenness/primary etc.
    min_overlap_warn: 0.10
  freeze: []                                         # optional state-dict prefix globs
```

`configuration_checks.py` additions: key/enum validation as above; error if `match: hashes` and the
source sidecar is missing ("re-save the source model with save_transfer_keys / regenerate labels for
v2 hashes"); error if any needed target label file lacks canonical hashes (points at A5 migration);
warn if `input_features.name != constant`.

### B5. Example: `examples/transfer_learning/`

```
examples/transfer_learning/
├── main.py                    # runs: pretrain -> overlap report -> finetune
├── measure_overlap.py         # B0 driver, usable standalone
├── pretrain_main.yml          # NCI1 (TUDataset), graph_classification
├── finetune_main.yml          # DHFR, graph_classification, same models yml
├── models_ShareGNN.yml        # shared architecture, transferable labels ONLY:
│                              #   conv heads: wl_labeled depth 2 + induced_cycles 5..6,
│                              #   properties: distances [1..6]
│                              #   aggregation: induced_cycles + closed_walks heads
│                              #   (no betweenness — non-canonical; bias labels: wl_labeled 0)
├── pretrain_parameters.yml    # save_best_model: True, input_features: constant
└── finetune_parameters.yml    # transfer: block as in B4 pointing at the pretrain results
```

`main.py` = two `FrameworkMain` runs (standard 5-step workflow each, per
`examples/share_gnn_basic/main.py` pattern) with the overlap report printed between them; a
`--fast` flag caps epochs for smoke-testing. Note the model YAML deliberately avoids `primary` and
`betweenness_centrality` label types (non-canonical) — the doc header explains why, since that's the
main user-facing pitfall.

### B6. Part B tests

- Unit: slot→key reconstruction equals a brute-force re-derivation on a tiny dataset (build layer,
  enumerate pairs by hand, compare); replica offsets; bias keys; pooling/PE keys; join engine on
  synthetic overlap matrices (full/partial/disjoint → matched counts); `on_missing: zero`;
  non-canonical head skipping; missing-sidecar and v1-label-file error paths.
- Integration: two synthetic datasets sampled from one generator (`custom_benchmarks` rings/strings)
  preprocessed **independently** → pretrain on A, transfer to B, assert (a) high matched fraction for
  WL labels — the case spec 11's original-value keys could not deliver, (b) matched weights
  bit-equal, (c) finetune runs. CI-fast.
- End-to-end (manual, not CI): the NCI1→DHFR example; record overlap + learning curves vs
  from-scratch in the example README.

### Effort summary

| Phase | Days |
|---|---|
| A0–A2 vocabulary + all non-WL producers | 1 |
| A3 canonical WL + parity tests | 1 |
| B0 overlap measurement (**go/no-go**) | 0.5 |
| B1–B2 slot keys + sidecar | 1 |
| B3–B4 remap engine + config | 1–1.5 |
| B5–B6 example + tests | 1 |
| **Total** | **~5.5–6** (with a hard stop after B0 if overlap is poor) |

### Decisions (carried from spec 17, finalized)

- 64-bit hashes, int64 storage, within-dataset collision = hard error; schema-versioned encodings
  with pinned test vectors (H1).
- Vocabulary lives inside the label `.pt` (v2 dict) — no sidecar desync (H2).
- `primary` non-canonical by default, per-dataset config override; atom-type normalization to atomic
  numbers stays a future upgrade (H3).
- Betweenness ships flagged non-canonical (H4).
- Hash keys are the only new manifest format (`match: hashes`); spec 11's `.keys.json`
  original-value manifest is **not** implemented (H5 resolved: superseded before it existed).
