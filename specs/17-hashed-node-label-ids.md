# Hashed Canonical Node-Label IDs for Cross-Dataset Weight Transferability

**Created:** 2026-07-17
**Priority:** HIGH (unblocks spec 11's D1)
**Status:** Plan only. No implementation in this document.

> **Builds on** [`11-sharegnn-transfer-learning.md`](11-sharegnn-transfer-learning.md) (weight-key
> manifest + key-based remap) and [`07-transfer-learning-redesign.md`](07-transfer-learning-redesign.md).
> Spec 11 matches invariant weights across datasets by *semantic key* but flags (D1, §3.2 caveat) that
> the keys are unreliable for `wl_*` and `primary` labels because label ids are compacted per dataset.
> This spec removes that limitation: **every labeling function emits, per label, a canonical hash
> computed from the label's structural signature alone** — the same structural role hashes to the same
> value in any dataset, with cryptographically negligible collision probability. Invariant weights then
> correspond to `(source_hash, target_hash, property_value)` triples (message passing) or a single
> `hash` (aggregation, bias, encoding), making them portable by construction.

---

## 1. Problem: label ids are dataset-relative

A ShareGNN invariant weight is identified by which labels/property it connects, but the label ids that
index it are artifacts of per-dataset compaction:

- Every labeling pipeline ends in a **per-dataset int compaction**. The fast WL path
  (`node_labeling_functions.py:65–129`, `_wl_color_refinement` + `_wl_labels_to_output`) compacts
  colors with `np.unique` / first-appearance ids. The nx fallback computes real blake2b subgraph
  hashes (`:199`) but immediately maps them to per-dataset ints (`hash_dict`, `:203–227`), discarding
  the canonical hash. Closed-walk profiles (`node_labeling.py:971–974`), cycle/clique count dicts
  (`_canonical_count_string`, `:977`), and labeled-degree identifiers (`:1147–1160`) are likewise
  canonical intermediates thrown away after `profile → i` enumeration.
- `relabel_node_labels` (`node_labeling.py:1631`) then **frequency-sorts** ids per dataset, and
  `NodeLabels` exposes only the compacted columns (`NodeLabels.py:9–12`). Nothing that reaches the
  model retains the label's identity.
- `InvariantBasedMessagePassingLayer._build_distributions`
  (`inv_based_message_passing.py:295–506`) encodes `(source_label, target_label)` pairs as
  `a*max_label + b`, runs `torch.unique` per property value, and assigns **contiguous parameter
  indices** into the flat `Param_W` (`weight_offset` bookkeeping, `:452–465`). Slot *k* therefore means
  a different triple in every dataset.
- Aggregation weights are indexed directly by label id (`inv_based_pooling.py:60–89`), bias weights by
  `(bias_label, feature, head)` (`inv_based_message_passing.py:473–496`).

Consequence: identical structural roles (same WL color, same closed-walk profile, same cycle profile)
get unrelated ids in different datasets, so weights cannot be matched across datasets — spec 11 works
around this with "original label values", which only helps for labelings whose raw value is already
canonical (degree, cycle lengths).

## 2. Proposal

### 2.1 Canonical hash as the label's identity

Each labeling function defines a **canonical byte encoding** of the structural signature it already
computes, and hashes it with a *stable, unsalted* algorithm — `hashlib.blake2b(digest_size=8)`
interpreted as `uint64` (stored as int64 bit-pattern). Explicitly **not** Python's built-in `hash()`
(salted per process via `PYTHONHASHSEED`) and not the id-compaction order.

Per labeling type:

| Label type | Canonical signature to hash | Canonical across datasets? |
|---|---|---|
| `degree` / `wl_0` | `b"deg:" + degree` | yes |
| `wl` (unlabeled init) | recursive WL hash, §2.2 | yes |
| `wl_labeled` | recursive WL hash seeded with base-label hashes | iff base labels canonical |
| `closed_walks` | profile tuple `((A^l)_ii …)` + min/max walk length | yes |
| `simple_cycles` / `induced_cycles` / `cliques` | `_canonical_count_string` output + parameter bounds | yes |
| `betweenness_centrality` | bin index | **no** — bins are dataset percentiles; hash carries a `dataset-relative` flag |
| `primary` | raw label value | only if the raw coding is shared (ZINC/QM9 atomic numbers: yes; TU per-dataset atom enumerations: no) — flag per dataset |
| `trivial` | constant | yes (degenerate) |
| `index` / `index_text` | — | no; excluded |
| combined labels (`combine_node_labels`, `node_labeling.py:53`) | `H(hash_a ‖ hash_b)` | iff both parts canonical |

The signature encoding includes a **schema version byte and the labeling parameters** (depth, walk
bounds, cycle bounds…), so `wl_3` and `wl_4` hashes never alias and any future encoding change
invalidates old vocabularies loudly instead of silently mismatching.

### 2.2 Canonical WL hashes without giving up the vectorized refinement

The fast path (`_wl_color_refinement`) stays untouched — its per-round color partitions are exactly
right, only their names are dataset-relative. Canonical hashes are attached **per color class, not per
node**, bottom-up over rounds:

1. Run the existing vectorized refinement, keeping the per-round color arrays (already computed
   internally; expose them).
2. Round 0: `canon[0][c] = H(seed)` where seed is the canonical init (constant for unlabeled WL, the
   base-label hash for `wl_labeled`).
3. Round r: for each unique color class `c`, pick one representative node `v` and compute
   `canon[r][c] = H(canon[r-1][color_{r-1}(v)] ‖ sorted multiset of canon[r-1][color_{r-1}(u)] for
   u ∈ N(v))`. Nodes in one class have identical multisets by definition of WL, so one representative
   suffices. Final label hash = `H(params ‖ canon[R][c])`.

This matches the partition the model already uses (bit-identical labels, unchanged
`num_unique_node_labels`), and reproduces the nx blake2b semantics as canonical identity — at
per-class rather than per-node cost.

**-1 / invalid labels** keep a reserved hash (never matched on import). **`max_labels` capping**: the
merged "other" bucket (`relabel_node_labels`, `:1650–1651`) is a dataset-relative mixture — its hash is
flagged non-canonical; all uncapped labels keep their true hashes.

### 2.3 Storage: a vocabulary sidecar, not a new runtime representation

The runtime keeps dense compacted ints everywhere — the hot paths (`_pv_*` int32 vectors, offset
arithmetic, `torch.unique` caches, forward gathers) are untouched. The hash lives in a small
**vocabulary**: per label file, a `(num_unique,)` int64 tensor mapping compacted id → hash, plus flags
(canonical yes/no, schema version). Saved next to / inside the existing
`<dataset>_labels_<name>.pt` (format v2: `(dataset_name, label_name, node_labels, label_hashes,
meta)`; loader stays backward compatible, old files simply have no vocabulary). `NodeLabels` gains
`label_hashes` and `has_canonical_hashes`.

Same idea for properties: distance keys (ints) and edge-label-distance keys (tuples,
`properties.py:218`) are already canonical values, no hashing needed — they are serialized as-is in
the manifest (edge-label distances are canonical only iff primary edge labels are, same caveat as
`primary`).

### 2.4 Weight export/import keyed by hash triples

This slots directly into spec 11's manifest + remap engine, replacing "original label values" as the
key space:

- **Export** (checkpoint sidecar): per invariant conv layer, one compact table per head config —
  `(src_hash: int64, tgt_hash: int64, property_key, replica) → Param_W slot`. The
  `(src_id, tgt_id)` per slot is reconstructed from the per-property `torch.unique` inverse: the
  unique encoded values are currently discarded (`inv_based_message_passing.py:424` keeps only
  `indices, counts`); either store the `uniques` in the fine-grained cache too (a few KB per key) or
  recover them via one scatter (`uniques[indices] = encoded_labels`). Ids → hashes via the
  vocabulary. Aggregation: `(hash, replica) → slot`; bias: `(hash, feature, head) → slot`.
  Prefer a binary `.keys.pt` sidecar over spec 11's `.keys.json` — hash keys are opaque to humans
  anyway and the table can hold millions of rows (spec 11 D3 revisited; keep a tiny human-readable
  `.yml` header with layer/head descriptions).
- **Import**: target layer builds normally on the target dataset, then a vectorized sort-merge join on
  `(src_hash, tgt_hash, prop, replica)` copies matched source weights into the fresh `Param_W`;
  unmatched slots keep their init (spec 11 §3.3 semantics, `on_missing` etc. unchanged). Non-canonical
  vocabularies (betweenness bins, capped buckets, per-dataset `primary`) are skipped or matched
  best-effort behind an explicit config flag.

**Bonus for multi-source pretraining (spec 11 §3.5):** with canonical hashes, label files computed
*independently per dataset* align by construction — a union label alphabet is just the union of
vocabularies. Joint pretraining no longer depends on concatenating datasets so that WL compaction runs
jointly, and Option B (sequential transfer) stops eroding: every hop matches against the same global
key space.

### 2.5 Rejected alternative: hashing as the runtime index

Indexing `Param_W` directly by hash (feature-hashing / `hash mod 2^b` embedding tables) was
considered and rejected:

- It replaces exact triples with **lossy buckets** — collisions silently tie unrelated structural
  roles together and are a quality, not just correctness, risk.
- It destroys the contiguous-offset layout the layer is built on (`weight_offset`, `_pv_*`/`_nw_*`
  replica arithmetic, occurrence-threshold filtering, the memory factorization of spec 10) and the
  sparse/dense forward caches.
- It buys nothing at runtime: the dense compaction is already optimal for gathers. Hashes are needed
  only at dataset-boundary crossings (vocabulary build, export, import), which is where this design
  puts them.

## 3. Runtime and resource impact

**Training / inference forward+backward: zero change.** No hash ever touches the hot path; the model
computes on the same compacted int ids and the same flat `Param_W` as today.

| Stage | Cost | Estimate (ZINC-full scale: ~250k graphs, ~5.8M nodes) |
|---|---|---|
| WL canonical hashing (preprocessing, once, cached) | O(rounds · E) gather + one blake2b call per (round, color class); classes ≤ nodes, typically ≪ | seconds to low tens of seconds on top of the existing refinement; dwarfed by current label generation, cached in the label file |
| Other labelings (closed walks, cycles, cliques, degree) | hash of an already-computed signature per unique label | negligible (µs per unique label) |
| Vocabulary storage | 8 B per unique label | ≤ 8 MB even at 10⁶ unique labels; typical: KBs |
| Export (once per checkpoint) | reconstruct uniques + gather hashes: O(total weights) | `sum(weight_num)` is 10⁵–10⁷ → sub-second to a few seconds; sidecar ~24 B/slot → ≤ ~250 MB worst case, typically MBs |
| Import (once per finetune run) | sort-merge join over source+target key tables | O((W_s + W_t) log) → seconds |
| Layer `__init__` / distribution build | unchanged (optionally +`uniques` in the fine-grained cache: KBs per key, no measurable time) | — |

Memory during preprocessing: the per-round color arrays needed for §2.2 are `rounds × N` int64
(~46 MB × rounds on ZINC-full) — bounded, and freeable round by round since hashing is bottom-up.

**Collision risk (the "minimal collisions" requirement):** with 64-bit hashes and V distinct labels
ever observed across all datasets, the birthday bound gives P(any collision) ≈ V²/2⁶⁵:
V = 10⁶ → ~3·10⁻⁸; V = 10⁷ → ~3·10⁻⁶; V = 10⁸ → ~3·10⁻⁴. Within one dataset, collisions are
**detected exactly** at vocabulary build time (two distinct signatures, same hash → hard error, since
the full signatures are in hand). Cross-dataset collisions are undetectable but bounded as above; a
collision merely initializes one weight from a wrong-but-random source value — no correctness impact
on training. If ever needed, the schema supports 128-bit (2×int64 columns) at double vocabulary/key
size; not the default.

## 4. Implementation phases (for a later session — not now)

1. **Vocabulary infrastructure (~1 day).** Stable hashing helper (versioned encodings); label file
   format v2 + backward-compatible loader; `NodeLabels.label_hashes`; hashes for the cheap signature
   labelings (degree, closed walks, cycles, cliques, trivial, combined).
2. **Canonical WL hashes (~1 day).** Expose per-round colors from `_wl_color_refinement`;
   per-class bottom-up hashing (§2.2); parity test against `nx.weisfeiler_lehman_subgraph_hashes` on
   small graphs; `wl_labeled` seeding incl. the primary-label canonicality flag.
3. **Export (~1 day).** Keep/recover per-property `uniques`; slot→triple table assembly; `.keys.pt`
   sidecar written next to the checkpoint (`model_configuration.py` save path); aggregation + bias
   tables.
4. **Import (~1 day).** Hash-join remap engine wired into spec 11's `transfer:` /
   `invariant_transfer:` flow (`match: hashes` joining `keys`/`none`); overlap telemetry; tests: two
   synthetic datasets sharing structure → full overlap for WL labels (the case spec 11 could not
   cover), disjoint → all reinit, collision-detection unit test.
5. **Validation (~0.5 day).** End-to-end: pretrain on NCI1, export, finetune DHFR with `wl_2` labels;
   assert overlap ≫ the original-value baseline of spec 11; regression `pytest tests -q`.

## 5. Open decisions

| ID | Decision | Recommendation |
|---|---|---|
| H1 | Hash width | **64-bit** default; schema reserves 128-bit. |
| H2 | Where vocabularies live | **Inside the label `.pt` (format v2)** — single artifact, cache-invalidation-free. Alternative sidecar file rejected: two files can desync. |
| H3 | `primary`/edge-label canonicality across molecular datasets | Out of scope here; a later mapping of TU atom enumerations → atomic numbers would upgrade `primary`, `wl_labeled`, and `edge_label_distances` to fully canonical. |
| H4 | Betweenness | Ship flagged non-canonical (dataset-relative bins); optional future: fixed global bin edges to make it canonical. |
| H5 | Relation to spec 11 manifest | Hash keys **supersede** original-value keys (`match: hashes`); keep `match: keys` as fallback for old label files without vocabularies. |
