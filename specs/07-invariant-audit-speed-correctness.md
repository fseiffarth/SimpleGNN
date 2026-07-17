# Invariant Audit: Speed and Correctness (2026-07-17)

Audit of every node-label invariant (`src/simplegnn/datasets/utils/node_labeling.py`)
and pairwise-property writer (`src/simplegnn/models/ShareGNN/preprocessing/properties.py`)
for (a) correctness and (b) speedups over the networkx-based implementations.
All numbers measured on seeded ER graphs at molecular scale (n≈15–35, p=0.15) in
this repo's venv (networkx 3.6.1). Regression tests: `tests/test_invariant_correctness.py`
(confirmed bugs are strict `xfail` tests — they flip to XPASS when fixed).

## Correctness findings

### Bugs

1. **Cycle and clique labels are not isomorphism-invariant** (`save_cycle_labels`,
   `save_clique_labels`). The per-node counts dict is stringified in *discovery
   order* of the cycles/cliques, and `str({3: 1, 4: 1}) != str({4: 1, 3: 1})`, so
   structurally identical nodes get different labels depending on node/edge
   numbering. Verified end-to-end (triangle+square hub under two edge numberings:
   labels 2 vs 0; clique hub flips in 8/24 node permutations). Also silently
   inflates the label vocabulary, hurting ShareGNN weight sharing.
   **Fix**: canonicalize before hashing, e.g. `str(sorted(d.items()))`.
   Tests: `test_cycle_labels_do_not_depend_on_edge_numbering`,
   `test_clique_labels_do_not_depend_on_node_numbering`.

2. **`LabeledDegreeNodeLabeling` (class, not the save function) is broken**: a
   single `node_to_hash` dict is shared across graphs, so later graphs overwrite
   the hashes of earlier graphs with the same node ids; neighbor labels are also
   not sorted (order-dependent). The pipeline uses `save_labeled_degree_labels`,
   which is correct — only the class is affected (classes are currently unused).
   Test: `test_labeled_degree_class_no_cross_graph_contamination`.

3. **`save_cycle_labels` crashes on datasets with < 10 graphs**: progress print
   does `i % (len(graphs) // 10)` → `ZeroDivisionError`.
   Test: `test_save_cycle_labels_works_for_small_datasets`.

4. **Filename mismatches between save functions and `get_label_string`** (the
   loader in `framework/utils/preprocessing.py` builds the path from
   `get_label_string`, so a mismatch = `FileNotFoundError` at training time):
   - cycles with `max_cycle_length` omitted: loader expects `simple_cycles_max`,
     save writes `simple_cycles_None` (current configs always set it — edge case);
   - `wl_labeled` with explicit `base_labels: {label_type: primary}`: loader
     expects `wl_labeled_primary_base_labels_3`, save writes `wl_labeled_3`
     (omitting `base_labels` works fine).
   Tests: `test_cycle_filename_without_max_matches_get_label_string`,
   `test_wl_labeled_primary_base_filename_matches_get_label_string`.

5. **Betweenness binning collapses on skewed distributions**
   (`BetweennessCentralityNodeLabeling`): percentile bin edges are not
   deduplicated, so on e.g. a star graph (80 % of nodes have centrality 0, all
   percentile edges are 0) every node — including the center — lands in one bin.
   **Fix**: `np.unique` on the bin edges. Test:
   `test_betweenness_labels_order_star_center_above_leaves`.

6. **`write_distance_edge_properties` label handling**: `label_occurrences` is
   indexed by the raw edge-label value — negative labels crash with `IndexError`,
   large label ids allocate huge tuple keys (label 10^6 → 10^6-element key);
   the `isinstance` check above is dead code (never raises), so float labels are
   silently truncated. Test: `test_edge_label_distance_rejects_or_handles_negative_labels`.

7. **WL labels depend on the networkx version**: nx ≥ 3.5 changed WL hash
   initialization for unlabeled graphs (trivial init instead of degree init), so
   `wl_3` computed today ≡ 2 refinement rounds from degree init, and label files
   cached before an nx upgrade are inconsistent with freshly computed ones.
   Delete cached `*_labels_wl_*.pt` when changing the nx major/minor version.

### Design smells (not bugs, but likely not the intended semantics)

- **`save_clique_labels`** counts only *maximal* cliques and drops maximal
  cliques larger than `max_clique` entirely: every K5 node with `max_clique=4`
  falls to the fallback label even though it lies in four K4s and six triangles;
  the densest nodes become indistinguishable. If "count cliques up to size k" is
  intended, enumerate all cliques (`nx.enumerate_all_cliques` with cutoff).
- **`save_subgraph_labels`** uses *induced* subgraph isomorphism
  (`subgraph_isomorphisms_iter`), so path/star patterns get **zero** matches
  inside denser regions (P3 in a triangle: 0 induced matches, 6 monomorphisms),
  and counts are inflated by |Aut(pattern)| (constant per pattern, so the
  partition is unaffected). Non-induced counting = `subgraph_monomorphisms_iter`.
- **`wl` vs `wl_labeled` depth semantics differ**: unlabeled uses
  `iterations=depth`, labeled uses `iterations=depth+1` — a `depth: 3` config
  means different refinement depths for the two types.
- **Self-pair asymmetry**: `write_distance_properties` includes distance-0 self
  pairs, `write_distance_edge_properties` excludes them.
- Label position corresponds to `graph.nodes()` *insertion order*, not node id.
  Safe in-pipeline (`to_networkx` inserts 0..n-1), but any hand-built nx graph
  with different insertion order silently misaligns labels.

### Verified correct

- `nx.simple_cycles` / `nx.chordless_cycles` usage: `length_bound` inclusive,
  each undirected cycle yielded exactly once, no 2-cycles — semantics match the
  config parameter names.
- Distance-property `slices_dict` cumulative bookkeeping (tested in
  `test_write_distance_properties_pairs_and_slices`).
- The multi-shortest-path aggregation in `write_distance_edge_properties`
  (tested: C4 diagonal → key `(2, 2, (4,))`).
- Degree, WL (invariance, refinement of degree, no cross-graph contamination),
  in-circle, subgraph marking — all covered by positive tests.

## Speed findings (faster than networkx)

| Invariant | Current | Faster approach | Speedup | Exactness |
|---|---|---|---|---|
| degree | nx per-node loop | `np.bincount` on edge arrays | **~190x** | identical |
| WL | nx WL hashes on `disjoint_union_all` | vectorized color refinement on flat edge arrays (`wl_fast2`, below) | **13–35x** (2k–10k graphs; 30–54 % of nx cost is `disjoint_union_all` itself) | exact partition match, labeled + unlabeled, depth 1–5 |
| distance properties | nx BFS + per-pair Python bucketing with per-pair `.item()` | scipy `csgraph.shortest_path` + vectorized `np` bucketing | **~7x** | identical pair sets |
| edge-label distances | `all_pairs_all_shortest_paths` + `deepcopy` (29 % of runtime!) + per-pair loops | BFS shortest-path-DAG counting with label-count DP | **6.7x** compute (2.5x end-to-end; rest is gzip/yaml serialization) | identical keys, pairs, slices |
| betweenness | nx serial | joblib over graphs (framework already ships joblib) | ~2x (overhead-bound on tiny graphs; not a hotspot: 3.9 s / 2000 graphs) | identical |
| simple/induced cycles | nx enumeration | inherently exponential in `max_cycle_length` (p=0.15, n=25: bound 6 → 2.3 s/200 graphs; bound 10 → 25 s; bound 20 → ~40 min). Embarrassingly parallel per graph → joblib gives ~n_cores, bit-identical | n_cores | identical |
| closed walks | — | already numpy (matrix powers) | — | — |

Notes: nx BFS itself is *not* slow on molecular-size graphs (scipy ≈ nx for the
distances themselves); the real cost sits in the Python pair-bucketing loops,
`deepcopy`, and per-pair tensor `.item()` calls. The biggest single win for WL
is skipping the `disjoint_union_all` copy and reading edges straight from the
PyG `edge_index` tensors (the nx graphs are themselves converted *from* those).

## Reference implementation: exact vectorized WL (`wl_fast2`)

Validated against `weisfeiler_lehman_node_labeling` (exact partition equality,
500 graphs, depth ∈ {1,2,3,5}, labeled and unlabeled). Input: disjoint-union
numbered `edge_src`/`edge_dst` (both directions per undirected edge), total node
count. Unlabeled: trivial init, `depth` rounds (mirrors nx ≥ 3.5); labeled:
init = int labels, `depth+1` rounds (mirrors the repo's `iterations=depth+1`).

```python
import numpy as np

def wl_core(edge_src, edge_dst, num_nodes, colors, rounds):
    N = int(num_nodes)
    colors = np.unique(np.asarray(colors, dtype=np.int64), return_inverse=True)[1]
    if N == 0 or rounds <= 0:
        return colors
    src = np.asarray(edge_src, dtype=np.int64)
    dst = np.asarray(edge_dst, dtype=np.int64)
    deg = np.bincount(dst, minlength=N)
    order = np.argsort(dst, kind="stable")
    dst_s, src_s = dst[order], src[order]
    start = np.concatenate([[0], np.cumsum(deg)[:-1]])
    node_order = np.argsort(deg, kind="stable")
    deg_sorted = deg[node_order]
    dmax = int(deg.max())
    C = int(colors.max()) + 1
    for _ in range(rounds):
        # per-node sorted neighbor colors via one sort of the packed key
        key = dst_s * np.int64(C) + colors[src_s]
        key.sort()
        neigh_sorted = key - dst_s * np.int64(C)
        # iterated pairwise folding in a strictly growing id namespace
        acc = colors.astype(np.int64, copy=True)
        A = C
        for k in range(dmax):
            i0 = np.searchsorted(deg_sorted, k + 1)
            sel = node_order[i0:]
            if sel.size == 0:
                break
            code = acc[sel] * np.int64(C) + neigh_sorted[start[sel] + k]
            uq, inv = np.unique(code, return_inverse=True)
            acc[sel] = A + inv
            A += uq.size
        uq, colors = np.unique(acc, return_inverse=True)
        if uq.size == C:          # partition stable -> stays stable
            break
        C = uq.size
    return colors

def wl_fast2(edge_src, edge_dst, num_nodes, depth, init_labels=None):
    if init_labels is None:
        colors = np.zeros(num_nodes, dtype=np.int64)   # nx >= 3.5 trivial init
        rounds = depth
    else:
        colors = np.asarray(init_labels, dtype=np.int64)
        rounds = depth + 1                             # mirrors repo labeled variant
    return wl_core(edge_src, edge_dst, num_nodes, colors, rounds)
```

## Implementation status (2026-07-17)

Items 1–5 of the recommended order are implemented; all former xfail tests in
`tests/test_invariant_correctness.py` are now positive tests.

- Bugs 1–6 fixed: cycle/clique/subgraph count dicts are canonicalized via
  `_canonical_count_string` (sorted items); the small-dataset progress-print
  crash is gone (per-graph loop replaced); betweenness binning deduplicates
  percentile edges (`np.unique` + `right=True` digitize); the
  `LabeledDegreeNodeLabeling` class uses per-graph hash dicts with sorted
  neighbor labels; filenames now match `get_label_string` (cycles append
  `_max` when the bound is omitted — the loader-side string also keeps the
  `min` prefix now — cliques drop `_None`, `wl_labeled` omits explicit primary
  base labels on the loader side); `write_distance_edge_properties` rejects
  negative/non-integer edge labels with a `ValueError`.
- WL labels use vectorized color refinement (`_wl_color_refinement` in
  `node_labeling_functions.py`) fed by flat edge arrays built directly from
  the nx graphs — no `disjoint_union_all`, no string hashing. Verified exact
  partition match vs the old nx path (depths 1/2/3/5, labeled + unlabeled);
  measured 13.6x on 2000 molecular-scale graphs. `with_edge_labels=True`
  still uses the nx fallback (`_weisfeiler_lehman_node_labeling_nx`).
- `write_distance_properties` uses scipy `csgraph.shortest_path` + vectorized
  bucketing; `write_distance_edge_properties` uses one BFS per source with a
  shortest-path-DAG label-count DP (`_edge_label_distance_keys`) instead of
  `all_pairs_all_shortest_paths` + `deepcopy`. Byte-identical keys, pair sets,
  and slices verified against the old implementations (with and without
  cutoff).
- Cycle/clique/in-circle per-graph loops go through `_adaptive_parallel_map`:
  a few graphs are probed serially and joblib only kicks in when the projected
  remaining serial time exceeds ~10 s (a fixed graph-count threshold made
  cheap workloads slower — worker spawn + graph pickling dominate there).
  Parallel and serial paths verified bit-identical.
- Item 6 (clique all-vs-maximal, subgraph induced-vs-monomorphism, wl depth
  semantics) is deliberately untouched: these change labels and are modeling
  decisions.

## Recommended order of work

1. Fix the invariance bug (1) — one-line canonicalization, affects results.
2. Fix crash (3) and the betweenness binning (5) — trivial.
3. Vectorize the two distance-property writers (biggest preprocessing wall-clock
   win; reference implementations validated in this audit).
4. Adopt `wl_fast2` for WL labels, feeding edges from `edge_index` directly.
5. joblib-parallelize the cycle/clique per-graph loops.
6. Decide intended semantics for cliques (all vs maximal) and subgraph
   (induced vs monomorphism) — these change labels, so treat as modeling choices.
