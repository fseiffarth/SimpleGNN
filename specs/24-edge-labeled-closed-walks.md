# 24 — Edge-labeled closed-walk node invariants

## Goal

Refine the existing `closed_walks` node labeling (diagonal entries of adjacency
matrix powers, `src/simplegnn/datasets/utils/node_labeling.py`) so that the walk
counts distinguish walks by the edge labels they traverse. Two new label types
are added; the existing `closed_walks` type is left untouched so cached label
files and running experiments are unaffected.

Edge labels are already available to the labelings: `GraphDataset.create_nx_graphs()`
attaches `primary_edge_labels` as a NetworkX edge attribute whenever the dataset
carries them (this is how `wl_labeled_edges` reads them).

## Notation

Split the adjacency matrix by edge label: `A = Σ_c A_c`, where `A_c` has a 1 for
every edge labeled `c`.

## Variant A — `closed_walks_edges` (multiset-refined, exact)

For a walk, record the *multiset* of edge labels traversed. Equivalently, work
over the commutative polynomial ring with one variable `x_c` per edge label:
entry `(i,j)` of the labeled adjacency holds `x_c` for an edge of label `c`, and
the diagonal of the `k`-th power is a polynomial whose coefficients count closed
walks of length `k` per label multiset.

Implementation: a dict `monomial -> n x n int64 matrix`, where `monomial` is a
sorted tuple of label values. Step from length `k-1` to `k` by multiplying every
tracked monomial matrix with every `A_c` and accumulating into the merged key.

- Profile size per node: `C(k + |L| - 1, k)` entries (ZINC: `|L|=3`, `k=8` → 45),
  versus `|L|^k` for the fully word-refined variant.
- Cost: `O(k · n³ · #monomials)` per graph, memory `#monomials · n²` int64.
  Fine at molecular scale; expensive for large graphs.
- Strictly finer than unlabeled `closed_walks`; isomorphism-invariant.

Node signature: sorted tuple of `((length, monomial), count)` over the nonzero
entries only, so the signature is independent of which labels happen to occur in
the dataset and is directly usable as a canonical hash signature.

## Variant B — `closed_walks_weighted` (weighted, cheap, lossy)

Assign a distinct prime to each edge label (sorted by label value, smallest
primes first), build the single weighted matrix `A_w`, and take `(A_w^k)_ii`.
The weight of a walk is the product of the primes along it, so different label
multisets carry different weights — but the diagonal entry is a *sum* over
walks, and distinct multisets can still sum-collide. Lossy by design; same
`O(k · n³)` cost as the unlabeled version.

Computed in float64 (BLAS-fast) with a hard guard that no entry exceeds `2^53`,
so all values are exactly representable integers; the guard raises with a
pointer to `closed_walks_edges` / a smaller `max_walk_length` rather than
silently returning rounded counts.

Because the label → prime assignment depends on the dataset's edge-label
vocabulary, that assignment is folded into the hash params, so a weighted label
from one vocabulary can never alias one from another.

## Transfer caveat

Both variants are only canonical across datasets that share an edge-label
vocabulary (ZINC bond types and TU edge labels do not). They belong with the
primary-label family, not with the structural labels used for cross-dataset
transfer.

## Touch points

1. `node_labeling.py`: `ClosedWalkEdgeLabeledNodeLabeling`,
   `ClosedWalkWeightedNodeLabeling`, plus `save_closed_walk_edges_labels` and
   `save_closed_walk_weighted_labels` mirroring `save_closed_walk_labels`.
2. `node_labeling.py::get_label_string`: branches for both types, so the model
   layers resolve the same filenames the save functions write.
3. `ShareGNN/preprocessing/preprocessing.py`: dispatcher branches.
4. `tests/test_node_labeling_helpers_unit.py`: label-string, correctness and
   file-writing tests.

## Config

```yaml
labels:
  label_type: closed_walks_edges   # or closed_walks_weighted
  min_walk_length: 2
  max_walk_length: 6
  max_labels: 100                  # optional
```

Filenames: `closed_walks_edges[_<min>]_<max>[_<max_labels>]` and
`closed_walks_weighted[_<min>]_<max>[_<max_labels>]`, mirroring `closed_walks`.
