# Optional Degree Normalization for the Invariant Convolution

**Status: implemented** (2026-07-17)

Follow-up to specs/13 ("degree-normalized aggregation … would likely close
most of the gap to GCN on citation networks"). The invariant-based message
passing layer aggregates neighbor features with unnormalized learned weights;
on graphs with skewed degrees (citation networks), hub nodes dominate the
sums and generalization from few labeled nodes suffers.

## Configuration

Per-layer option on `invariant_based_convolution`:

```yaml
- { layer_type: invariant_based_convolution,
    degree_normalization: symmetric,   # or 'row'; omit for old behavior
    ... }
```

- `row`: `w[h,i,j] /= deg_h(i)` — mean aggregation per head.
- `symmetric`: `w[h,i,j] /= sqrt(deg_h(i) * deg_h(j))` — GCN-style.

The "degree" is the nonzero-pattern count of the *head's own weight matrix*:
distance-k pairs count, and a distance-0 self loop counts as well. Heads with
different property values therefore get different normalizations, which
generalizes the classical A+I normalization (property values `[0, 1]`
reproduce it exactly).

## Implementation

`InvariantBasedMessagePassingLayer._degree_norm_scale()` computes a
per-nonzero factor from the assembled `(head, i, j)` rows via `bincount` over
batch-global `(node, head)` keys — so counts never mix across graphs of a
batch or across heads. The factor is multiplied into the gathered `Param_W`
values in all four execution paths:

- per-graph sparse forward (factors cached in `_sparse_row_cache` — they are
  weight-independent, so single-graph node tasks pay nothing per step),
- per-graph dense (`set_weights`),
- batched sparse (`_batched_sparse_messages`),
- batched dense (`_batched_dense_messages`).

The stale, never-functional config options `degree_matrix` /
`use_in_degrees` now point to this option in their error message.

Tests: `tests/test_degree_normalization.py` — semantics against a dense
reference, cross-path equivalence (all 4 paths × both modes), gradient flow,
config validation.

## Interaction with training hyperparameters (important)

Normalization divides each weight's contribution — and therefore its
gradient — by ~`deg` (up to ~45 on Cora with distances ≤ 2). Two practical
consequences, both observed on Cora:

1. **Weight decay dominates**: with `weight_decay: 0.0005` the conv weights
   decay faster than their (now tiny) gradients restore them; validation
   accuracy collapsed over training (best 31.8%, final 18.4%). Remove/reduce
   weight decay when training normalized conv weights (63.6% without).
2. **Frozen conv is a strong baseline**: `convolution_grad: False` +
   `constant` weight initialization + `degree_normalization: symmetric`
   makes the layer exactly an SGC-style fixed propagation
   `D^-1/2 P D^-1/2 x` over the distance-≤2 pattern. On Cora this reaches
   **78.2% validation** (manual ceiling check: 77.6%), vs ~64% for the best
   trainable-normalized run and ~67% for the unnormalized learned model.

The Cora example (`examples/node_classification/`) now ships the
normalized + frozen-conv configuration.

## Follow-ups

- Per-layer (not global) `convolution_grad`, or per-parameter-group learning
  rates, so normalized conv weights can be trained jointly with a readout
  without the gradient-scale mismatch.
- Optional normalization for `invariant_based_aggregation` (graph-level
  readout means over nodes).
