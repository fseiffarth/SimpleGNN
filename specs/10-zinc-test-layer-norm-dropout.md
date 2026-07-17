# ZINC Test Network: Layer Normalization & Dropout Analysis

Status: **analysis only — no config or code changes applied.**

Target: `experiments/base_paper/regression/ZINC/configs/network_ZINC_test.yml`
(wired via `main_config_ZINC_test.yml`, run by `main_ZINC_test.py`).

## 1. Current state

Layer stack (norms/dropout highlighted):

```
linear(→32, LeakyReLU) → layer_norm → invariant_based_convolution
→ linear(aggr_channels →100, LeakyReLU) → layer_norm → invariant_based_aggregation
→ reshape                                  # graph-level embedding
→ linear(→100, LeakyReLU) → layer_norm
→ linear(→100, LeakyReLU)                  # ← no norm here
→ linear(→1, Identity) → reshape
```

- Three `layer_norm` layers (indices 1, 4, 8 in the YAML list).
- **No dropout anywhere** in the network.

## 2. Is the layer_norm implementation correct?

Source: `src/simplegnn/models/layers/nn_standard/layer_normalization.py`.

### No learnable weights

**`LayerNormalization` has NO learnable parameters.** It calls
`nn.functional.layer_norm(x, normalized_shape=[...])` without passing `weight`
or `bias`, and registers no `nn.Parameter` of its own. This differs from
`torch.nn.LayerNorm`, whose default (`elementwise_affine=True`) learns a
per-feature scale γ and shift β after normalization.

Is that a problem here?

- **Mathematically valid**: the normalization itself (mean 0, variance 1 over
  the feature axis) is computed correctly.
- **Where the missing affine is harmless**: a learnable affine
  `diag(γ)x + β` followed by a linear layer is fully absorbed by that linear
  (`W·(diag(γ)x + β) + c = (W diag(γ))x + (Wβ + c)`). The readout norm
  (index 8) is followed by a plain `linear`, so a learnable affine there would
  add nothing.
- **Where it may matter**: the norms at indices 1 and 4 feed the
  **invariant** convolution/aggregation layers, whose weights are sparse and
  structurally shared — they cannot express an arbitrary per-feature rescaling.
  There, the missing γ/β genuinely restricts the model compared to standard
  LayerNorm. Adding optional affine support would be a source change
  (lazy per-feature parameters, batched + per-graph paths, `precision`
  handling) — noted as possible future work, not part of this analysis.

### Batched vs. per-graph consistency (verified OK for this network)

- All three norm placements in this net receive 1D/2D tensors: the input
  linear and the `aggr_channels` linear output 2D `(N, F')`
  (`nn_standard/linear.py` contracts channels away), and the graph embedding
  after `reshape` is 1D per-graph / 2D batched. All of these normalize over
  the feature axis in **both** forward modes (the 1D graph-embedding case was
  fixed previously; covered by
  `tests/test_batched_share_gnn.py::test_layer_norm_on_graph_embedding_matches_per_graph`).
- **Caveat for other placements**: for node-level **3D** `(C, N, F)` input the
  per-graph forward normalizes over `(N, F)` jointly while the batched forward
  normalizes over `F` only — a `layer_norm` placed directly after an invariant
  convolution (before the channel-aggregating linear) would compute different
  functions in the two forward modes. The ZINC test net does not have a norm
  in that position, but keep this in mind before moving one there.

### Verdict

The implementation is **correct for how this network uses it**, but it is
*normalization-only* — there are **no layer-norm weights** (no learnable
γ/β), unlike standard `torch.nn.LayerNorm`.

## 3. Proposed changes (not applied)

Rationale: keep the message-passing trunk dropout-free (sparse shared weights,
tiny `lower_upper ±0.001` init; ZINC GNN baselines use dropout 0 in conv
layers). Regularize the dense 100→100→1 readout, which is where overfitting
happens; `specs/09` and the tuned `small`/`labels`/`factorized` variants use
`dropout p=0.1` before the readout.

Implementation constraints (verified in source):

- Dropout must be a **standalone** `{layer_type: dropout, p: …}` layer
  (`nn_standard/dropout.py`, default `p: 0.5`, train-only). A `dropout:` key on
  a `linear` layer is silently ignored, and the global `dropout:` parameter in
  `parameters_ZINC_test.yml` does not affect the ShareGNN layer stack.
- `layer_norm` takes no config options.

Readout section of `network_ZINC_test.yml` (currently lines 632–649), proposed:

```yaml
- layer_type: reshape
- layer_type: dropout        # NEW: dropout on the graph embedding
  p: 0.1
- layer_type: linear         # →100, LeakyReLU (unchanged)
  mode: aggr_features
  out_features: 100
  bias: true
  activation: torch.nn.LeakyReLU()
- layer_type: layer_norm
- layer_type: dropout        # NEW: between the two hidden readout layers
  p: 0.1
- layer_type: linear         # →100, LeakyReLU (unchanged)
  mode: aggr_features
  out_features: 100
  bias: true
  activation: torch.nn.LeakyReLU()
- layer_type: layer_norm     # NEW: normalize the 2nd hidden readout layer too
- layer_type: linear         # →1, Identity (unchanged)
  mode: aggr_features
  out_features: 1
  bias: true
  activation: torch.nn.Identity()
- layer_type: reshape
```

Summary — three insertions, nothing else changes:

1. `{layer_type: dropout, p: 0.1}` after the graph-level `reshape`.
2. `{layer_type: dropout, p: 0.1}` after the existing readout `layer_norm`.
3. `{layer_type: layer_norm}` after the second 100-unit readout linear.

The three existing `layer_norm` layers stay where they are (they stabilize
training given the tiny weight init). No changes to `parameters_ZINC_test.yml`.

## 4. Verification (when the changes are applied)

1. `python experiments/base_paper/regression/ZINC/main_ZINC_test.py --num_threads 4`
   from the repo root (venv active, `PYTHONPATH=src` as in `ZINC.sh`): confirm
   preprocessing + first epochs run and the new layers appear in the model.
2. `pytest tests -q` as a smoke check (no source changes involved).
