# 15 — Invariant-Based Positional Encoding + Pre-Norm Residual for the Invariant Conv

**Status update (2026-07-17, user decision)**: the PE head interface was
simplified after implementation — `dim` is removed. A head now has only
`num` = learned entries per label value (identical semantics to `num` on
aggregation heads): each node gets `num` weights tied to its invariant value,
a head contributes `num` features per node, and the layer outputs one
concatenated vector per node (`pe_dim = Σ num_h`). The (num, dim) split was
redundant for a lookup table (`num·dim` free scalars either way) and only
mattered for replica-structured pruning, which the aggregation-style `num`
covers. The label-tied scalar entries also multiply directly with the conv's
label-pair weights in the following message passing. YAML snippets below that
show `dim` predate this simplification.

**Status**: Implemented (layer + registration + pre-norm residual + tests in
`tests/test_positional_encoding.py`; full suite green, MUTAG end-to-end smoke
run passed). Note: once `weight_initialization` exists in the parameter config,
the conv layer (pre-existing behavior) requires entries for its own init types
(`convolution`, `convolution_bias`) — the `positional_encoding` entry alone is
not enough; the PE layer itself falls back to `normal(0, 0.1)` when its entry
is missing.
**Scope**: new layer `src/simplegnn/models/ShareGNN/layers/inv_based_positional_encoding.py`,
registration plumbing, and a transformer-style pre-norm residual option for
`inv_based_message_passing.py`.

---

## Context

Two related requests from the ZINC work:

1. **Invariant-based positional encoding.** ZINC currently feeds only the
   one-hot atom type (`input_features: {name: node_labels, transformation:
   one_hot}`). Structural/positional encodings (RWSE, LapPE) are the proven
   input-side win on ZINC, but ShareGNN already computes rich node invariants
   (induced/simple cycles, WL labels) during preprocessing. Idea: reuse them —
   in the spirit of `invariant_based_aggregation`, give each node an **ID from
   its invariant value** and map each ID to a **learnable embedding**. Multiple
   heads, each head on a different label type, each with its own embedding
   table. The embeddings are **concatenated** to the incoming node features
   (decision: concat, not sum — keeps the atom-type signal intact; a following
   linear layer mixes them).

2. **Residual connections.** Analysis result (see below): residuals are **not**
   redundant with self-loops in the current ZINC setup, because distance 0 is
   excluded from every conv head's `values` — a node's own feature never
   reaches its own output, and the conv bias does not depend on `x`. Decision:
   add a **transformer-style pre-norm block** to the invariant conv — optional
   LayerNorm on the layer input, with the residual wrapping (LayerNorm +
   message passing): `out = x̃ + Conv(LN(x))`.

## Residual vs. self-loop analysis (recorded for reference)

- `set_weights` only populates `current_W[h, i, j]` for `(i, j)` pairs produced
  by the configured property values (`inv_based_message_passing.py:1110-1114`);
  forward is `activation(current_W @ x + current_B)`.
- The distance property files **do contain distance 0** (self-pairs) —
  `preprocessing/properties.py` uses `nx.all_pairs_shortest_path_length`, which
  includes 0 — so `values: [0, 1, ...]` is a valid config that creates a
  learned, label-pair-conditioned diagonal weight (a generalized self-loop).
  The ZINC config uses `values: [1..23]`, so no diagonal exists today.
- The `residual` flag already exists in `FrameworkLayer.__init__`
  (`framework_layer.py:251`) and is honored by the classical MPNN wrappers
  (e.g. `gcn_conv.py:29-30`, `gat_conv.py:39-43`) but ignored by the invariant
  layers.
- Output layout of the conv is `(H, N, F).permute(1, 2, 0).flatten(1)` →
  column index `f*H + h`. A feature-aligned residual therefore uses
  `x.repeat_interleave(num_heads, dim=1)` (NOT `x.repeat(1, H)`, which matches
  the PyG `h*F + f` layout used in the GAT wrapper).

---

## Part 1 — `InvariantBasedPositionalEncodingLayer`

New file: `src/simplegnn/models/ShareGNN/layers/inv_based_positional_encoding.py`
Subclasses `InvariantBasedLayer` (`inv_based.py`), mirroring
`InvariantBasedAggregationLayer` (`inv_based_pooling.py`) for label resolution,
factored index buffers, `init_weights`, and batched forward conventions.

### YAML interface

```yaml
- { layer_type: invariant_based_positional_encoding,
    dim: 8,                      # layer-wide default embedding dim (optional, default 8)
    heads: [
      { num: 1, dim: 8, labels: { label_type: induced_cycles, min_cycle_length: 5, max_cycle_length: 10 } },
      { num: 1,         labels: { label_type: wl_labeled, depth: 0 } },   # dim falls back to layer default
    ],
  }
```

- Aggregation-style head config: `labels: {label_type: ...}` (no head/tail —
  `LayerHead` already falls back to the whole labels dict as `source_labels`).
- `num`: independent replica embedding tables per head (default 1).
- `dim`: per-head embedding dimension, falling back to the layer-level `dim`.
- No bias (an embedding table is already a free parameter per ID).
- `concatenate_input: True` (default) → output `[x | emb_head1 | emb_head2 |…]`,
  shape `(N, F_in + Σ_h num_h·dim_h)`. With `False` the embeddings alone are
  returned (usable as a pure input layer).

### Mechanics (mirrors `inv_based_pooling.py:58-95`)

- Per head: `desc = layer.get_source_string(head_id)`; per-node labels from
  `graph_data.node_labels[desc].node_labels`;
  `torch.unique(..., return_inverse=True)` → per-node ID; register as
  non-persistent int32 buffer `_pe_idx_{head_id}`.
- One flat `Param_W` of length `Σ_h num_h · n_labels_h · dim_h`, created by an
  `init_weights(total, init_type='positional_encoding')` (same structure as
  the pooling `init_weights`, reading
  `weight_initialization: { positional_encoding: {...} }` from the parameter
  config). Fallback when unconfigured: `normal(0, 0.1)` — a constant fallback
  would make all embeddings identical (uninformative at init), so the pooling
  layers' `constant 0.01` fallback is deliberately not reused.
- Row layout for head h, replica k, node-ID i:
  `offset_h + k·n_h·d_h + i·d_h  …  + d_h`. Forward gathers
  `(N_sel, num_h, d_h)` via broadcasted index arithmetic, reshapes to
  `(N_sel, num_h·d_h)`, concatenates heads, then concatenates onto `x`.
- Buffers `_pe_x_slices` (clone of `graph_data.slices['x']`) + Python-int
  `_pe_slices` for node ranges; single-graph forward uses
  `slice(_pe_slices[pos], _pe_slices[pos+1])`, batched forward
  (`is_batched_pos(pos)`) uses `range_gather(self._pe_x_slices, positions)` —
  a per-node op, so single and batched paths share the same embedding gather.
- `out_features = in_features + Σ_h num_h·dim_h` (or just the sum when
  `concatenate_input: False`), `out_channels = 1`, set in `__init__`
  (overrides the generic values, same pattern as the conv layer at
  `inv_based_message_passing.py:225-228`). Input must be 2-D `(N, F)`.

### Registration (integration checklist)

1. `src/simplegnn/models/ShareGNN/utils.py`
   - `Layer.__init__` (line 140): add `'invariant_based_positional_encoding'`
     to the layer-type list so `layer_heads` populate → label preprocessing
     (`layer_to_labels`) and loading (`framework/utils/preprocessing.py:727-747`)
     work unchanged.
   - `LayerHead.__init__`: parse `self.dim = info_dict.get('dim', None)`.
2. `src/simplegnn/models/layers/utils/layer_types.py`: enum value
   `INVARIANT_BASED_POSITIONAL_ENCODING` already exists — no change.
3. `src/simplegnn/models/layers/utils/layer_loader.py`
   - `layer_from_yml` (line 28): route the new type through
     `layer_from_yml_invariant_based` (validation path).
   - `check_layer`: new elif — require `heads` list; each head a dict with
     `labels.label_type`; `num`/`dim`/`bias` optional.
4. `src/simplegnn/models/model.py`
   - import the new class next to the other ShareGNN imports;
   - `get_model_layer`: new elif instantiating
     `InvariantBasedPositionalEncodingLayer(layer=..., parameters=self.para,
     graph_data=self.graph_data).to(self.precision)` (trainable by default).

---

## Part 2 — Pre-norm residual block for `invariant_based_convolution`

File: `src/simplegnn/models/ShareGNN/layers/inv_based_message_passing.py`

New config keys on the conv layer (both off by default, so existing configs
are bit-identical):

```yaml
- { layer_type: invariant_based_convolution,
    residual: True,          # out = skip(x) + conv-block(x)
    pre_layer_norm: True,    # conv-block(x) = Conv(LayerNorm(x)); residual wraps both
    ... }
```

- `__init__`: `self.pre_layer_norm = layer.layer_dict.get('pre_layer_norm', False)`;
  if set, `self.pre_norm = nn.LayerNorm(self.in_features)` (moved to the right
  dtype by the model's `.to(self.precision)`).
- `forward` (single-graph path, `:1240-1285`) and `_forward_batched`
  (`:1322-1353`): keep `x_in`, apply `self.pre_norm` to the input when enabled,
  run the existing message passing + activation unchanged, then
  `out = out + x_in.repeat_interleave(self.num_heads, dim=1)` when
  `self.residual` — feature-aligned with the `f*H + h` output layout.
- `self.residual` is already parsed by `FrameworkLayer`; no config plumbing
  needed. `check_layer` ignores unknown keys, so no validator change needed.
- The aggregation layer gets no residual (node→graph pooling has no identity
  path to preserve).

---

## Implementation notes (verified against code before implementation)

- Label preprocessing AND loading are both driven by `layer.layer_heads`:
  generation in `ShareGNN/preprocessing/preprocessing.py:225-244`
  (`get_unique_layer_dicts` per layer), loading in
  `framework/utils/preprocessing.py:727-747`. Heads without `properties` are
  safe everywhere: `PropertyDict(None).property_dict is None` is filtered in
  `get_unique_property_dicts` (utils.py:166) and in the property-loading loop
  (preprocessing.py:753). So the ONLY switch needed for preprocessing is the
  layer-type list in `Layer.__init__` (`ShareGNN/utils.py:140`).
- `get_model_layer` (model.py:440-464) generically sets
  `num_heads = sum(head['num'])` and `out_channels = in_channels * num_heads`
  from the `heads` key; the new layer must override `self.out_features` /
  `self.out_channels` in its own `__init__` (same pattern as the conv layer,
  inv_based_message_passing.py:225-228).
- Test scaffolding: `tests/conftest.py` provides `mutag_main_config` and
  `share_gnn_setup_factory(models_file=...)` — preprocesses real MUTAG and
  returns `(graph_data, para)` ready for `GraphModel(graph_data, para, seed,
  device='cpu')`. Add a new fixture model YAML under
  `tests/fixtures/share_gnn_mutag/` (e.g. `models_ShareGNN_pe.yml`, copy of
  `models_ShareGNN.yml` with a PE layer inserted and
  `residual/pre_layer_norm: True` on the conv) and pattern the new test after
  `tests/test_share_gnn_mutag_integration.py` /
  `tests/test_batched_share_gnn.py`.
- `check_layer` (layer_loader.py:158) only checks required keys per type —
  extra keys like `residual`/`pre_layer_norm` pass through untouched, so only
  the new PE elif branch is needed there, plus routing at line 28.
- `init_weights` to copy: `inv_based_pooling.py:108-153` (accepts int or
  shape tuple, reads `weight_initialization[<init_type>]` from
  `para.run_config.config`).

## Verification

1. `pytest tests -q` — full suite must stay green (no existing config uses the
   new keys, so behavior must be unchanged).
2. New test `tests/test_positional_encoding.py` (pattern after the existing
   ShareGNN tests/fixtures): build a small experiment whose model config
   includes an `invariant_based_positional_encoding` layer; check
   (a) layer builds, out_features = F_in + Σ num·dim;
   (b) forward output shape/finiteness on a real graph;
   (c) single-graph vs batched forward rows agree;
   (d) two nodes with the same invariant ID get identical embedding slices,
       different IDs (generically) different ones;
   (e) conv layer with `residual: True, pre_layer_norm: True` — output equals
       `conv(LN(x)) + repeat_interleave(x)` against a manual computation, and
       gradient flows.
3. End-to-end smoke run: copy of `examples/share_gnn_basic` (or the fastest
   existing example) config in the session scratchpad with a PE layer inserted
   before the first linear layer + a conv with `residual/pre_layer_norm` —
   `preprocessing()` + a couple of training epochs must run without errors.
   User-facing ZINC configs are NOT modified (mid-experiment).

## Usage note for ZINC (not applied automatically)

To try it on ZINC, insert before the first linear layer of
`examples/zinc/models_ShareGNN.yml`:

```yaml
- { layer_type: invariant_based_positional_encoding,
    heads: [
      { num: 1, dim: 8, labels: { label_type: induced_cycles, min_cycle_length: 5, max_cycle_length: 10 } },
      { num: 1, dim: 8, labels: { label_type: wl_labeled, depth: 0 } },
    ],
  }
```

and optionally add `positional_encoding: { type: 'normal', mean: 0.0, std: 0.1 }`
under `weight_initialization` in `parameters.yml`. Then `residual: True,
pre_layer_norm: True` on the conv layer replaces the free-standing
`layer_norm` before it (pre-norm block).
