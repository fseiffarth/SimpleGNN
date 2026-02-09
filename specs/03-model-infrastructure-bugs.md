# Model Infrastructure Bugs and Inconsistencies

## Critical Bugs

### 1. Undefined Variable in ShareGNNLinear

**Location:** `src/models/ShareGNN/layers/share_gnn_linear.py:34`

**Bug:** Variable `out_features` is referenced but never defined as a parameter:

```python
self.out_features = graph_data.num_node_features
if out_features is not None:    # NameError: 'out_features' is not defined
    self.out_features = out_features
self.out_features = layer.layer_dict.get('out_features', self.in_features)  # Line 36 overrides anyway
```

**Additional issue on line 36:** Fallback is `self.in_features` instead of `self.out_features`.

**Fix:** Remove lines 33-35 (dead code since line 36 overrides). Or add `out_features` as a constructor parameter if the intent was to allow overrides.

---

### 2. Undefined Variables in GraphModel.get_model_layer

**Location:** `src/models/model.py:126-131`

**Bug:** When instantiating `ShareGNNLinear`, the code references `layer_id` and `in_features` which are not defined in this scope:

```python
return ShareGNNLinear(layer_id=layer_id,    # undefined
                     seed=self.seed,
                     layer=layer,
                     parameters=self.para,
                     graph_data=self.graph_data,
                     in_features=in_features)  # undefined
```

**Fix:** Extract `layer_id` and `in_features` from `layer_args` dict before this point, similar to how other branches handle it.

---

### 3. Type Mismatch in Unique Property Dict Check

**Location:** `src/models/ShareGNN/utils.py:131-134`

**Bug:** Checks if `head.property_dict.property_dict` (a dict) is in the list, but appends `head.property_dict` (a PropertyDict object):

```python
if head.property_dict.property_dict not in unique_dicts and head.property_dict.property_dict is not None:
    unique_dicts.append(head.property_dict)  # Appends PropertyDict, not dict!
```

Subsequent `in` checks will never match because they compare `dict` against `PropertyDict` objects.

**Fix:** Append `head.property_dict.property_dict` (the raw dict) consistently, or implement `__eq__` on `PropertyDict`.

---

### 4. Security: eval() for Activation Functions

**Location:** `src/models/layers/framework_layer.py:91`

**Bug:** Activation functions from YAML config are instantiated via `eval()`:

```python
self.activation = eval(layer_args.get('activation', 'torch.nn.Identity()'))
```

A malicious config could execute arbitrary code.

**Fix:** Use a whitelist mapping:
```python
ACTIVATION_MAP = {
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'identity': torch.nn.Identity(),
    ...
}
```

---

### 5. Dead Code in GraphModel.forward

**Location:** `src/models/model.py:75`

**Bug:** `continue` statement makes lines 76-79 unreachable:

```python
for i in range(len(self.net_layers)):
    x = self.net_layers[i](x, batch_data, *args, **kwargs)
    continue                          # Makes everything below unreachable
    representation_list.append(x)     # Never executed
    ...
```

**Fix:** Remove the `continue` and dead code, or restore ensemble averaging if intended.

---

## Tensor Shape Inconsistencies

### 6. Documented vs. Actual Shape Convention

**Location:** `src/models/layers/framework_layer.py:15-22`

**Documented convention:** All layers should use `(C, N, F)` tensors where C=channels, N=nodes, F=features.

**Actual behavior:**

| Layer | Expected Input | Documented Input | Mismatch? |
|-------|---------------|-----------------|-----------|
| LinearLayer (aggr_features) | (N, F) | (C, N, F) | YES |
| LinearLayer (aggr_channels) | (C, N, F) | (C, N, F) | No, but output is (N, F') not (C', N, F') |
| LinearLayer (channel_wise) | (C, N, F) | (C, N, F) | No |
| BatchNormLayer | (N, F) | (C, N, F) | YES |
| All classical GNN convolutions | (N, F) | (C, N, F) | YES |
| GlobalPooling | (N, F) | (C, N, F) | YES |
| ShareGNN message passing | (N, F) -> (heads, N, F) | (C, N, F) | Partial |
| ShareGNN aggregation | (heads, N, F) -> (1, flat) | (C, N, F) | YES |

**Impact:** Mixing classical GNN layers (2D) with ShareGNN layers (3D) in the same model will fail silently or crash due to shape mismatches.

**Fix:** Either update the documentation to reflect reality (most layers are 2D), or add dimension adapters between layer types.

---

### 7. Residual Connection Shape Assumption

**Location:** All classical GNN conv layers (e.g., `gcn_conv.py:28-29`)

**Bug:** Residual connections assume identical shapes:

```python
if self.residual:
    node_representation = node_representation + x  # Direct addition
```

No validation that `x.shape == node_representation.shape`. Will crash if `in_channels != out_channels`.

GAT layers handle multi-head residuals by repeating input (line 47), but this changes semantics (residual of repeated input is not a true skip connection).

**Fix:** Add a linear projection for residual when dimensions don't match:
```python
if self.residual:
    if x.shape != node_representation.shape:
        x = self.residual_proj(x)  # Linear(in_features, out_features)
    node_representation = node_representation + x
```

---

## Performance Issues in Layer Code

### 8. Dropout Instance Created Every Forward Pass

**Location:** `src/models/layers/mpnn_classical/gcn_conv.py:31` (and all other conv layers)

**Problem:** `torch.nn.Dropout(self.dropout)(node_representation)` creates a new Dropout module on every forward call.

**Fix:** Use `torch.nn.functional.dropout(node_representation, p=self.dropout, training=self.training)` or pre-instantiate the dropout layer in `__init__`.

---

### 9. LinearLayer aggr_channels Loses Channel Dimension

**Location:** `src/models/layers/nn_standard/linear.py:55-62`

**Problem:** After `aggr_channels` mode, output is `(N, F')` but `out_channels` is set to 1 (line 31). Downstream layers expecting `(C, N, F)` will fail.

There's a commented-out line `node_representation.unsqueeze(0)` (line 62) that would restore the channel dimension but it's disabled.

**Fix:** Either uncomment the unsqueeze to produce `(1, N, F')`, or clearly document that `aggr_channels` is a terminal channel operation.

---

### 10. BatchNorm Incompatible with 3D Input

**Location:** `src/models/layers/nn_standard/batch_normalization.py:13-14`

**Problem:** Uses `torch_geometric.nn.BatchNorm` which expects `(N, F)`. If input is `(C, N, F)` from a multi-head layer, it will fail or produce incorrect results.

**Fix:** Add shape handling like LayerNormalization does:
```python
if len(node_representation.shape) == 3:
    C, N, F = node_representation.shape
    node_representation = node_representation.reshape(C * N, F)
    node_representation = self.batch_norm_layer(node_representation)
    node_representation = node_representation.reshape(C, N, F)
```

---

### 11. ActivationLayer and DropoutLayer Missing Required Args

**Location:** `src/models/layers/nn_standard/activation.py`, `dropout.py`

**Problem:** These layers call `FrameworkLayer.__init__` which requires `in_features`, `out_features`, `in_channels`, `out_channels`, `seed`, `dtype`. It's unclear how these pass-through layers satisfy all requirements.

**Impact:** May work if layer_args is populated by the model builder, but fragile.

---

## TODOs Found in ShareGNN Code

| Location | TODO |
|----------|------|
| `inv_based_message_passing.py:29` | "allow different F for different graphs" |
| `inv_based_message_passing.py:65` | "add symmetric case" |
| `inv_based_message_passing.py:98` | threshold handling |
| `inv_based_message_passing.py:144` | "symmetric case" |
| `inv_based_message_passing.py:178` | "add pruning" |
| `preprocessing.py:192` | "change the edge_label_distances to the new torch format" |
