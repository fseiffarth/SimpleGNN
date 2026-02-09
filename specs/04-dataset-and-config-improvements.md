# Dataset and Configuration Improvements

## Dataset Handling

### 1. Expensive Label Computation Without Parallelism

**Location:** `src/datasets/utils/node_labeling.py`

**Problem:** Label computation for ShareGNN (WL iterations, cycle detection, clique finding) runs sequentially across all graphs. Cycle detection and clique finding are NP-hard operations.

**Cost estimates per 1,000 graphs:**

| Label Type | Time |
|-----------|------|
| trivial/index/degree | < 1 second |
| wl (depth 3) | 10-60 seconds |
| simple_cycles (len 6) | 30+ seconds |
| induced_cycles (len 6) | 60+ seconds |
| cliques | 2-5 minutes |
| subgraph isomorphism | hours |

Labels are cached to disk after first computation, so this is a one-time cost per dataset + label config. But the initial run can be very slow.

**Proposed fix:** Parallelize graph-level label computation using `joblib.Parallel` since each graph is independent.

---

### 2. One-Hot Encoding Memory Overhead

**Location:** `src/datasets/graph_dataset.py:588-595`

**Problem:** One-hot encoding creates dense `(num_nodes, max_label)` tensor. For datasets with many unique labels (e.g., 100+), this significantly increases memory.

**Example:** ZINC with 800k nodes and 100 unique labels = 320 MB for one-hot features alone.

**Proposed fix:** Use sparse tensors for one-hot features where applicable, or consider learned embeddings as an alternative to one-hot.

---

### 3. Recursive File I/O in Label Preprocessing

**Location:** `src/models/ShareGNN/preprocessing/preprocessing.py:24-44`

**Problem:** When `label_type` is a list, the function recursively calls itself for each type, loading labels from disk each time:

```python
for label_type in layer['label_type']:
    l_path = layer_to_labels(experiment_configuration, json.dumps(new_layer_string), graph_data, ...)
    labels.append(load_labels(l_path))
```

**Proposed fix:** Batch-load all label types, or compute them in-memory and only save at the end.

---

### 4. NetworkX Conversion Overhead

**Location:** `src/datasets/utils/node_labeling.py:554-555` (WL), `:654` (cycles)

**Problem:** Several label types require NetworkX graph objects. The conversion from PyG edge_index to NetworkX happens per graph and is not cached.

**Proposed fix:** Cache NetworkX graphs in `GraphDataset.nx_graphs` (field already exists at line 44 but is unused by default). Compute once, reuse across label types.

---

### 5. Multiple Label Schemes in Memory

**Location:** `src/datasets/graph_dataset.py:44-48`

**Problem:** Multiple label variants stored simultaneously in `graph_data.node_labels` dict. Using 7 label schemes on ZINC = ~20-25 MB of label tensors.

Not critical for small datasets, but wasteful if only a subset of labels is needed for the current model config.

**Proposed fix:** Lazy-load labels on demand from disk cache instead of keeping all in memory.

---

## Configuration System

### 6. Missing Semantic Validation

**Location:** `src/framework/utils/configuration_checks.py:153-207`

**Problem:** Configuration validation checks presence of parameters but not their values:

| Parameter | What's Checked | What's Missing |
|-----------|---------------|----------------|
| `input_features.name` | Key exists | Not validated against `['node_labels', 'constant', 'node_features', 'all']` |
| `input_features.transformation` | Not checked | Should validate against `['one_hot', 'normalize', 'normalize_positive', 'unit_circle']` |
| `loss` | Key exists | Not validated against supported loss functions |
| `optimizer` | Key exists | Not validated against supported optimizers |
| `label_type` (in model YAML) | Not checked | Should validate against supported label types |
| `layer_type` | Checked against enum | OK |
| `activation` | Not checked | Passed to `eval()` (see bugs spec) |

**Proposed fix:** Add value validation for all parameters with known valid values. Report meaningful error messages for invalid values.

---

### 7. No Schema Enforcement

**Problem:** YAML configs have no schema definition. Typos in parameter names are silently ignored (fall through to defaults).

Example: `learnnig_rate: [0.01]` (typo) would silently use the default learning rate.

**Proposed fix:** Define a JSON Schema or Pydantic model for each config level. Validate on load, reject unknown keys.

---

### 8. Missing Configuration Options

**Useful additions:**

| Feature | Description | Config Level |
|---------|-------------|-------------|
| `validation_frequency` | Validate every N epochs | parameters.yml |
| `eval_batch_size` | Evaluation batch size (currently hardcoded 512) | parameters.yml |
| `max_configurations` | Limit grid search size | parameters.yml |
| `mixed_precision` | Enable AMP training | parameters.yml |
| `gradient_checkpointing` | Trade compute for memory | parameters.yml |
| `label_workers` | Parallel workers for label computation | parameters.yml |
| `log_frequency` | How often to print training progress | parameters.yml |
| `tensorboard` | Enable TensorBoard logging | parameters.yml |

---

## Dataset Preprocessing

### 9. Merged Dataset Padding Inefficiency

**Location:** `src/datasets/graph_dataset_preprocessing.py:159-186`

**Problem:** When merging multiple datasets, all feature dimensions are padded to the maximum across datasets. Creates intermediate tensors.

**Impact:** Minor for small datasets, but wasteful for datasets with very different feature sizes.

**Proposed fix:** Use sparse tensors or pad only when necessary (at batch time, not preprocessing time).

---

### 10. TUDataset Missing Feature Handling

**Location:** `src/datasets/graph_dataset_preprocessing.py:364-378`

**Problem:** If a TUDataset has no node features, the code creates zero tensors and recomputes slices from edge_index. This is correct but slow for large datasets due to the per-graph loop for slice computation.

**Proposed fix:** Vectorize slice computation using `torch.unique_consecutive` on batch indices.

---

## Priority Order

1. **Section 6** - Add semantic validation (1-2 hours, prevents silent config errors)
2. **Section 1** - Parallelize label computation (2-4 hours, major preprocessing speedup)
3. **Section 4** - Cache NetworkX graph conversions (1 hour)
4. **Section 8** - Add missing config options (incremental, per-feature)
5. **Section 7** - Schema enforcement (4-8 hours, best as a separate project)
