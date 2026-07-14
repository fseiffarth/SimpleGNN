# 07 — General Codebase Analysis (Fable)

**Date**: 2026-07-14
**Scope**: Full repo — `src/simplegnn/framework/`, `src/simplegnn/models/` (incl. ShareGNN), `src/simplegnn/datasets/`, `tests/`, `examples/`/`experiments/` (config wiring only).
**Method**: Three parallel deep-exploration passes (framework orchestration / models & layers / datasets & preprocessing & tests), cross-referenced against specs 00–06 and `next_steps.md`. Every finding carries a `file:line` reference verified against the current working tree (branch `claude`).

**Legend**: **[NEW]** not covered by any existing spec · **[DOC]** already documented in a spec, still open · **[FIXED]** spec item confirmed resolved in code.

---

## 1. Executive Summary

The framework is functional on its main path (ShareGNN + classical GNN graph classification/regression via examples), but several **secondary code paths are outright broken** (node classification training, split generation, degree-matrix message passing, GAT/GATv2 per-head batch norm, NEL datasets with multiple edge labels), and several **silent correctness bugs** affect results (run 0 never reshuffles between epochs, ROC-AUC computed from hard predictions, class-weight construction, non-deterministic label caching). Extensibility is the biggest structural weakness: adding a new layer, label, or property requires editing 3–4 synchronized `if/elif` sites each, with drift already visible in the code.

### Top-priority items

| # | Finding | Where | Impact |
|---|---------|-------|--------|
| 1 | `train_node_task` signature mismatch — node classification always crashes | `model_configuration.py:401` vs `:1359` | Broken feature |
| 2 | `create_splits` / `pretraining_finetuning` / `gdtgl` undefined — NameError | `framework/utils/preprocessing.py:460,425,720` | Broken feature |
| 3 | Run 0 never reshuffles training data between epochs (`seed * run_id` with `run_id=0`) | `model_configuration.py:1438` (+4 sites) | Silent wrong results |
| 4 | ROC-AUC computed from argmax predictions, not scores | `model_configuration.py:854,949,1076` | Silent wrong metric |
| 5 | Betweenness label cache broken (recomputes + overwrites every run; NameError on cache-hit) | `node_labeling.py:1419-1434` | Perf + crash |
| 6 | ShareGNN `degree_matrix`/`use_in_degrees` forward uses never-assigned attributes | `inv_based_message_passing.py:767,769` | Broken feature |
| 7 | Non-deterministic label integers from unordered `set` iteration | `node_labeling.py:957-971` | Irreproducible caches |
| 8 | Registration of layers/labels/properties spread over 3–4 `if/elif` sites each | see §8 | Extensibility |

### Cross-reference to existing specs

| Spec item | Status |
|---|---|
| 01 §1 per-graph forward loop (no batching) | **[DOC]** open — `model_configuration.py:1282,1356` |
| 01 §2 dense `(H,N,N)` weight matrix per forward | **[DOC]** open — `inv_based_message_passing.py:632,649` |
| 01 §3 quadratic `torch.cat` in weight-distribution build | **[FIXED]** — single cat per graph (`:384-397`) |
| 01 §4 config lookups in forward | **[FIXED]** — flags cached (`:419-420`), but exposed dead code, see §2.4 |
| 01 §5 / 06 §2 clone in label relabeling | **[FIXED]** — `torch.stack` (`:264-267`) |
| 01 §6 pooling triple loop + per-head unique | **[DOC]** open — `inv_based_pooling.py:54-60` |
| 01 §7 pooling flattens all structure | **[DOC]** open — `inv_based_pooling.py:162` |
| 01 §8 `eval()` security risk in layer parsing | **[FIXED]** — `ast.literal_eval` (`layer_loader.py:154`); but `eval()` remains in `graph_dataset.py:1078` (`transform_data`) |
| 01 §9 empty positional-encoding file | **[DOC]** open |
| 01 §10 / 05 parallel label computation / layer init | **[DOC]** open |
| 02 §2 `torch.no_grad()` in evaluation | **[FIXED]** — `model_configuration.py:1335,1419` |
| 02 §3 `os.system` copy | **[FIXED]** — `shutil.copy2` (`core.py:1072-1077`) — but see §2.3, the guard is dead |
| 02 §4 per-epoch CSV writes | **[FIXED]** — `_csv_buffer`/`_flush_csv_buffer` (`model_configuration.py:1249-1261`) |
| 02 §9 hardcoded eval batch size 512 | **[FIXED]** — configurable `eval_batch_size` (`:1338`) |
| 02 §10 full val/test eval every epoch | **[FIXED]** — `validation_frequency` (`:416-422`) |
| 04 §1 sequential label computation | **[DOC]** open |
| 04 §2 one-hot memory overhead | **[DOC]** open |
| 04 §4 nx conversion not cached across label types | **[DOC]** open |
| 04 §10 TUDataset no-feature slice loop | **[DOC]** — documented as *slow*; also a **correctness** risk (isolated nodes), see §3.9 |
| 06 §1–3 2-D `torch.unique` on label pairs | **[FIXED]** — 1-D encoding (`inv_based_message_passing.py:297-306`) |
| 06 §4 `unique(dim=0)` on bias/pooling labels | **[DOC]** open — `:362`, `inv_based_pooling.py:57` |

Everything in §§2–9 below marked **[NEW]** is not covered by specs 00–06.

---

## 2. Critical Bugs — Broken Code Paths

These paths crash deterministically when exercised.

### 2.1 Node classification training crashes — **[NEW]**
`model_configuration.py:401` calls `self.train_node_task(epoch=..., values=..., train_batches=..., timer=...)`, but the definition at `:1359` is `def train_node_task(self, epoch, values, train_batches, random_variation_bool, timer)`. `random_variation_bool` is required, never passed, and never computed anywhere → `TypeError` on every `node_classification` run. Fix: read `random_variation_bool` from the model/config (cf. `model.py:218`) or drop the parameter and read the config inside, as `train_graph_task` does.

### 2.2 Undefined names in `framework/utils/preprocessing.py` — **[NEW]**
Used but never imported nor defined anywhere in `src/` (grep-confirmed):
- `create_splits(...)` at `:460` — the **default** split-generation path in `create_split_file()` (`with_splits: true`).
- `pretraining_finetuning(...)` at `:425` — transfer-learning split merging.
- `gdtgl.draw_graph(...)` at `:720` — triggered by `plot_graphs: true`.

Any config that needs split generation (i.e., no pre-existing split JSON), pretraining/finetuning splits, or graph plotting raises `NameError`. These look like casualties of the package restructuring; the functions must be restored or reimplemented.

### 2.3 Config archival never runs (dead truthiness guard) — **[NEW]**
`core.py:1074`:
```python
if not results_path.joinpath(f"{graph_db_name}/config.yml"):
```
A `Path` is always truthy, so the body never executes and the experiment config is **never** copied to results (the docstring claims it is). Intent was `if not ....exists():`. Reproducibility feature is currently a silent no-op.

### 2.4 ShareGNN degree-matrix forward uses never-assigned attributes — **[NEW]**
`inv_based_message_passing.py:767,769` use `self.in_edges[pos]` and `self.D[pos]`, which are **never assigned** in the class or its base (`InvariantBasedLayer`). Any config with `degree_matrix: true` or `use_in_degrees: true` crashes with `AttributeError`. Spec 01 §4 cached the gating flags (`:419-420`) but didn't notice the gated code is dead/broken. Either implement `self.D`/`self.in_edges` construction or remove the options + config-check support.

### 2.5 GAT/GATv2 per-head batch norm reads non-existent key — **[NEW]**
`gat_conv.py:37` and `gatv2_conv.py:42` read `self.gat_args['out_features']` / `self.gatv2_args['out_features']` in the `merge_heads=False` batch-norm branch, but those dicts only define `'out_channels'` (`gat_conv.py:12-22`) → `KeyError` with `batch_norm: true, merge_heads: false`.

### 2.6 ShareGNN `base_labels` preprocessing crashes — **[NEW]**
`ShareGNN/preprocessing/preprocessing.py:242`:
```python
for layer in proprocessed_label_dicts_first:
    layer_to_labels(layer)
```
`layer_to_labels` requires `(experiment_configuration, layer_strings, graph_data, ...)` (cf. correct call at `:244`). Any label dict with `base_labels` (e.g. `wl_labeled`) that populates this set crashes with `TypeError`. (Also note the typo `proprocessed_`.)

### 2.7 NEL loader v2 drops node attributes / crashes on multi-class edge labels — **[NEW]**
`graph_dataset.py:588-711` (`read_nel_data_v2`, the reader actually in use):
- `node_attributes` is initialized `None` at `:592` and never reassigned (real data goes into local `node_attr` at `:622`), so `sizes['num_node_attributes']` is always 0 (`:690`) and `data.node_attributes` is always empty (`:695-699`) — attributes only survive implicitly inside `x`.
- `:701` references `data.edge_labels`, which is never set (only `edge_attr`/`edge_data` exist) → `AttributeError` for any NEL dataset with more than one edge-label class.
- `:662` `one_hot(edge_labels)` without `num_classes` — width depends on which labels happen to occur; `num_edge_labels` (`:691`) then measures one-hot width, inconsistent with `set_sizes()`'s `len(torch.unique(...))` convention elsewhere.

### 2.8 Betweenness-centrality label cache broken by indentation — **[NEW]**
`node_labeling.py:1419-1434`: the `if not file.exists():` block contains only the `print` + `start_time`; generation and `save_labels_to_file` run **unconditionally**, so betweenness (O(nm) per graph) is recomputed and the cache file overwritten on every run. On a cache-hit with `save_times` set, `start_time` is undefined → `NameError`. The trailing `else: print("already exists. Skipping.")` binds to `if save_times is not None`, not to the existence check. This is the one labeler refactored through the (otherwise dead) class hierarchy (§5.2) and it lost the guard all other `save_*` functions have.

### 2.9 Misc guaranteed crashes in reachable-but-rare paths — **[NEW]**
- `model.py:447-448`: the `raise ValueError(f'... in_channels={in_channels}')` references a non-existent local — when the guard fires, you get a `NameError` instead of the informative error.
- `model.py:218`: `config.get('input_features', None).get('random_variation', None)` → `AttributeError` when `input_features` is absent from the config.
- `graph_dataset.py:974` (`batches_from_ids`): undefined `id_batch` → `NameError` (docstring itself says it's buggy; dead but shipped).
- `inv_based_pooling.py:108-110`: `he` init returns a 1-D `torch.randn(num_weights)` while callers pass 2-D shapes for biases (`:68`) — shape mismatch (the `lower_upper` branch handles 2-D correctly).

---

## 3. Correctness Bugs — Silent Wrong Results

These don't crash; they quietly produce wrong numbers.

### 3.1 Run 0 never reshuffles training data across epochs — **[NEW]**
`model_configuration.py:1438` (and `:1447,1475,1482,1538`):
```python
shuffling_seed = seeds[epoch][self.k_val] * self.run_id + self.seed
```
With `run_id == 0` the epoch-dependent term vanishes → `shuffling_seed == self.seed` for **every epoch** → identical batch order all training long for run 0. Fix: make the mixing non-multiplicative, e.g. `hash((seeds[epoch][self.k_val], self.run_id, self.seed))` or `seeds[epoch][self.k_val] * (self.run_id + 1) + self.seed`.

### 3.2 Weighted-loss class weights: shape and device — **[NEW]**
`model_configuration.py:389-391`: `torch.unique(y[batch], return_counts=True)[1]` returns counts only for classes **present** in the batch. Any batch missing a class → shape-mismatch on assignment into the `(num_batches, num_classes)` tensor (or, worse, silently misaligned weights if shapes coincide). Use `torch.bincount(y[batch], minlength=num_classes)`. Also, `class_weights` is created on CPU and never moved to `self.device` → wrong-device error (or hidden transfer) for CUDA runs when passed to `CrossEntropyLoss(weight=...)` (`:1288-1289`).

### 3.3 ROC-AUC computed from hard predictions — **[NEW]**
`model_configuration.py:854,949,1076`: `sklearn.metrics.roc_auc_score(labels, prediction)` where `prediction` is argmax'd class labels. AUC requires scores/probabilities; with hard labels the value is a rescaled accuracy, not AUC. Pass softmax scores (`outputs[:, 1]` binary / `multi_class='ovr'` multiclass). The tensors are also passed to sklearn without `.cpu()`.

### 3.4 Test-metric columns silently 0.0 during grid search — **[NEW]**
`model_configuration.py:1020`: test evaluation only runs `if config.get('best_model', False)`, but `postprocess_writer` unconditionally writes `test_values` (default 0.0) into the CSV. During normal grid search, TestAccuracy/TestLoss are all zeros — easy to misread. Either skip the columns or write NaN.

### 3.5 Grid-search option generation corrupts layer options — **[NEW]**
`run_configuration.py:48-50`: `key_list.remove(key)` inside `for key in key_list:` — the classic mutate-while-iterating bug; adjacent removable keys are skipped. `:78`/`:88`: `curr_layer_dict = base_dict` aliases the **same dict** across all iterations, so every appended option shares one `'channels'` slot and all options end up with the last-written channels. Fix: iterate over a copy, and `copy.deepcopy(base_dict)` per option.

### 3.6 Non-deterministic label integers (cache poisoning) — **[NEW]**
`node_labeling.py:957-971` (`save_labeled_degree_labels`): the label→int mapping is built by enumerating an unordered `set` of strings → depends on `PYTHONHASHSEED`; cached `.pt` content differs run-to-run/machine-to-machine for the same graphs. Fix: `sorted(unique_neighbor_labels)` before enumerating (as `save_cycle_labels`/`save_clique_labels` already do).

### 3.7 Precision handling gaps — **[NEW]**
- `graph_dataset.py:282-284`: only `x`, `node_attributes`, `edge_attributes` are cast to configured precision; **`y` is not**. With `precision: double` (all example regression configs), ZINC/QM9 targets stay float32 → silent upcasts / stricter-op mismatches; normalization (`:916-924`) writes `y` in place without dtype control.
- `model.py:303-312` (`random_variation`): noise created without `device=` → CUDA device mismatch; precision default at `:306` is `'double'` while `__init__` (`:215`) defaults `'float'` — the two disagree when `precision` is unset.

### 3.8 Duplicate evaluators disagree on label format — **[NEW]**
`core.py:928` (`evaluate_model`) computes accuracy via `torch.argmax(labels[i])` (assumes one-hot), while near-identical `evaluate_model_on_graphs` (`core.py:855`) compares labels as class indices. At most one is right for a given label format. Merge into one function (§4.3).

### 3.9 TUDataset no-feature branch undercounts isolated nodes — partially **[DOC 04 §10]**, correctness aspect **[NEW]**
`graph_dataset_preprocessing.py:367-378` recomputes per-graph node counts as `max(edge_index slice) − min(...) + 1` — assumes contiguous indices and that every node has an incident edge. Isolated nodes → wrong slices for `x`/`primary_node_labels`. Use the dataset's `ptr`/`num_nodes` metadata instead.

### 3.10 `relabel_node_labels` invalid-marker restore is a no-op — **[NEW]**
`node_labeling.py:1438-1462`: negatives are bucketed to `max_id` and the bucket sorts last, but (a) `:1458` caps `>= max_number_labels` to `max_number_labels−1`, folding the invalid bucket into "other"; (b) `:1460` restores `−1` where `frequency_sorted_labels == max_id`, which never holds since frequency labels live in `0..k−1`. Nodes with negative source labels lose their `−1` marker in the frequency column. Verify downstream ShareGNN interpretation of `−1` before fixing.

### 3.11 In-place mutation of cached PyG slices during merge — **[NEW]**
`graph_dataset_preprocessing.py:219-223,339-342` (ZINC/Substructure): `validation_data.slices[key] += train_data.slices[key][-1]` mutates the source datasets' `.slices` in place — corrupts them if those objects are ever reused in-process.

### 3.12 Split handling — **[NEW]**
`framework/utils/preprocessing.py:530-540`: only `model_selection[0]` is used per fold (silent truncation for nested-CV split files), and indices are never range-checked against `len(dataset)` — out-of-range surfaces later as an opaque indexing error.

### 3.13 Exception swallowing — **[NEW]**
- `core.py:189-193`: bare `except:` collapses all config load/validation failures into a generic `ValueError`, discarding `check_main_configuration_file`'s specific messages.
- `framework/utils/preprocessing.py:270,308,319`: bare `except:` prints and continues with `self.graph_data = None` — failure surfaces later as a confusing downstream error.
- `inv_based_message_passing.py:280`: `except (FileNotFoundError, Exception)` — swallows **all** errors in the cache-load path, reporting genuine bugs as "cache miss"; also `Exception` subsumes `FileNotFoundError`.
- `parameters.py:429`: bare `except: pass`.

### 3.14 Smaller issues — **[NEW]**
- GAT reads `layer_args.get('num_heads', 1)` (`gat_conv.py:15`) but GATv2 reads `layer_args.get('heads', 1)` (`gatv2_conv.py:15`); `model.py:439-440` normalizes to `num_heads`, so GATv2 silently runs with 1 head for the same-looking config.
- `linear.py:44` constructs a `torch.nn.Linear` the forward never uses — dead parameters registered with the optimizer (wasted memory/compute, and it perturbs any parameter-count reporting).
- BatchNorm built as `BatchNorm(in_channels=self.in_features)` (`framework_layer.py:265`) and applied directly to the representation (`batch_normalization.py:14`) — only valid for `(N,F)`; multi-channel `(C,N,F)` inputs normalize the wrong axis or error. Latent — verify which configs feed BatchNorm.
- `model_configuration.py:1441` mutates `self.para.run_config.batch_size` in place inside `get_train_batches` — the shrink persists across epochs/folds within a worker.
- `core.py:271`: `num_threads` is folded across the dataset loop without reset — a small value from one dataset leaks into the next.
- `evaluation.py:581,595,643`: `model_selection_evaluation` annotated `-> int` but returns `None` on early-exit branches; `core.py:474` indexes `run_configs[...]` with the result.
- `preprocess_writer` opens the Network `.txt` in append mode (`model_configuration.py:1115`) — reruns duplicate content; results CSV is opened twice (`"w"` at `:1192`, `"a"` at `:1209`).

---

## 4. Code Duplication

Each item: what's duplicated → consolidation proposal. All **[NEW]** unless noted.

### 4.1 Classical GNN wrapper epilogue (5×)
`gcn_conv.py:23-33`, `gin_conv.py:30-43`, `sage_conv.py:24-34`, `gat_conv.py:28-46`, `gatv2_conv.py:30-51` all repeat `batch_norm → activation → residual(+x) → dropout(if training)` (GAT/GATv2 with head-merge variants). → Add `def _finalize(self, x_in, x_out)` to the currently near-empty `GNNConvLayer` ABC (`gnn_conv.py:9-14`); wrappers become `return self._finalize(x, self.layer(x, edge_index))`. This also fixes 2.5/3.14 in one place.

### 4.2 `model_configuration.py` internal duplication
- Regression inverse-transform (`output_features_inverse` + standard/minmax/minmax_zero `invert_outputs`) copy-pasted 3× (`:826-838, :911-936, :1038-1063`) → one `_invert_regression_outputs(outputs, labels)` helper.
- Validation vs test evaluation in `evaluate_results` near-identical (`:889-1015` vs `:1017-1106`) → one `_evaluate_split(split)` (§5.1).
- `get_train_batches` default branch (`:1436-1443`) byte-identical to the `else` fallback (`:1537-1543`).
- Prediction-CSV dump block 3× (`:875-887, :1003-1015, :1094-1106`), with a hardcoded relative `"Results/Parameter/..."` path.

### 4.3 `core.py` twin evaluators
`evaluate_model_on_graphs` (`:792-860`) vs `evaluate_model` (`:862-934`) — same routine, different data source, **divergent accuracy computation** (§3.8). → Merge; parameterize the data source.

### 4.4 ShareGNN layer internals
- Two near-identical `init_weights` (`inv_based_message_passing.py:533-623`, `inv_based_pooling.py:80-115`) → one function on `InvariantBasedLayer` taking a `shape`; fixes the `he` 2-D bug (§2.9) once.
- ~450 lines of duplicated `draw()` graph-layout code (`inv_based_message_passing.py:796-1029` — duplicated *twice internally* at 810-852/954-994 — and `inv_based_pooling.py:173-316`) → shared drawing util.
- Pooling registers pruning params (`inv_based_pooling.py:74-78`) while message passing has the same block commented out (`:409-414`) — divergent feature support.

### 4.5 Dead parallel node-labeling class hierarchy (~400 lines)
`node_labeling.py:363-760`: `NodeLabelingBase` + 8 subclasses referenced only in docstrings (grep-confirmed), duplicating the used `save_*` functions — and buggy (`save_labels_to_file` at `:574` unconditionally `raise NotImplementedError` at `:590`; `WeisfeilerLehmanNodeLabeling.generate` at `:748` indexes `optional_parameters['depth']` on a list). Only `BetweennessCentralityNodeLabeling` (`:763`) is reachable — via the broken path in §2.8. → **Delete the hierarchy**, inline betweenness into a normal `save_*` function (fixing §2.8), *before* building the label registry (§8).

### 4.6 Dataset-layer duplication
- Two NEL readers: `read_nel_data` (v1, `graph_dataset.py:514`, unused) vs `read_nel_data_v2` (`:588`) → delete v1.
- Legacy `GraphData`/`GraphDataUnion`/`BenchmarkDatasets`/`zinc_to_graph_data` stack (`graph_dataset.py:1084-1518`) re-implements the input-feature pipeline (`normalize`, `unit_circle`, `one_hot`, …) that `preprocess_share_gnn_data` (`:736-926`) also implements, with subtle differences → deprecate/remove the legacy stack or extract one shared transform module.
- `save_*` boilerplate (filename build, existence check, nx-graph guard, timing try/except) copy-pasted ~11× in `node_labeling.py` → absorbed by the label registry wrapper (§8).
- `properties.py`: `write_distance_properties` (`:17-74`) and `write_distance_edge_properties` (`:152-245`) share the slices-accumulation skeleton.
- `evaluation.py`: the "locate EpochLoss column → size-weight → groupby → divide" aggregation and the summary-CSV writing are repeated across 4 functions; the summary header string is duplicated verbatim (`:630`,`:633`).
- `custom_benchmarks/*.py` generators repeat the seed/build/permute skeleton (`long_rings.py:11,51`; `even_odd_rings.py:21,105`) → shared base helper.

---

## 5. Structural Improvements

### 5.1 God methods to decompose — **[NEW]**
| Method | Size | Mixed responsibilities | Split into |
|---|---|---|---|
| `ModelConfiguration.evaluate_results` (`model_configuration.py:718-1108`) | ~390 lines | train-metric averaging, val eval, test eval, best-model checkpointing, prediction dumps, string-dispatched | `_evaluate_split()`, `_invert_regression_outputs()`, `_maybe_checkpoint_best()`, `_dump_predictions()` |
| `InvariantBasedMessagePassingLayer.__init__` (`inv_based_message_passing.py:141-420`) | ~280 lines | metadata, per-head index building, threshold filtering, disk caching, bias, param init | `_process_head()`, `_build_bias()`, `_load_or_build_distributions()` (aligns with specs 05/06) |
| `layer_from_yml_invariant_based` (`layer_loader.py:37-155`) | ~120 lines, 5-deep nesting | manual cross-product of head combinations | use `itertools.product` over normalized option lists |
| `load_preprocessed_data_and_parameters` (`framework/utils/preprocessing.py:558-722`) | ~165 lines | output flags, label loading, property loading, parameter setters, plotting | one function per concern |
| `GraphDataset.process()` (`graph_dataset.py:336-466`) | ~130-line if/elif | some sources get preprocessing classes, others inline handlers; `sizes` shape varies per branch (MoleculeNet stores a tensor under `num_node_labels`, `:370`) | one preprocessing class per source, uniform `sizes` contract |

### 5.2 Dead code inventory (delete) — **[NEW]** except where noted
- `node_labeling.py:363-760` dead class hierarchy (§4.5).
- `save_in_circle_labels` (`node_labeling.py:1208`) and `write_distance_circle_properties` (`properties.py:77-149`) — orphaned, not reachable from any dispatch; the latter uses the obsolete `.prop`/`graph_data.graphs` API.
- `LayerTypes.SHARE_GNN_LINEAR` / `SHARE_GNN_LAYER_NORM` (`layer_types.py:25-26`) — enum members with no `get_model_layer`/`check_layer` handling.
- `inv_based_positional_encoding.py` — empty file for an enumerated layer type **[DOC 01 §9]**.
- `read_nel_data` v1 (`graph_dataset.py:514`), `batches_from_ids` (`:928-975`), `load_model_old` (`load_model.py:37`), `no_curriculum_sampling` (`data_sampling.py:68`), unused `sorted_labels/sorted_indices` (`data_sampling.py:62-63`), `test_weight_update` (`model_configuration.py:652`), commented pruning scaffolding (`model.py:412-414`, `inv_based_message_passing.py:409-414`, `model_configuration.py:689-701`).
- Broken/unused imports: `framework_layer.py:4-6` (`from torch._C.cpp import nn`, unused `Sequential/Linear/ReLU/BatchNorm1d`), `inv_based_pooling.py:8` (`from pandas.core.array_algos.masked_accumulations import cumsum` — pandas internals, brittle across versions).
- Dead `torch.zeros` alloc immediately overwritten in `inv_based_pooling.set_weights` (`:117-124`) + trailing `pass`.

### 5.3 Magic values and naming — **[NEW]**
- Seed base `42` hardcoded (`core.py:719`, `load_model.py:86,154`); initial best-loss `1000000.0` (`model_configuration.py:364`) → `float('inf')`.
- `evaluation.py` hardcodes relative `"Results/..."` paths, ignoring the configured `paths['results']`; hardcoded dataset-name allowlist controls aggregation semantics (`evaluation.py:329`).
- Typos frozen into APIs: `RSMELoss`/`rsme_error` (should be RMSE, `model_configuration.py:564,497`), `resize_graph` stored as `resize_grad` (`parameters.py:271`); `k_val` vs `validation_id` used interchangeably.
- Config-default drift: `evaluation_metric` default `'accuracy'` re-specified 6+ times; `precision` defaults `'float'` in some sites, `'double'` in others (`model.py:215` vs `:306`; `core.py:1128`; `framework/utils/preprocessing.py:281,354`). → central typed accessors on `RunConfig`.

---

## 6. Runtime Optimizations (training/eval path)

Open **[DOC]** items first (see cross-ref table §1): per-graph ShareGNN forward (01 §1), dense `(H,N,N)` rebuild per forward call (01 §2), pooling init loops (01 §6), pooling flatten (01 §7), parallel init (01 §10/05), `unique(dim=0)` on 1-D labels (06 §4). New findings:

- **[NEW]** `evaluate_graph_task` re-allocates the zero `outputs` tensor **inside** the accumulation loop with growing size (`model_configuration.py:1341-1344`) — compute `len(graph_ids)` once, allocate once.
- **[NEW]** `set_weights`/`set_bias` rebuild `current_W`/`current_B` on **every** forward (`inv_based_message_passing.py:762-779`) even in eval where parameters are frozen — cache per `pos` within `no_grad` evaluation sweeps.
- **[NEW]** `parameters.py:421` (`set_file_index`): full `os.listdir` + filename parse of the results dir on every run.
- **[NEW]** `evaluation.py` re-reads and re-concats every per-run CSV for each configuration id across its four evaluation functions — read once, aggregate vectorized.
- **[NEW]** class-weight einsum recomputed per epoch even with static batch composition (`model_configuration.py:387-392`), on CPU (device copy per batch, §3.2).
- **[NEW]** MoleculeNet label remap is O(N·U) Python (`graph_dataset.py:363,367`: `torch.tensor([torch.where(unique == x)[0] for x in column])`) → `torch.searchsorted`/`bucketize`.

---

## 7. Loader / Preprocessing Optimization Plan

Ordered; each step is independently shippable.

1. **Fix cache correctness first** (prereq for everything cached): betweenness indentation bug (§2.8); deterministic ordering in `save_labeled_degree_labels` (§3.6); include generating parameters + a dataset identity component in cache filenames — today labels live in `data/.../labels/<name>/` keyed only by dataset name and label params, so regenerating a dataset under the same name silently reuses stale `.pt` files. The idempotency test (`tests/test_cache.py`) checks sizes only and cannot catch this.
2. **Parallelize per-graph label/property computation** with `joblib.Parallel` — every `save_*` and both property writers loop `graph_data.nx_graphs` sequentially in pure Python (`nx.all_pairs_shortest_path_length` `properties.py:35`, `simple_cycles` `node_labeling.py:1156-1160`, `find_cliques` `:1335`, subgraph isomorphism `:1272`, betweenness `:836`). Embarrassingly parallel per graph. **[DOC 04 §1]**, still the single biggest preprocessing win.
3. **Cache the NetworkX conversion once per run** — `create_nx_graphs` is triggered lazily by the first labeler (the eager call is commented out at `ShareGNN/preprocessing/preprocessing.py:240`), and worse, `create_nx_graph` (singular) is called **per graph at layer-build time** (`inv_based_message_passing.py:798`, `inv_based_pooling.py:175`). **[DOC 04 §4]**.
4. **Cut `edge_label_distances` cost**: `copy.deepcopy` of all-pairs-all-shortest-paths per graph (`properties.py:171-174`) plus O(n²) node-pair loops (`:194-195`) — restructure to iterate paths once without the deepcopy.
5. **Unify serialization**: properties use gzip-pickle (`properties.py:57-60`, loaded eagerly and whole into memory `edge_labeling.py:32-33`), node labels use `fs.torch_save` (`node_labeling.py:587`), the dataset uses `torch_save` of `to_dict()` (`graph_dataset.py:483`). Standardize on `torch.save`+versioned dataclass payloads; consider lazy/per-split property loading for ZINC/QM9-scale pairwise tensors.
6. **One-hot memory** (`graph_dataset.py:628,752`, ZINC edge one-hot `graph_dataset_preprocessing.py:233`) — keep integer labels + embed at model input, or sparse one-hot. **[DOC 04 §2]**.
7. Remove the remaining `eval()` in `transform_data` (`graph_dataset.py:1078`) — a lookup table of allowed transforms (cf. next_steps #2).

---

## 8. Extensibility Roadmap — new labels, distances/properties, new GNNs

The core problem is identical in three subsystems: **registration by synchronized `if/elif` chains**. Drift has already happened (dead enum members, orphaned labelers/properties, `get_label_string` naming diverging from `save_*` filenames — e.g. `degree`→`wl_0`, betweenness naming differs between `node_labeling.py:348` and `:1410`).

### 8.1 Current cost of adding each extension type

| Extension | Edit sites today |
|---|---|
| New GNN layer | 1) `layer_types.py` enum member; 2) `model.py:453-492` if/elif branch; 3) `layer_loader.py:158-261` `check_layer` validation branch; 4) short-form handling in `layer_from_yml*` (`layer_loader.py:14-155`) |
| New node-label function | 1) `save_<x>_labels` in `node_labeling.py` (~40 lines boilerplate); 2) dispatch branch in `layer_to_labels` (`ShareGNN/preprocessing/preprocessing.py:49-188`); 3) `get_label_string` branch (`node_labeling.py:142`) — filename must match the `save_*` function exactly or caching silently misses; 4) validation lists (config checks) |
| New pairwise property/distance | 1) `write_<x>_properties` in `properties.py`; 2) branch in `property_to_properties` (`preprocessing.py:199-212`, currently a hardcoded 2-way if/elif); 3) substring-match branch in `Properties.add_properties` (`edge_labeling.py:41-58`); 4) match the undocumented pickled `(valid_properties, final_dict, slices_dict)` triple |
| New optimizer / loss / scheduler / metric | if/elif in `set_optimizer` (`model_configuration.py:592-605`, **silent Adam fallback on typo**), `set_loss_function` (`:559-576`), `set_scheduler` (`:638-641`, silent `None` on unknown type), and metric strings threaded through `evaluate_results`/`postprocess_writer`/`model_selection_evaluation` with per-metric CSV-header branches (`:1196-1205, :1232-1247`) |

### 8.2 Proposed: three registries + one pattern

**a) Layer registry** — `src/simplegnn/models/layers/utils/registry.py`:
```python
LAYER_REGISTRY: dict[str, LayerSpec] = {}   # LayerSpec = (cls, required_fields, yml_normalizer)

@register_layer('gcn_convolution', required=('in_features', 'out_features'))
class GCNConv(GNNConvLayer): ...
```
- `model.get_model_layer` → single dict lookup; `check_layer` validates `required` generically; unknown type → error listing registered names.
- `LayerTypes` enum kept as a thin view over registry keys (or deprecated).
- Optional: `importlib.metadata` entry-points group `simplegnn.layers` for out-of-tree layers → truly plugin-like.
- Combined with the shared `_finalize()` epilogue (§4.1), a new classical GNN becomes **one new file, zero edits elsewhere**.

**b) Label registry** — in `datasets/utils/node_labeling.py`:
```python
@register_label('wl', params=('depth', 'max_labels'))
def compute_wl(graph_data, depth, max_labels): -> NodeLabels payload
```
The decorator wrapper owns all currently copy-pasted boilerplate: canonical filename derivation (single source of truth — kills the `get_label_string` drift), cache-existence guard, `create_nx_graphs` guard, timing capture, `save_labels_to_file`. `layer_to_labels` becomes a registry lookup on `label_type`. Prereq: delete the dead class hierarchy (§4.5) and fix §2.8.

**c) Property registry** — same shape for `property_to_properties`; plus a small frozen dataclass documenting the on-disk contract (`valid_properties`, `final_dict`, `slices_dict`) so a new property (resistance distance, effective resistance, positional encodings, learned distances) is a single registered function producing that dataclass. Replace the substring dispatch in `Properties.add_properties` (`edge_labeling.py:45,55`) with a per-property `interpret_valid_values` hook carried in the registry entry.

**d) Optimizers/losses/schedulers/metrics** — plain dict registries in `model_configuration.py`; unknown names **raise** with the list of known names (no silent Adam/None fallback). A `Metric` protocol (`compute(outputs, labels) -> float`, `csv_columns`) removes the per-metric branches in the CSV writers.

### 8.3 After-state

| Extension | Edits after refactor |
|---|---|
| New GNN layer | 1 file (`@register_layer` + class) |
| New label | 1 function (`@register_label`) |
| New distance/property | 1 function (`@register_property`) |
| New optimizer/loss/metric | 1 registry entry (+ metric class if new columns) |

Natural first candidates to validate the design: the currently-orphaned `save_in_circle_labels` and `write_distance_circle_properties` (register or delete), and the empty `INVARIANT_BASED_POSITIONAL_ENCODING` layer (01 §9) as the first registry-registered ShareGNN layer.

---

## 9. Test Coverage & Test Hygiene

### 9.1 New/modified tests (currently uncommitted)
- `tests/test_share_gnn_mutag_integration.py` — full pipeline on MUTAG incl. gradient-flow check on invariant-layer `Param_W`; `tests/test_classical_gnn_mutag_integration.py` — same, parametrized over GCN/GAT/GATv2/GIN/SAGE; fixtures under `tests/fixtures/`.
- `tests/conftest.py` — adds `mutag_main_config`, autouse `seed_all(1337)`, `share_gnn_setup`.
- `tests/test_cache.py` — label-preprocessing idempotency (two runs, byte-size equality of `labels/MUTAG/*.pt`); `tests/test_cache_direct.py` — invariant-layer weight-distribution cache reproducibility (`torch.equal`).

### 9.2 Hygiene issues in these tests — **[NEW]**
- They write labels/properties/caches into the **real repo `data/TUDatasets/` tree** (`conftest.py:37-39`, `test_cache.py:27`, `test_cache_direct.py:27`), not `tmp_path` → shared mutable state, order-dependent; the idempotency assertion depends on artifacts left by prior runs.
- Network-dependent (MUTAG download on first run) with no offline skip → add a `pytest.mark.skipif`/download guard.
- `test_cache_direct.py` never clears the cache before asserting existence — a stale file from an unrelated run satisfies it.

### 9.3 Unit-test gaps (highest value first)
1. Tiny hand-built-graph tests for each `save_*` labeler and both property writers (known expected labels/distances) — would have caught §2.8, §3.6, §3.10.
2. `read_nel_data_v2` with a fixture containing node attributes + ≥2 edge-label classes — catches §2.7 outright.
3. `graph_dataset_preprocessing.py`: merge path (incl. `all_edge_atr`/`edge_attr` key inconsistency at `:143-145`), TUDataset no-feature branch with an isolated node (§3.9).
4. `load_splits`: multi-`model_selection` truncation, out-of-range indices (§3.12).
5. `generate_layer_options` option-independence (catches §3.5 aliasing).
6. A smoke test per task type — `node_classification` would have caught §2.1 immediately.

---

## 10. Prioritized Roadmap

Order respects dependencies (cache fixes before perf work that relies on caches; dead-code deletion before registries).

### P0 — Broken paths & silent wrong results (small, independent fixes)
| Item | Ref | Effort |
|---|---|---|
| `train_node_task` signature | §2.1 | XS |
| Restore/implement `create_splits`, `pretraining_finetuning`; guard `gdtgl` | §2.2 | M |
| `copy_experiment_config` `.exists()` | §2.3 | XS |
| Shuffling seed `run_id==0` | §3.1 | XS |
| Class weights: `bincount` + device | §3.2 | XS |
| ROC-AUC from scores | §3.3 | S |
| Betweenness cache indentation | §2.8 | XS |
| Deterministic label enumeration (sort) | §3.6 | XS |
| `generate_layer_options` aliasing/iteration | §3.5 | S |
| GAT/GATv2 `out_features`→`out_channels`, unify `num_heads` key | §2.5, §3.14 | XS |
| `y` precision cast; `random_variation` device/precision | §3.7 | S |
| NEL v2 attributes/edge-labels | §2.7 | S |
| Remove `self.D`/`self.in_edges` options or implement them | §2.4 | S/M |
| Replace bare excepts with narrow ones | §3.13 | S |

### P1 — Performance
1. Parallelize label/property preprocessing (joblib) + nx-conversion caching (§7.2–3) — biggest preprocessing win.
2. ShareGNN batching / sparse forward (specs 01 §1–2) — biggest training win; out of scope here but unblocked by nothing above.
3. Small wins: eval-tensor allocation (§6), eval-time `current_W` caching (§6), MoleculeNet remap (§6), `edge_label_distances` deepcopy (§7.4).

### P2 — Extensibility (registries)
1. Delete dead labeling class hierarchy + orphans (§4.5, §5.2) — prereq.
2. Label registry (§8.2b), then property registry (§8.2c) — validates on existing labelers with unchanged cache filenames.
3. Layer registry + `_finalize()` epilogue (§8.2a, §4.1).
4. Optimizer/loss/scheduler/metric registries (§8.2d).

### P3 — Cleanup & test hygiene
1. Decompose god methods (§5.1) — best done after the registries reduce their branching.
2. Deduplicate `model_configuration.py` / `core.py` / `evaluation.py` (§4.2–3, §4.6).
3. Central config accessors; kill magic paths/values; fix typo APIs with deprecation shims (§5.3).
4. Redirect tests to `tmp_path` + offline skips; add the unit tests of §9.3.

---

*Analysis produced by Claude (Fable) on branch `claude`; complements specs 00–06 — items marked [DOC] remain tracked in their original spec; this document is the tracking home for all [NEW] items.*
