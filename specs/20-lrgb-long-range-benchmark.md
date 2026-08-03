# 20 — Long Range Graph Benchmark (Peptides-func / Peptides-struct)

## Context

The most commonly used long-range GNN benchmark is the **Long Range Graph
Benchmark** (LRGB, Dwivedi et al., NeurIPS 2022 D&B). It has 5 datasets;
**Peptides-func** (multi-label graph classification, 10 tasks) and
**Peptides-struct** (11-target graph regression) are the two most widely
reported in the graph-transformer / long-range literature, because — unlike
PascalVOC-SP/COCO-SP (node classification on large superpixel graphs) or
PCQM-Contact (link prediction) — they are plain graph-level tasks that plug
directly into SimpleGNN's existing `graph_classification`/`graph_regression`
pipelines.

`torch_geometric.datasets.LRGBDataset` (PyG 2.7.0, already installed) provides
both. There was already a half-finished, broken attempt at this in
`graph_dataset.py` (an orphaned `elif self.from_existing_data == 'Peptides':`
branch) that never set `primary_node_labels`/`node_attributes`/
`primary_edge_labels`/`edge_attributes`, was never added to the `source`
allow-list in `configuration_checks.py`, and only loaded the default
`split='train'` instead of merging the official train/val/test split. This
spec replaces it with a proper implementation, following the same pattern as
the QM9 migration (spec 19): a dedicated `*GraphDataPreprocessing` class, a
fixed split JSON, and a new `experiments/lrgb/` folder.

## Feature layout (verified against installed PyG source)

`torch_geometric/datasets/lrgb.py`'s `process()` confirms Peptides-func/struct
use OGB's `atom_to_feature_vector`/`bond_to_feature_vector` encoding: `x` is
`[N, 9]`, `edge_attr` is `[E, 3]` — the same column layout
`OGBGraphPropertyGraphDataPreprocessing` already assumes (col 0 = primary
label, remaining columns = attributes). `y` is `[1, 10]` (func) or `[1, 11]`
(struct) per graph.

## Critical finding: `graph_classification` cannot be used for Peptides-func as-is

The framework's `graph_classification` code path unconditionally flattens `y`
(`graph_dataset.py:246-250`) and uses argmax-based single-label predictions
throughout `model_configuration.py` (training/validation/`evaluate_network`).
That is a single-label assumption. Peptides-func's target is genuinely
**multi-label** (10 independent binary tasks per graph) — using
`graph_classification` as originally scoped would silently corrupt the
targets during the flatten/slice step, not just mis-report a metric.

**Decision:** model Peptides-func as `task: graph_regression` with
`regression_targets: [0..9]` and `loss: BCEWithLogitsLoss`. `graph_regression`
never flattens `y` (correctly keeps all 10 columns), and `set_loss_function`
dispatches purely on the loss name, so `BCEWithLogitsLoss` works unmodified.
This is a **documented deviation**: the framework will report BCE loss and a
non-meaningful "MAE"/"accuracy" surrogate, not the paper's Average Precision
metric. A real multi-label patch to `graph_dataset.py`/`model_configuration.py`
(sketched in the planning notes, not implemented here) is future work if AP
parity is ever needed.

Peptides-struct needs no such workaround — it is already a standard
`graph_regression` task, handled exactly like QM9 (`regression_targets:
[0..10]`, `output_features: {normalization: standard}` +
`invert_outputs: {normalization: standard}` to report MAE in original units).

## Work

1. **`LRGBGraphDataPreprocessing`** (`graph_dataset_preprocessing.py`) — hybrid
   of `ZINCGraphDataPreprocessing`'s 3-way official-split merge and
   `OGBGraphPropertyGraphDataPreprocessing`'s 9/3-column atom/bond split.
2. **Dispatch** — remove the broken `'Peptides'` branch in `graph_dataset.py`;
   add `elif self.from_existing_data == 'LRGB': ... LRGBGraphDataPreprocessing(self.name) ...`
3. **Allow-list** — add `'LRGB'` to `configuration_checks.py`'s source enum.
4. **Splits** — `src/simplegnn/utils/lrgb_splits.py`: unlike QM9's random
   80/10/10, LRGB ships an *official* split; the generator just measures the
   three official split lengths and emits contiguous offsets (train → val →
   test, matching the preprocessing class's concatenation order) to
   `src/simplegnn/datasets/splits/fixed/peptides-func_splits.json` and
   `.../peptides-struct_splits.json`.
5. **`experiments/lrgb/`** — mirrors `experiments/qm9/`'s layout, with two
   parallel experiments (func/struct) sharing one folder. ShareGNN multi-head
   architecture adapted from `experiments/qm9/models_QM9.yml`, with the
   `distances` property capped at `cutoff: 15` (down from QM9's 0-23) since
   Peptides graphs are far larger (~150 nodes, diameter ~57 vs QM9's <20
   atoms) — a disclosed, pragmatic limitation on receptive field, not a claim
   of matching published LRGB results.

## Verification

1. `FrameworkMain(main_config_peptides_func.yml)` / `..._struct.yml` construct
   without error (allow-list, model-config expansion, split-file-exists check).
2. Split file sanity: 1 fold, disjoint train/val/test, union covers every
   index, func/struct splits share identical index sets.
3. `preprocessing(num_threads=1)` smoke run for both — shape assertions
   (`x.shape[1]==9`, `edge_attr.shape[1]==3`, `y.shape==(15535,10 or 11)`,
   no NaNs in `y`).
4. Short training run (few epochs) for both configs — loss decreases, no
   shape-mismatch errors, `evaluate_results()` writes `summary.csv` cleanly.
5. `pytest tests -q` stays green.
