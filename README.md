# SimpleGNN

A PyTorch-based Graph Neural Network experimentation framework for benchmarking and developing GNN architectures. Experiments are driven entirely by YAML configuration, parallelized with joblib, and evaluated with k-fold cross-validation and automatic model selection.

## Features

- **Classical message-passing layers**: GCN, GIN, GAT, GATv2, GraphSAGE (thin wrappers over PyTorch Geometric)
- **ShareGNN**: invariant-based message-passing layers using node/edge labels and pairwise properties
- **Tasks**: graph classification, graph regression, node classification
- **Three-tier YAML configuration** (main → model → hyperparameters) with grid search over list-valued parameters
- **Parallel execution** via joblib across folds, runs, and configurations
- **k-fold cross-validation** with validation-based model selection and final test-set evaluation

## Installation

The import package is `simplegnn`; the distribution name on PyPI is `simple-gnn`. Python 3.10–3.13 is supported.

### Recommended: `install.sh` (handles PyTorch CUDA/CPU setup)

```bash
git clone https://github.com/fseiffarth/SimpleGNN.git
cd SimpleGNN
./install.sh        # creates venv/, auto-detects CUDA/CPU PyTorch, runs pip install -e .
```

### Manual / development installation

```bash
git clone https://github.com/fseiffarth/SimpleGNN.git
cd SimpleGNN

# Step 1: install PyTorch (pick the wheel matching your hardware)
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.6
# or: pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only

# Step 2: install SimpleGNN in editable mode
pip install -e .
```

To install the published package instead of a checkout: `pip install simple-gnn` (install PyTorch first, as above).

## Quick Start

Run experiments from the **repository root** after installing the package. The full workflow is a five-step pipeline (see `examples/share_gnn_basic/main.py`):

```python
from pathlib import Path
from simplegnn.framework import FrameworkMain

experiment = FrameworkMain(Path('examples/share_gnn_basic/main.yml'))
experiment.preprocessing(num_threads=1)                  # load data, generate labels/splits
experiment.run_configurations(num_threads=-1)            # grid search (-1 = all CPUs)
experiment.evaluate_results()                            # select best config on validation set
experiment.run_best_configuration(num_threads=-1)        # re-run the best config
experiment.evaluate_results(evaluate_best_model=True)    # final test-set evaluation
```

Or simply run a bundled example:

```bash
python examples/share_gnn_basic/main.py
```

## Configuration System

Each experiment is described by three YAML tiers:

1. **Main config** (`main.yml`) — datasets, task type, and paths to the model and hyperparameter configs.
2. **Model config** (`models_*.yml`) — the layer architecture as a list of layer definitions (a list of lists triggers an architecture grid search).
3. **Hyperparameter config** (`parameters.yml`) — training settings (optimizer, loss, learning rate, epochs, batch size, input features). List-valued entries are expanded into the cartesian product for grid search.

Configurations are validated against mandatory parameter sets by `src/simplegnn/framework/utils/configuration_checks.py`. Layer `layer_type` strings must match the `LayerTypes` enum exactly.

## Examples & Tests

- **Examples** (`examples/`): `classical_gnns`, `share_gnn_basic`, `share_gnn_hyperparameter_search`, `test_betweenness`, `zinc`.
- **Paper reproduction** (`experiments/base_paper/`): see [Reproducing the paper experiments](#reproducing-the-paper-experiments) below.
- **Tests**: `pytest tests -q` (full suite) or `pytest tests/test_imports_and_loader_api.py -q` (fast import/API smoke test).

## Transfer Learning with ShareGNN

ShareGNN supports hash-keyed transfer learning: a model pretrained on one dataset can warm-start
training on a different dataset, even though the two were preprocessed independently and have
unrelated raw label ids. This works because every node/edge label is given a **canonical hash**
derived from its structural signature (e.g. a WL color's neighborhood identity, a cycle-count
profile) rather than from its dataset-relative integer id — so the same structure hashes to the
same value in any dataset. Standard `linear`/`layer_norm`/`batch_norm` layers still transfer by
name and shape, as usual.

**Requirements**:
- Both datasets must be preprocessed with label file format v2 (the current default — regenerate
  labels by deleting the relevant `labels/<dataset>/` directory if you have old v1 files).
- Only architectures built from **canonical** label types transfer their invariant weights.
  Canonical types include `wl`/`wl_labeled` seeded from a canonical base, `induced_cycles`,
  `closed_walks`, `degree`, and shortest-path `distances`. Non-canonical types — `primary` (TU
  datasets enumerate atom types dataset-relatively, unless `primary_labels_canonical: True` is set
  for a known-shared coding) and `betweenness_centrality` (percentile bins are dataset-relative) —
  are skipped by the transfer engine unless `invariant_transfer.allow_non_canonical: True`.
- The pretraining and finetuning model configs should share the same invariant-layer architecture
  (same heads, same label/property types) so weight slots line up head-by-head, and both should use
  `input_features: {name: constant, value: 1.0}` so the standard `linear` layers shape-match too.

**Workflow**:

1. **Pretrain** with `save_best_model: True` and `save_transfer_keys: True` in the hyperparameter
   config. This writes, alongside each `model_*.pt` checkpoint, a `model_*.keys.pt` sidecar that
   names every invariant weight slot by its canonical `(src_hash, tgt_hash, property)` key.
2. **(Optional) measure overlap** between the source and target label vocabularies before
   finetuning — a large mismatch means little will actually transfer. See
   `examples/transfer_learning/measure_overlap.py`.
3. **Finetune** by adding a `transfer:` block to the target dataset's hyperparameter config:

   ```yaml
   transfer:
     source:
       results_path: results/pretrain/ShareGNN_NCI1/   # results dir of the pretraining run
       dataset: NCI1                                    # source dataset name
       select: best                                     # best | {config_id, run_id, validation_id}
     strategy: finetune                                 # finetune | linear_probe
     head:
       reinit: always                                   # the task-specific head always gets a fresh init
     invariant_transfer:
       match: hashes                                    # hashes | none (reinit all invariant layers)
       on_missing: reinit                                # reinit | zero, for slots with no source match
       allow_non_canonical: False
       min_overlap_warn: 0.10
     freeze: []                                          # optional state-dict prefix globs to freeze
   ```

   Standard layers are copied by name+shape (except the head, which is re-initialized); invariant
   layers are remapped slot-by-slot via the canonical hash join between source and target
   vocabularies. Unmatched slots keep their fresh initialization (or are zeroed, with
   `on_missing: zero`). A per-run transfer report (matched/total slot counts, coverage) is written
   next to the run's results and printed at runtime.

**Full worked example**: `examples/transfer_learning/` pretrains ShareGNN on NCI1 and finetunes on
DHFR (both TUDatasets, auto-downloaded):

```bash
python examples/transfer_learning/main.py            # full run
python examples/transfer_learning/main.py --fast     # smoke test (1 epoch, 1 fold)
```

See `specs/18-canonical-hash-transfer-implementation.md` for the full design (canonical hashing
scheme, checkpoint sidecar format, remap engine).

## Reproducing the paper experiments

All paper experiments live in `experiments/base_paper/` and are launched with the shell scripts in that directory. Each script activates `venv/`, sets `PYTHONPATH`, changes to the repository root, and runs the full pipeline (preprocessing → grid search → validation-based model selection → best-config re-run → test evaluation). **Always run them from the repository root.**

**Prerequisites**
- Install the package and have a `venv/` at the repo root — the easiest path is `./install.sh`.
- Datasets are downloaded automatically on first run into `data/`; results are written under `results/base_paper/`.
- Each script hardcodes a thread count (`NUM_THREADS`, default `30`; `ZINC_full.sh` uses `1`). Edit the variable at the top of the script to change it. `NUM_THREADS` controls the grid-search/parallel stages only — preprocessing always runs single-threaded (a documented requirement, see Known Limitations).
- Synthetic-dataset splits ship in `experiments/base_paper/splits/synthetic/`; all other splits are read from `src/simplegnn/datasets/splits/`.

**Fast smoke test** (smallest config; verifies the pipeline end to end):

```bash
bash experiments/base_paper/ZINC_test.sh
```

| Script | Entry point(s) | Datasets | Splits | Results under `results/base_paper/` |
|---|---|---|---|---|
| `ZINC_test.sh` | `regression/ZINC/main_ZINC_test.py` | ZINC (smaller network, fast) | `fixed/ZINC_splits.json` | `regression/ZINC_test/` |
| `ZINC.sh` | `regression/ZINC/main_ZINC.py` | ZINC | `fixed/ZINC_splits.json` | `regression/ZINC/` |
| `ZINC_full.sh` | `regression/ZINC/main_ZINC_full.py` | ZINC-full | `fixed/ZINC-full_splits.json` | `regression/ZINC-full/` |
| `substructure_counting.sh` | `regression/substructure_counting/main_substructure_counting.py` | `multi`, `triangle`, `tri_tail`, `star`, `cycle4`, `cycle5`, `cycle6` | `substructure_counting/<name>_splits.json` | `regression/substructure_counting/` |
| `synthetic.sh` | `classification/synthetic/experiments_synthetic.py` | CSL, EvenOddRings2_16, EvenOddRingsCount16, LongRings100, Snowflakes — run as 4 variants (full / random-input / encoder-only / decoder-only) | `experiments/base_paper/splits/synthetic/<name>_splits.json` | `classification/Synthetic/{,Random/,Encoder/,Decoder/}` |
| `TUDatasets.sh` | `classification/tu/experiments_fair_real_world.py`, then `…/experiments_standard_real_world.py` | fair: IMDB-BINARY, IMDB-MULTI, NCI1, NCI109, Mutagenicity, DHFR (4 variants); standard/SOTA: IMDB-MULTI, IMDB-BINARY, NCI1, NCI109 (2 variants) | `fair/<name>_splits.json` and `standard/<name>_splits.json` | `classification/RealWorld/{,Random/,Encoder/,Decoder/}` and `classification/Sota/{,Random/}` |
| `TUDatasets_ablation.sh` | `classification/tu/experiments_ablation_distance.py`, then `…/experiments_ablation_threshold.py` | distance: NCI109, Mutagenicity, DHFR, NCI1; threshold: the 6 fair datasets × {lower, lower_upper, upper} × thresholds 1–20, 30, 40, 50 (69 configs) | `fair/<name>_splits.json` | `classification/Ablation/Distance/` and `classification/Ablation/Threshold/{Lower,LowerUpper,Upper}/<n>/` |

**Running a single experiment directly.** The shell scripts are thin wrappers — you can call any entry point yourself (from the repo root, with the package installed) and choose the thread count:

```bash
python experiments/base_paper/regression/ZINC/main_ZINC_test.py --num_threads 4
python experiments/base_paper/classification/tu/experiments_fair_real_world.py --num_threads 4
```

**Regenerating the threshold configs.** The 69 ablation-threshold configs (and their per-threshold hyperparameter files) are generated by a helper; re-run it after editing the template:

```bash
python experiments/base_paper/tools/migrate_threshold_configs.py
```

**QM9** lives outside `base_paper/`, in its own `experiments/qm9/` directory (it was
reconstructed after the migration — see `specs/19-qm9-migration.md`):

```bash
bash experiments/qm9/run_qm9.sh          # or: python experiments/qm9/main_QM9.py --num_threads 8
```

It trains the ShareGNN multi-head architecture on QM9 target 0 (dipole moment) with a
fixed 80/10/10 split. The split file (`src/simplegnn/datasets/splits/fixed/QM9_splits.json`)
ships with the repo; regenerate it with `python -m simplegnn.utils.qm_splits --dataset QM9`.
Results are written to `results/qm9/`.

**BREC** (`experiments/brec/`) is the expressiveness benchmark: 400 non-isomorphic graph
pairs, scored by how many a model can actually tell apart. It does *not* use
`run_configurations()` — BREC's RPC protocol trains one siamese model per pair and decides
with a Hotelling T² test — so it has its own runner (see
`specs/21-brec-expressiveness-benchmark.md` and `experiments/brec/README.md`):

```bash
MODEL=smoke PARTS=Basic PAIRS=2 EPOCHS=3 ./experiments/brec/run_brec.sh   # quick check
./experiments/brec/run_brec.sh                                            # full protocol (hours)
MODEL=gin ./experiments/brec/run_brec.sh                                  # negative control (must be 0/400)
```

The 51,200-graph dataset (400 pairs + 400 isomorphic controls × 32 relabelings × 2) is
downloaded and built automatically; `BREC-r<k>` variants keep only *k* relabelings for cheap
runs. Results are written to `results/brec/`.

### Tables & plots

After runs complete, figures and LaTeX tables are generated by standalone scripts under `experiments/base_paper/src/` (run from the repo root, package installed, with results already produced). None take CLI args — each has a bare `main()` guarded by `if __name__ == '__main__':`, with datasets/graph ids hardcoded as constants at the top of the file; edit those to target different data. All of them skip regenerating a figure if its output file already exists.

```bash
python experiments/base_paper/src/latex.py          # LaTeX result tables
python experiments/base_paper/src/latex_plots.py     # ablation curves, heatmaps, network visualizations, shared-weight plots
python experiments/base_paper/src/plot_zinc.py        # ZINC message-passing visualization
python experiments/base_paper/regression/substructure_counting/plot_substructure_counting.py  # substructure-counting message-passing visualizations
```

(The `classification_baselines.py` and `get_gnn_comparison_data.py` helpers are deprecated — they depend on modules removed in the migration and exit with a notice.)

**`plot_common.py`** — shared helpers used by all the scripts below:

| Function | Signature | Produces |
|---|---|---|
| `setup_pgf` | `setup_pgf(font_size=30)` | Configures `matplotlib.rcParams` for LaTeX/pgf output (serif, `text.usetex=True`, `pgf.texsystem=lualatex`). Call before creating any figure that will be saved via `save_latex_figure`. |
| `save_latex_figure` | `save_latex_figure(fig, path)` | Saves a figure to `path` with `bbox_inches='tight', backend='pgf'`, creating parent directories as needed. |

```python
from plot_common import setup_pgf, save_latex_figure

setup_pgf(font_size=12)   # smaller font for ablation/visualization figures; default 30 is for molecule figures
...
save_latex_figure(fig, 'results/base_paper/regression/Latex/ZINC_500_message_passing.pdf')
```

**`latex_plots.py`** — the paper's ablation and network-visualization figures, written under `results/base_paper/classification/Latex/Plots/` (network/graph plots go to dataset-specific `Latex/` folders instead). `latex_plots.main()` regenerates the full set used in the paper.

| Function | Signature | Produces |
|---|---|---|
| `ablation_threshold` | `ablation_threshold(dataset, threshold_type)` | Accuracy (errorbar, left axis) and parameter count in thousands (right axis) vs. threshold value, reading `results/base_paper/classification/Ablation/{threshold_type}/{1..20,30,40,50}/`. `threshold_type` is `'Lower'`, `'Upper'`, or `'LowerUpper'`. |
| `ablation_distance` | `ablation_distance(dataset='NCI1', max_distance=12, fontsize=9)` | Heatmap (encoder layers × max message-passing distance) of test accuracy, reading `results/base_paper/classification/Distance/{dataset}/summary.csv`. |
| `plot_network` | `plot_network(path, db_name, graph_ids, filtering, draw_type=None, with_labels_from_invariant=True, with_aggregation=False, molecule=False, channel=0, headers=True)` | Loads a trained model (from the config at `path`) and draws, per graph in `graph_ids`, the node-label graph plus one panel per (layer, filtering) combination of message-passing weights. `filtering` is a list of `None` / `{'absolute': n}` entries controlling which weights are shown per panel. |
| `plot_specific_graphs_from_db` | `plot_specific_graphs_from_db(path, db_name, graph_ids, draw_type=None, node_size=200, output_path=None)` | Draws the raw node-label-colored graphs for `graph_ids` side by side, each labeled with its target/class. |
| `rules_vs_occurences` | `rules_vs_occurences(layer, db_name, channel=0, appendix='')` | Scatter of encoder weights sorted by dataset occurrence count, colored by invariant property, with threshold markers at counts 1–10. Returns `(sort_indices, steps)` for `rules_vs_weights`. |
| `rules_vs_weights` | `rules_vs_weights(layer, sort_indices, steps, db_name, channel=0, appendix='')` | Companion plot of the actual weight values, in the same sorted/colored order as `rules_vs_occurences`. |
| `plot_shared_weights` | `plot_shared_weights(path, db_name, appendix='')` | Loads a model and calls `rules_vs_occurences` + `rules_vs_weights` together on its first invariant message-passing layer. |

Examples (from `latex_plots.main()`):

```python
ablation_threshold('NCI1', 'Lower')
ablation_distance('NCI1', 20, fontsize=6)

plot_network(
    'experiments/base_paper/regression/ZINC/configs/main_config_ZINC.yml',
    'ZINC', [500], draw_type='kawai',
    filtering=[None, {'absolute': 3}], molecule=True, headers=False,
)

plot_specific_graphs_from_db(
    'experiments/base_paper/classification/configs/main_config_fair_synthetic.yml',
    db_name='CSL', graph_ids=[0, 16, 31], draw_type='kawai',
)

plot_shared_weights(
    'experiments/base_paper/classification/configs/main_config_fair_real_world_random_variation.yml',
    'DHFR',
)
```

**`plot_zinc.py`** — `main()` loads the ZINC regression model (`experiments/base_paper/regression/ZINC/configs/main_config_ZINC.yml`, `config_id=0, run_id=0, validation_id=0`), draws graph id `500`'s molecule structure plus three attention heads (`head=0`, `head=10`, `head=18`) of `net.net_layers[2]`, and saves to `results/base_paper/regression/Latex/ZINC_500_message_passing.pdf`.

```bash
python experiments/base_paper/src/plot_zinc.py
```

Edit `graph_ids` and the head indices at the top of the script to plot a different graph or heads.

**`plot_substructure_counting.py`** — `main()` loads the substructure-counting model (`experiments/base_paper/regression/substructure_counting/configs/main_config_substructure_counting.yml`) and, for each of the 7 tasks (`triangle`, `tri_tail`, `cycle5`, `cycle4`, `cycle6`, `star`, `substructure_counting`), draws graph id `3947`'s message-passing weights across 5 attention heads (labeled `3-Cycle`/`4-Cycle`/`5-Cycle`/`6-Cycle`/`Degree`) from `net.net_layers[0]`, saved per task to `experiments/base_paper/regression/substructure_counting/Plots/{task}_substructure_counting.pdf`.

```bash
python experiments/base_paper/regression/substructure_counting/plot_substructure_counting.py
```

**Underlying drawing primitives.** The scripts above build on library code you can reuse for custom figures:

- `src/simplegnn/datasets/utils/graph_drawing.py` — `GraphDrawing` (node/edge styling config), `CustomColorMap` / `TabColorMap` / `RandomColorMap` (matplotlib colormaps), `compute_positions(graph, draw_type, root_node=None)` and `resolve_positions(graph, draw_type, pos_path='', root_node=None)` (layout computation with on-disk position caching), `filter_weight_bounds(...)` (weight-magnitude filtering for drawing). Covered by `tests/test_layer_drawing.py`.
- `InvariantBasedMessagePassingLayer.draw(ax, graph_id, graph_drawing, head=0, filter_weights=None, with_graph=True, graph_only=False, draw_bias_labels=False, pos_path='')` (`src/simplegnn/models/ShareGNN/layers/inv_based_message_passing.py:1786`) and the analogous `InvariantBasedPoolingLayer.draw(...)` (`inv_based_pooling.py:335`) — draw one graph's message-passing/pooling weights onto a given matplotlib `ax`. These are the primitives `plot_network`, `plot_zinc.main`, and `plot_substructure_counting.main` all call internally.

## Known Limitations / Pitfalls

- **Run `preprocessing()` with `num_threads=1`.** With `num_threads>1`, preprocessing runs in joblib subprocesses and the loaded train/val/test splits are computed on a pickled copy of the config — they do **not** propagate back to the parent process, so the later stages fail. Data, labels, and properties written to disk are unaffected; only the in-memory splits are lost. See `TODO.md` for the full analysis and planned fix.
- **Keep `precision` consistent.** The `precision` parameter (`float`/`double`) must match between data and model; a mismatch causes runtime errors.
- **ShareGNN requires preprocessing first.** ShareGNN models depend on node/edge labels and pairwise properties generated during `preprocessing()`.

## Requirements

- Python 3.10–3.13
- PyTorch (install separately, matched to your CUDA/CPU setup)
- PyTorch Geometric ≥ 2.7
- numpy, pandas, scikit-learn, networkx, joblib, pyyaml, matplotlib
- Optional: ogb, rdkit (molecular datasets)

## License

Apache License 2.0 — see the `LICENSE` file.
