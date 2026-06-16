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
| `QM9.sh` | — | — | — | **Unavailable**: the QM9 experiment files were lost in the repository migration, so this script prints a notice and exits 1. Only the splits survive, at `src/simplegnn/datasets/splits/tu_splits/QM9_splits.json`. |

**Running a single experiment directly.** The shell scripts are thin wrappers — you can call any entry point yourself (from the repo root, with the package installed) and choose the thread count:

```bash
python experiments/base_paper/regression/ZINC/main_ZINC_test.py --num_threads 4
python experiments/base_paper/classification/tu/experiments_fair_real_world.py --num_threads 4
```

**Regenerating the threshold configs.** The 69 ablation-threshold configs (and their per-threshold hyperparameter files) are generated by a helper; re-run it after editing the template:

```bash
python experiments/base_paper/tools/migrate_threshold_configs.py
```

**Tables & plots.** After runs complete, `experiments/base_paper/src/latex.py` / `latex_plots.py`, `experiments/base_paper/src/plot_zinc.py`, and `experiments/base_paper/regression/substructure_counting/plot_substructure_counting.py` produce the paper tables and figures. (The `classification_baselines.py` and `get_gnn_comparison_data.py` helpers are deprecated — they depend on modules removed in the migration and exit with a notice.)

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
