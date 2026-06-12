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
- **Paper reproduction** (`experiments/base_paper/`): shell scripts for TUDatasets, ZINC, QM9, ablations, and synthetic benchmarks.
- **Tests**: `pytest tests -q` (full suite) or `pytest tests/test_imports_and_loader_api.py -q` (fast import/API smoke test).

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
