# CLAUDE.md - SimpleGNN Project Guide

## Project Overview

SimpleGNN is a PyTorch-based Graph Neural Network experimentation framework for benchmarking and developing GNN architectures. It supports classical message-passing GNNs (GCN, GIN, GAT, GATv2, GraphSAGE) and a proprietary ShareGNN variant with invariant-based layers. The framework handles graph classification, graph regression, and node classification tasks.

## Repository Structure

```
SimpleGNN/
├── src/                              # All source code
│   ├── framework/                    # Training/evaluation orchestration
│   │   ├── core.py                   # FrameworkMain - main entry point class
│   │   ├── model_configuration.py    # Single config training: model init, train/eval loops
│   │   ├── run_configuration.py      # Hyperparameter grid search combinations
│   │   └── utils/
│   │       ├── parameters.py         # Parameters class (all experiment hyperparams)
│   │       ├── preprocessing.py      # Dataset loading, splits, label computation
│   │       ├── evaluation.py         # Result analysis and visualization
│   │       ├── configuration_checks.py  # YAML config validation
│   │       └── data_sampling.py      # Batch sampling strategies
│   ├── models/                       # GNN models and layers
│   │   ├── model.py                  # GraphModel - main PyTorch model class
│   │   ├── layers/
│   │   │   ├── framework_layer.py    # Abstract base class for all custom layers
│   │   │   ├── mpnn_classical/       # Classical GNN wrappers (gcn, gin, gat, gatv2, sage, pooling)
│   │   │   ├── nn_standard/          # Standard layers (linear, activation, batchnorm, dropout, reshape)
│   │   │   └── utils/                # LayerTypes enum, layer_loader
│   │   └── ShareGNN/                 # Proprietary ShareGNN implementation
│   │       ├── layers/               # inv_based_message_passing, inv_based_pooling, positional_encoding
│   │       ├── preprocessing/        # ShareGNN-specific label/property preprocessing
│   │       └── utils.py
│   ├── datasets/                     # Dataset handling
│   │   ├── graph_dataset.py          # GraphDataset (PyG InMemoryDataset)
│   │   ├── graph_dataset_preprocessing.py  # Dataset-specific preprocessing
│   │   ├── custom_datasets.py        # Dataset factory and registration
│   │   ├── custom_benchmarks/        # Synthetic benchmarks (rings, snowflakes, strings)
│   │   ├── splits/                   # Pre-computed train/val/test splits (JSON)
│   │   └── utils/                    # Node/edge label utilities, graph functions
│   └── utils/                        # General utilities (timer, path conversions)
├── examples/                         # Example experiments with YAML configs
│   ├── basic_example/                # GCN/GIN on MUTAG
│   ├── basic_example_share_gnn/      # ShareGNN on MUTAG
│   └── zinc/                         # ShareGNN on ZINC
├── experiments/                      # Shell scripts for reproducing paper results
│   └── base_paper/                   # TUDatasets, ZINC, QM9, ablation, synthetic benchmarks
├── data/                             # Dataset storage (gitignored, auto-downloaded)
├── results/                          # Experiment results (gitignored)
└── docs/                             # Sphinx documentation
```

## Configuration System

The framework uses a **three-tier YAML configuration**:

1. **Main config** (`main.yml`): Datasets, task type, paths to model/hyperparameter configs, splits
2. **Model config** (`models_*.yml`): Layer architecture as list of layer definitions (supports grid search via list of lists)
3. **Hyperparameter config** (`parameters.yml`): Training params (optimizer, loss, lr, epochs, batch size, input features). Lists = grid search.

Configs are validated by `src/framework/utils/configuration_checks.py` against mandatory parameter sets.

## Running Experiments

Entry point pattern (see `examples/*/main.py`):

```python
from pathlib import Path
from framework.core import FrameworkMain

experiment = FrameworkMain(Path('examples/basic_example/main.yml'))
experiment.preprocessing(num_threads=1)        # Load data, generate labels/splits
experiment.run_configurations(num_threads=-1)   # Grid search (-1 = all CPUs)
experiment.evaluate_results()                   # Find best config on validation set
experiment.run_best_configuration(num_threads=-1)  # Re-run best config
experiment.evaluate_results(evaluate_best_model=True)  # Final test-set evaluation
```

Run from the `src/` directory (imports are relative to `src/`):
```bash
cd src && python -m examples.basic_example.main
```

Experiment shell scripts are in `experiments/base_paper/`.

## Key Architecture Decisions

- **GraphModel** (`src/models/model.py`): Sequential `nn.ModuleList` of layers built from YAML config. All layers extend `FrameworkLayer` base class.
- **Layer types**: Defined in `LayerTypes` enum (`src/models/layers/utils/layer_types.py`). New layers must be registered there and in `layer_loader.py`.
- **Tensor shapes**: Layers handle `(C, N, F)` for multi-channel/multi-head data or `(N, F)` for standard. C = channels/heads, N = nodes, F = features.
- **LinearLayer modes**: `aggr_features` (standard), `aggr_channels` (aggregate across channels), `channel_wise` (independent per channel).
- **ShareGNN layers** use node/edge labels and pairwise properties for invariant-based message passing (multi-head output).

## Dependencies

Core: PyTorch, PyTorch Geometric (torch_geometric), numpy, pandas, scikit-learn, networkx, joblib, pyyaml, matplotlib.
Optional: ogb, rdkit (molecular datasets).

Virtual environment is in `venv/`. Note: `requirements.txt` and `setup.py` have been removed from tracking.

## Coding Conventions

- **Classes**: PascalCase (`FrameworkMain`, `GraphDataset`, `GCNConv`)
- **Functions/methods**: snake_case (`run_configurations`, `forward`, `load_preprocessed_data_and_parameters`)
- **Constants**: UPPER_SNAKE_CASE (`MANDATORY_MAIN_CONFIG_PARAMS`)
- **Private members**: underscore prefix (`_num_graph_nodes`)
- **Imports**: Standard library first, then third-party (torch, numpy), then local modules. Local imports use dot-notation relative to `src/` (e.g., `from models.model import GraphModel`).
- **Type hints**: Used in method signatures where present, not comprehensive across the codebase.
- **Indentation**: 4 spaces.
- **Docstrings**: Present on classes and key methods; not exhaustive. Don't add docstrings to code you didn't write.

## Git Conventions

- **Main branch**: `master`
- **Commit messages**: Imperative mood, descriptive ("Refactor linear layer implementation to support 'aggr_channels' mode")
- **No unit test suite**: Testing is done via integration experiments (example scripts and experiment shell scripts)
- **Gitignored**: `data/`, `results/`, `*.csv`, `venv/`, `__pycache__/`, `.idea/`, `.auto-claude/`, `/.claude/`

## Important Patterns

- **Adding a new GNN layer**: Create wrapper class extending `FrameworkLayer` in `src/models/layers/mpnn_classical/`, add to `LayerTypes` enum, register in `layer_loader.py`, then reference in YAML model config.
- **Adding a new dataset**: Implement preprocessing class in `graph_dataset_preprocessing.py`, register in `custom_datasets.py`, create split files in `datasets/splits/`.
- **Grid search**: Use lists in YAML parameter files. The framework computes the cartesian product of all list-valued parameters.
- **Parallel execution**: `joblib` handles parallel runs across splits/configs. `num_threads=-1` uses all CPUs.
- **Results**: Saved as CSV per epoch per configuration in the results directory. Evaluation finds best config by validation metric.

## Common Pitfalls

- Always run from `src/` directory or ensure `src/` is on the Python path; imports are relative to it.
- YAML model configs use `layer_type` string keys that must match `LayerTypes` enum values exactly.
- ShareGNN preprocessing must run before training ShareGNN models (generates required node/edge properties).
- The `precision` parameter (`float`/`double`) must be consistent between data and model; mismatches cause runtime errors.
