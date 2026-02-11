# SimpleGNN

A PyTorch-based Graph Neural Network experimentation framework for benchmarking and developing GNN architectures.

## Features

- Classical GNN layers: GCN, GIN, GAT, GATv2, GraphSAGE
- ShareGNN: Proprietary invariant-based message passing layers
- Task support: Graph classification, graph regression, node classification
- YAML-based experiment configuration with grid search

## Installation

### Recommended: Using install.sh (handles PyTorch CUDA setup)

```bash
git clone https://github.com/fseiffarth/SimpleGNN.git
cd SimpleGNN
./install.sh
```

### Manual installation

**Step 1: Install PyTorch**
```bash
# CUDA 12.6
pip install torch>=2.10.0 --index-url https://download.pytorch.org/whl/cu126
# OR CPU-only
pip install torch>=2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

**Step 2: Install SimpleGNN**
```bash
pip install simple-gnn
```

### Development installation

```bash
git clone https://github.com/fseiffarth/SimpleGNN.git
cd SimpleGNN
pip install torch>=2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

## Quick Start

```python
from pathlib import Path
from simplegnn.framework import FrameworkMain

experiment = FrameworkMain(Path('config/experiment.yml'))
experiment.preprocessing(num_threads=1)
experiment.run_configurations(num_threads=-1)
experiment.evaluate_results()
```

## Requirements

- Python 3.10-3.13
- PyTorch 2.10.0+
- PyTorch Geometric 2.7.0+

## License

Apache License 2.0 - See LICENSE file.
