# CLAUDE.md - SimpleGNN Project Guide

## Project Overview

SimpleGNN is a PyTorch-based Graph Neural Network experimentation framework for benchmarking and developing GNN architectures. It supports classical message-passing GNNs (GCN, GIN, GAT, GATv2, GraphSAGE) and a proprietary ShareGNN variant with invariant-based layers. The framework handles graph classification, graph regression, and node classification tasks.

---

## Multi-Agent Workflow Guide

> **Note**: SimpleGNN has successfully used single-agent workflows for all development to date (see specs/ folder for examples). Multi-agent patterns are available for complex tasks but should be used judiciously—they add latency and cost. Start simple and escalate to multi-agent only when needed.

### Quick Decision Flowchart

```
Task Complexity → Workflow Pattern
├─ Trivial (typo, comment) → Single agent
├─ Standard feature (follows existing pattern) → Develop → Test
├─ Novel/Complex (5+ files, new patterns) → Plan → Review → Develop → Test
└─ Documentation only → Single agent (Review optional)
```

### When to Use Multi-Agent Patterns

**SimpleGNN-specific triggers**:

| Task Type | Workflow | Example |
|-----------|----------|---------|
| New GNN layer | Develop → Test | Add custom layer, test on MUTAG |
| Framework core changes | Plan → Develop → Test | Modify `src/simplegnn/framework/core.py`, test multiple experiments |
| ShareGNN optimizations (complex) | Plan → Review → Develop → Test | Parallel initialization (see specs/05-parallel-layer-initialization.md) |
| Dataset additions | Develop → Test | Add preprocessing, verify data loading |
| Documentation/specs | Single agent | Evidence: specs/Documentation.md (100% complete, single-agent) |

**Skip multi-agent for**: Bug fixes, single-file changes, typos, quick debugging, documentation updates.

### Available Workflows

#### 1. Planning → Review Workflow
**When**: Major refactoring, novel features, high-risk changes (5+ files, core framework)
**Pattern**: Architect Chief (Opus) explores and creates plan → Architect Reviewer (Opus) validates, identifies risks → Developer implements approved plan
**SimpleGNN example**: ShareGNN parallel initialization (specs/05-parallel-layer-initialization.md)
**Skip for**: Features following existing patterns, bug fixes, single-file changes

#### 2. Development → Testing Workflow
**When**: After modifying GNN layers, framework core, datasets, or model configuration
**Pattern**: Developer implements changes → Tester (Sonnet/Haiku) runs relevant experiment, reports PASS/FAIL
**SimpleGNN example**: After adding layer in `src/simplegnn/models/layers/`, run `python examples/share_gnn_basic/main.py`
**Skip for**: Trivial changes, documentation-only updates

#### 3. Development → Review Workflow
**When**: Complex changes requiring quality review
**Pattern**: Developer implements → Reviewer (Sonnet) analyzes for conventions, bugs, performance
**Skip for**: Simple implementations with clear requirements

### Agent Roles and Model Selection

| Role | Model | Use Case | Description Parameter |
|------|-------|----------|----------------------|
| 🏗️ Architect Chief | Opus | Complex planning | `"🏗️ Architect Chief: [task]"` |
| 🔍 Architect Reviewer | Opus | Plan validation | `"🔍 Architect Reviewer: [task]"` |
| 👨‍💻 Developer | Sonnet | Implementation | *(default, current agent)* |
| 🧪 Tester | Sonnet/Haiku | Test execution | `"🧪 Tester: verify [feature]"` |
| 👀 Reviewer | Sonnet | Code review | `"👀 Reviewer: analyze [files]"` |
| 🔎 Explorer | Sonnet | Codebase research | `"🔎 Explorer: investigate [topic]"` |
| ⚡ Runner | Haiku | Quick operations | `"⚡ Runner: [simple task]"` |

**Haiku use cases**: File reads, grep/glob searches, simple edits, running tests with clear instructions
**Avoid Haiku for**: Debugging, architectural decisions, multi-step problem-solving

### Implementation Reference

<details>
<summary>Task tool usage examples (click to expand)</summary>

**Planning → Review**:
```python
Task(
    subagent_type="Plan",
    model="opus",
    description="🔍 Architect Reviewer: validate plan",
    prompt="""[AGENT ROLE: 🔍 Architect Reviewer]
Review plan at [path]. Verify completeness, correctness, risks, conventions.
Report: "🔍 Architect Reviewer: [APPROVED/NEEDS REVISION] - [analysis]"
"""
)
```

**Development → Testing**:
```python
Task(
    subagent_type="Bash",
    model="sonnet",  # or "haiku" for simple tests
    description="🧪 Tester: verify implementation",
    prompt="""[AGENT ROLE: 🧪 Tester]
1. From the repo root, run pytest tests -q (or a specific example)
2. e.g. python examples/share_gnn_basic/main.py
3. Check for errors
4. Report: "🧪 Tester: PASS/FAIL - [details]"
"""
)
```

**Development → Review**:
```python
Task(
    subagent_type="general-purpose",
    model="sonnet",
    description="👀 Reviewer: analyze code",
    prompt="""[AGENT ROLE: 👀 Reviewer]
Review [files] for: conventions, patterns, bugs, performance.
Report: "👀 Reviewer: [summary and recommendations]"
"""
)
```
</details>

**Communication**: Agents should prefix reports with role emoji and name (e.g., "🧪 Tester: ✓ PASS - all tests passed").

---

## Context Management for Agents

**When to use /clear**:
- Before starting a new feature (avoid context pollution from previous work)
- After completing a workflow phase (e.g., after 🧪 Tester finishes, clear before next task)
- When switching between unrelated tasks
- When conversation exceeds ~100 messages or feels cluttered

**Preserve important context first**:
- Save plans to specs/ before clearing (e.g., `specs/new-feature-plan.md`)
- Document decisions in commit messages
- Export key findings to markdown files

**Example workflow**:
1. 🏗️ Architect Chief creates plan → save to `specs/feature-plan.md`
2. /clear (fresh context for implementation)
3. 👨‍💻 Developer implements → commit changes with descriptive message
4. /clear (fresh context for testing)
5. 🧪 Tester runs experiments → report results
6. If more work needed: save results, /clear, continue

---

## Conversation Management

**Auto-compression**: Claude Code automatically compresses older messages when approaching context limits, preserving key information while reducing token usage. This is beneficial for large codebase exploration and multi-file analysis.

**Manual management**: Use /clear strategically (see Context Management above) rather than relying solely on auto-compression.

**Settings**: Compression typically activates around 150K tokens, retaining the last 20-30 messages uncompressed.

---

## Repository Structure

<details>
<summary>Full directory tree (click to expand)</summary>

```
SimpleGNN/                            # Repo root (== git root)
├── pyproject.toml setup.py requirements.txt install.sh  # Packaging / install
├── src/simplegnn/                    # Installable package (`pip install -e .`)
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
│   │   ├── evaluations/              # Dataset evaluation helpers
│   │   ├── splits/                   # Pre-computed train/val/test splits (JSON)
│   │   └── utils/                    # Node/edge label utilities, graph functions
│   └── utils/                        # General utilities (timer, path conversions)
├── tests/                            # pytest suite (unit + import/API smoke tests)
├── examples/                         # Example experiments with YAML configs
│   ├── classical_gnns/               # Classical MPNNs (GCN/GIN/GAT/...)
│   ├── share_gnn_basic/              # ShareGNN baseline
│   ├── share_gnn_hyperparameter_search/  # ShareGNN grid search
│   ├── test_betweenness/             # Betweenness-based benchmark
│   └── zinc/                         # ShareGNN on ZINC
├── experiments/                      # Shell scripts for reproducing paper results
│   └── base_paper/                   # TUDatasets, ZINC, QM9, ablation, synthetic benchmarks
├── data/                             # Dataset storage (gitignored, auto-downloaded)
├── results/                          # Experiment results (gitignored)
├── docs/                             # Sphinx documentation
└── specs/                            # Planning, analysis, and documentation history
    ├── Documentation.md              # NumPy-style docstring implementation progress
    ├── 00-analysis-overview.md       # Codebase analysis overview
    ├── 01-sharegnn-optimizations.md  # ShareGNN layer optimizations
    ├── 02-training-pipeline-optimizations.md
    ├── 03-model-infrastructure-bugs.md
    ├── 04-dataset-and-config-improvements.md
    ├── 05-parallel-layer-initialization.md
    ├── 06-invariant-layer-torch-unique-optimization.md
    ├── 06-package-installation-implementation.md
    ├── agent-workflow-original.md
    └── next_steps.md                 # Future optimization roadmap
```
</details>

**Key directories**: `src/simplegnn/framework/` (training orchestration), `src/simplegnn/models/` (GNN implementations), `src/simplegnn/datasets/` (data handling), `examples/` (experiment configs), `tests/` (pytest suite), `specs/` (planning/analysis history)

---

## Planning and Documentation History

The **`specs/`** folder contains planning documents, technical analysis, and documentation implementation tracking created during Claude-assisted development sessions. This folder serves as:

- **Context for future planning**: Review previous analyses and decisions before starting new work
- **Implementation history**: Track what has been optimized, documented, or refactored
- **Technical specifications**: Detailed analysis of optimization opportunities and architectural decisions
- **Progress tracking**: `Documentation.md` tracks the NumPy-style docstring implementation across 7 core files (4,149 lines, 100% complete)

**Key documents**:
- `Documentation.md`: Complete record of NumPy-style docstring implementation
- `next_steps.md`: Roadmap for future optimizations and improvements
- `0X-*.md` files: Detailed technical analyses of specific optimization areas

When planning new features or refactoring, **always review relevant specs/** documents first to understand previous design decisions and avoid duplicating work.

---

## Configuration System

**Three-tier YAML config**: Main → Model → Hyperparameters

1. **Main config** (`main.yml`): Datasets, task type, paths to model/hyperparameter configs, splits
2. **Model config** (`models_*.yml`): Layer architecture as list of layer definitions (supports grid search via list of lists)
3. **Hyperparameter config** (`parameters.yml`): Training params (optimizer, loss, lr, epochs, batch size, input features). Lists = grid search.

**Validation**: Configs checked by `src/simplegnn/framework/utils/configuration_checks.py` against mandatory parameter sets.

---

## Running Experiments

<details>
<summary>Standard experiment workflow (click to expand)</summary>

Entry point pattern (see `examples/*/main.py`):

```python
from pathlib import Path
from simplegnn.framework.core import FrameworkMain

experiment = FrameworkMain(Path('examples/share_gnn_basic/main.yml'))
experiment.preprocessing(num_threads=1)        # Load data, generate labels/splits
experiment.run_configurations(num_threads=-1)   # Grid search (-1 = all CPUs)
experiment.evaluate_results()                   # Find best config on validation set
experiment.run_best_configuration(num_threads=-1)  # Re-run best config
experiment.evaluate_results(evaluate_best_model=True)  # Final test-set evaluation
```

**Key steps**:
1. `preprocessing()` - Load data, generate labels/splits
2. `run_configurations()` - Grid search over hyperparameters
3. `evaluate_results()` - Find best config on validation set
4. `run_best_configuration()` - Re-run best config
5. `evaluate_results(evaluate_best_model=True)` - Final test-set evaluation

</details>

**Run from the repo root** (after `pip install -e .`): `python examples/share_gnn_basic/main.py`
**Tests**: `pytest tests -q` (full suite) or `pytest tests/test_imports_and_loader_api.py -q` (fast smoke test)
**Shell scripts**: See `experiments/base_paper/` for paper reproduction

---

## Key Architecture Decisions

- **GraphModel** (`src/simplegnn/models/model.py`): Sequential `nn.ModuleList` of layers built from YAML config. All layers extend `FrameworkLayer` base class.
- **Layer types**: Defined in `LayerTypes` enum (`src/simplegnn/models/layers/utils/layer_types.py`). New layers must be registered there and in `layer_loader.py`.
- **Tensor shapes**: Layers handle `(C, N, F)` for multi-channel/multi-head data or `(N, F)` for standard. C = channels/heads, N = nodes, F = features.
- **LinearLayer modes**: `aggr_features` (standard), `aggr_channels` (aggregate across channels), `channel_wise` (independent per channel).
- **ShareGNN layers** use node/edge labels and pairwise properties for invariant-based message passing (multi-head output).

---

## Dependencies

**Core**: PyTorch, PyTorch Geometric (torch_geometric), numpy, pandas, scikit-learn, networkx, joblib, pyyaml, matplotlib
**Optional**: ogb, rdkit (molecular datasets)

**Packaging**: `pyproject.toml`, `setup.py`, and `requirements.txt` are all present and tracked. Install with `./install.sh` (creates `venv/`, auto-detects CUDA/CPU PyTorch, runs `pip install -e .`) or `pip install -e .` for an editable dev install. Python 3.10+ (`pyproject.toml` enforces `<3.14`).

---

## Coding Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `FrameworkMain`, `GraphDataset`, `GCNConv` |
| Functions/methods | snake_case | `run_configurations`, `forward` |
| Constants | UPPER_SNAKE_CASE | `MANDATORY_MAIN_CONFIG_PARAMS` |
| Private members | underscore prefix | `_num_graph_nodes` |
| Imports | stdlib → 3rd party → local | `from simplegnn.models.model import GraphModel` |
| Type hints | Where present | Not comprehensive across codebase |
| Indentation | 4 spaces | |
| Docstrings | Classes and key methods | Not exhaustive; don't add to code you didn't write |

---

## Git Conventions

- **Main branch**: `main`
- **Commit messages**: Imperative mood, descriptive ("Refactor linear layer implementation to support 'aggr_channels' mode")
- **Tests**: A `pytest` suite lives in `tests/` (unit tests + import/API smoke tests). Integration experiments (example scripts and `experiments/` shell scripts) remain the end-to-end check. Run `pytest tests -q` before opening a PR.
- **Gitignored**: `data/`, `results/`, `*.csv`, `venv/`, `__pycache__/`, `.auto-claude/`, `.claude_settings.json`, `.claude/settings.local.json`, `.idea/` (untracked — it held personal SFTP deployment config, not shared settings). Note: the agent docs (`AGENTS.md`, `.claude/CLAUDE.md`) **are** tracked.

---

## Important Patterns

- **Adding a new GNN layer**: Create wrapper class extending `FrameworkLayer` in `src/simplegnn/models/layers/mpnn_classical/` → add to `LayerTypes` enum → register in `layer_loader.py` → reference in YAML model config
- **Adding a new dataset**: Implement preprocessing class in `src/simplegnn/datasets/graph_dataset_preprocessing.py` → register in `custom_datasets.py` → create split files in `src/simplegnn/datasets/splits/`
- **Grid search**: Use lists in YAML parameter files. The framework computes the cartesian product of all list-valued parameters
- **Parallel execution**: `joblib` handles parallel runs across splits/configs. `num_threads=-1` uses all CPUs
- **Results**: Saved as CSV per epoch per configuration in the results directory. Evaluation finds best config by validation metric

---

## Common Pitfalls

- Install the package (`pip install -e .`) and run from the repo root; imports use the `simplegnn.` package prefix (e.g. `from simplegnn.framework.core import FrameworkMain`)
- YAML model configs use `layer_type` string keys that must match `LayerTypes` enum values exactly
- ShareGNN preprocessing must run before training ShareGNN models (generates required node/edge properties)
- The `precision` parameter (`float`/`double`) must be consistent between data and model; mismatches cause runtime errors
