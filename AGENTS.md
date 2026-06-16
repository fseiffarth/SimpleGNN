# Repository Guidelines

## Project Structure & Module Organization
- `src/simplegnn/`: installable package code.
- `src/simplegnn/framework/`: experiment orchestration (`FrameworkMain`, configs, evaluation helpers).
- `src/simplegnn/models/` and `src/simplegnn/models/layers/`: model definitions and layer implementations.
- `src/simplegnn/datasets/`: dataset loading, preprocessing, and split metadata (`splits/**/*.json`).
- `tests/`: regression and integration tests.
- `examples/`: runnable example experiments (`main.py`, `main.yml`, model/parameter YAMLs).
- `experiments/`: paper and benchmark scripts/configs.

## Build, Test, and Development Commands
Run commands from the repo root.
- `./install.sh`: creates `venv`, installs PyTorch (CUDA/CPU auto-detect), installs dependencies, then `pip install -e .`.
- `pip install -e .`: editable install for active development.
- `pytest tests -q`: run full automated test suite.
- `pytest tests/test_imports_and_loader_api.py -q`: fast smoke test for package import and loader API.
- `python examples/share_gnn_basic/main.py`: run a baseline local experiment.

## Coding Style & Naming Conventions
- Python 3.10+ required (`pyproject.toml` enforces `<3.14`).
- Follow PEP 8: 4-space indentation, `snake_case` for functions/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep module boundaries aligned with package structure (`simplegnn.<domain>...`) rather than ad-hoc imports.
- Prefer clear docstrings for public APIs and configuration-heavy functions.
- YAML config files should use descriptive names, e.g. `main_config_<task>.yml`, `models_<arch>.yml`.

## Testing Guidelines
- Framework: `pytest` with `test_*.py` naming under `tests/`.
- Add tests for new behavior and bug fixes in the nearest relevant file (or create a focused new `test_<feature>.py`).
- Cover both import/API stability and runtime behavior for model/dataset changes.
- Before opening a PR, run `pytest tests -q` and at least one representative example script for changed components.

## Commit & Pull Request Guidelines
Recent history favors concise, imperative commit titles (e.g., `Refactor caching logic...`, `Fix PYTHONPATH...`).
- Keep commits focused and self-contained.
- Use message pattern: `<Verb> <component> <intent>`.
- PRs should include: problem statement, summary of changes, test evidence (commands + outcomes), and any config/data impact.
- Link related specs/issues and include logs or metrics when behavior/performance changes.

## Security & Configuration Tips
- Do not commit generated artifacts (`results/`, local caches, virtual environments).
- Keep environment-specific paths out of committed YAML.
- Validate new dataset/model YAML against existing loader conventions before merge.
