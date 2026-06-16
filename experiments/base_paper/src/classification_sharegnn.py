"""Run the full ShareGNN classification suite (migrated).

Orchestrates the migrated sibling experiment scripts. Splits are now referenced
directly from the repository, so the old Data/Splits copy helper is gone, and the
deprecated get_gnn_comparison_data step has been dropped.

Run from the repository root (so the relative config paths resolve):
    python experiments/base_paper/src/classification_sharegnn.py --num_threads -1
"""
import importlib.util
from pathlib import Path

import click

BASE = Path(__file__).resolve().parent.parent  # experiments/base_paper


def _load(relpath):
    """Load a migrated sibling script by file path (base_paper is not a package)."""
    path = BASE / relpath
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@click.command()
@click.option('--num_threads', default=-1, help='Number of tasks to run in parallel')
def main(num_threads):
    experiments_synthetic = _load('classification/synthetic/experiments_synthetic.py')
    experiments_fair_real_world = _load('classification/tu/experiments_fair_real_world.py')
    experiments_standard_real_world = _load('classification/tu/experiments_standard_real_world.py')
    experiments_ablation_distance = _load('classification/tu/experiments_ablation_distance.py')
    experiments_ablation_threshold = _load('classification/tu/experiments_ablation_threshold.py')

    experiments_synthetic.main_synthetic(num_threads)
    experiments_fair_real_world.main_fair_real_world(num_threads)
    experiments_standard_real_world.main_standard_real_world(num_threads)
    experiments_ablation_distance.main_ablation_distance(num_threads)
    experiments_ablation_threshold.main_ablation_threshold(num_threads)

    # Result tables / plots (optional post-processing).
    latex = _load('src/latex.py')
    latex_plots = _load('src/latex_plots.py')
    latex.main()
    latex_plots.main()


if __name__ == '__main__':
    main()
