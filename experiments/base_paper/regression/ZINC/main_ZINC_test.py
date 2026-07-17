## ZINC graph-regression experiment (migrated to FrameworkMain).
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain

CONFIGS = Path('experiments/base_paper/regression/ZINC/configs')

# --version selects the main config: 1 = the plain test network, 2 = specs/15
# variant (invariant-based positional encoding as node encoder + pre-norm
# residual conv), writing results to results/base_paper/regression/ZINC_test_v2/.
# 3 = v2 architecture with rule_occurrence_threshold=1 (no count-based pruning;
# proximal L1 does loss-driven rule selection) + cosine-annealing lr that anneals
# the L1 threshold, writing to results/base_paper/regression/ZINC_test_v3/.
# 4 = v2 architecture with cycle-based node labels replaced by closed-walk labels
# ((A^k)_ii diagonals, k=5-10), writing to results/base_paper/regression/ZINC_test_v4/.
MAIN_CONFIGS = {
    1: CONFIGS / 'main_config_ZINC_test.yml',
    2: CONFIGS / 'main_config_ZINC_test_v2.yml',
    3: CONFIGS / 'main_config_ZINC_test_v3.yml',
    4: CONFIGS / 'main_config_ZINC_test_v4.yml',
}


def main_ZINC(num_threads=-1, version=1):
    experiment = FrameworkMain(MAIN_CONFIGS[version])
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
@click.option('--version', default=1, type=click.Choice([1, 2, 3, 4]),
              help='Network version: 1 = test network, 2 = PE encoder + pre-norm residual (specs/15), '
                   '3 = v2 + rule_occurrence_threshold=1 + cosine-annealed L1, '
                   '4 = v2 + closed-walk node labels')
def main(num_threads, version):
    main_ZINC(num_threads=num_threads, version=version)


if __name__ == '__main__':
    main()
