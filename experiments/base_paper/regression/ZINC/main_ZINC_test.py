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
# 5 = fresh copy of v2 (currently identical to the promoted main ZINC config),
# writing to results/base_paper/regression/ZINC_test_v5/.
# 6 = v2 architecture with the LeakyReLU nonlinearities replaced by SwiGLU
# blocks, writing to results/base_paper/regression/ZINC_test_v6/.
# 7 = v2 architecture with the positional encoder's closed_walks labels replaced
# by the monomial (edge-label multiset) refinement closed_walks_edges (specs/24),
# writing to results/base_paper/regression/ZINC_test_v7/.
# 8 = like 7, but with the cheap prime-weighted refinement closed_walks_weighted
# (specs/24 variant B) instead of the monomials, writing to
# results/base_paper/regression/ZINC_test_v8/.
# 9 = v2 with induced/simple-cycle labels replaced only in the aggregation step
# by ordinary (unlabeled) closed-walk labels of lengths 1-10, writing to
# results/base_paper/regression/ZINC_test_v9/.
# 10 = exact v2 architecture and hyperparameters except positional-encoding L1
# is disabled, writing to results/base_paper/regression/ZINC_test_v10/.
# 11 = v2 with the positional-encoder closed-walk length-1 head removed.
# 12 = v11 with rule_occurrence_threshold raised from 1 to 10.
MAIN_CONFIGS = {
    1: CONFIGS / 'main_config_ZINC_test.yml',
    2: CONFIGS / 'main_config_ZINC_test_v2.yml',
    3: CONFIGS / 'main_config_ZINC_test_v3.yml',
    4: CONFIGS / 'main_config_ZINC_test_v4.yml',
    5: CONFIGS / 'main_config_ZINC_test_v5.yml',
    6: CONFIGS / 'main_config_ZINC_test_v6.yml',
    7: CONFIGS / 'main_config_ZINC_test_v7.yml',
    8: CONFIGS / 'main_config_ZINC_test_v8.yml',
    9: CONFIGS / 'main_config_ZINC_test_v9.yml',
    10: CONFIGS / 'main_config_ZINC_test_v10.yml',
    11: CONFIGS / 'main_config_ZINC_test_v11.yml',
    12: CONFIGS / 'main_config_ZINC_test_v12.yml',
}


def main_ZINC(num_threads=-1, version=1):
    experiment = FrameworkMain(MAIN_CONFIGS[version])
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=1)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=1)
    experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
@click.option('--version', default=1, type=click.Choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
              help='Network version: 1 = test network, 2 = PE encoder + pre-norm residual (specs/15), '
                   '3 = v2 + rule_occurrence_threshold=1 + cosine-annealed L1, '
                   '4 = v2 + closed-walk node labels, '
                   '5 = copy of v2 (scratch slot for the next variant), '
                   '6 = v2 with LeakyReLU replaced by SwiGLU blocks, '
                   '7 = v2 with monomial (edge-labeled) closed walks in the PE encoder, '
                   '8 = v2 with prime-weighted (edge-labeled) closed walks in the PE encoder, '
                   '9 = v2 with unlabeled closed walks (lengths 1-10) replacing cycle labels only in aggregation, '
                   '10 = v2 with positional-encoding L1 disabled, '
                   '11 = v2 without the PE closed-walk length-1 head, '
                   '12 = v11 with rule_occurrence_threshold=10')
def main(num_threads, version):
    main_ZINC(num_threads=num_threads, version=version)


if __name__ == '__main__':
    main()
