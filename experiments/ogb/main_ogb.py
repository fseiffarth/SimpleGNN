## OGB graph-property-prediction experiments (ShareGNN multi-head).
##
## Covers the single-task `ogbg-mol*` datasets. The multi-task members of the
## family (moltox21, molsider, molclintox, moltoxcast, molmuv, molpcba) are not
## included: their targets are NaN-masked matrices, and the framework's
## classification/regression heads take a single target per graph.
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain

CONFIG_DIR = Path('experiments/ogb/configs')

# short name -> main config, mirroring simplegnn.utils.ogb_splits.SUPPORTED_DATASETS
DATASETS = {
    'molhiv': CONFIG_DIR / 'main_molhiv.yml',
    'molbace': CONFIG_DIR / 'main_molbace.yml',
    'molbbbp': CONFIG_DIR / 'main_molbbbp.yml',
    'molesol': CONFIG_DIR / 'main_molesol.yml',
    'molfreesolv': CONFIG_DIR / 'main_molfreesolv.yml',
    'mollipo': CONFIG_DIR / 'main_mollipo.yml',
    # molhiv overfitting sweep: identical apart from rule_occurrence_threshold,
    # which is read as a scalar in the layer and so cannot be a grid axis.
    # Each arm writes to its own results root; see the config headers.
    'molhiv-t1': CONFIG_DIR / 'main_molhiv_t1.yml',
    'molhiv-t50': CONFIG_DIR / 'main_molhiv_t50.yml',
    # optimal arms: tuned models/parameters pairs (see OPTIMAL_CONFIG.md);
    # same datasets/splits, results land in results/ogb_optimal/. The
    # <dataset>-<variant> naming keeps run_ogb.sh's split generation working.
    'molhiv-optimal': CONFIG_DIR / 'main_molhiv_optimal.yml',
    'molbace-optimal': CONFIG_DIR / 'main_molbace_optimal.yml',
    'molbbbp-optimal': CONFIG_DIR / 'main_molbbbp_optimal.yml',
    'molesol-optimal': CONFIG_DIR / 'main_molesol_optimal.yml',
    'molfreesolv-optimal': CONFIG_DIR / 'main_molfreesolv_optimal.yml',
    'mollipo-optimal': CONFIG_DIR / 'main_mollipo_optimal.yml',
}


def main_ogb(dataset='molhiv', num_threads=-1):
    """Run the full grid-search -> select -> rerun-best -> test pipeline."""
    config_path = DATASETS[dataset]
    experiment = FrameworkMain(config_path)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--dataset', default='molhiv', type=click.Choice(sorted(DATASETS)),
              help='Which ogbg-mol* dataset to run (default: molhiv)')
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(dataset, num_threads):
    main_ogb(dataset=dataset, num_threads=num_threads)


if __name__ == '__main__':
    main()
