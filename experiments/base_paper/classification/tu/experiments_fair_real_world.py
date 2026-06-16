## Fair real-world TU graph-classification experiments (migrated to FrameworkMain).
## Splits are referenced directly from src/simplegnn/datasets/splits/fair/ ;
## the old Data/Splits copy helper has been removed.
##
## Two model-selection protocols are available:
##   * global (default): one best hyperparameter config shared by every fold
##     (evaluate_results / run_best_configuration).
##   * Errica-style per-fold (--fair-selection): an independent model selection
##     inside each outer fold (evaluate_results_fair / run_best_configuration_fair),
##     reporting the test estimate as mean +/- std across folds. Outputs use
##     distinct filenames (summary_fair*.csv, Best_Configuration_Fair_* results),
##     so both protocols can reuse the same grid search side by side.
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain

CONFIG_DIR = Path('experiments/base_paper/classification/configs')
MAIN_CONFIGS = [
    'main_config_fair_real_world.yml',
    'main_config_fair_real_world_random_variation.yml',
    'main_config_fair_real_world_only_encoder.yml',
    'main_config_fair_real_world_only_decoder.yml',
]


def main_fair_real_world(num_threads=-1, fair_selection=False):
    for config_name in MAIN_CONFIGS:
        experiment = FrameworkMain(CONFIG_DIR / config_name)
        experiment.preprocessing(num_threads=1)
        experiment.run_configurations(num_threads=num_threads)
        if fair_selection:
            # Errica-style per-fold model selection and assessment.
            experiment.evaluate_results_fair()
            experiment.run_best_configuration_fair(num_threads=num_threads)
            experiment.evaluate_results_fair(evaluate_best_model=True)
        else:
            # Global (single best config) model selection.
            experiment.evaluate_results()
            experiment.run_best_configuration(num_threads=num_threads)
            experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
@click.option('--fair-selection/--no-fair-selection', default=False,
              help='Use Errica-style per-fold model selection instead of the global one')
def main(num_threads, fair_selection):
    main_fair_real_world(num_threads=num_threads, fair_selection=fair_selection)


if __name__ == '__main__':
    main()
