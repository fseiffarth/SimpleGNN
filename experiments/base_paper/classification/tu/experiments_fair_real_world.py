## Fair real-world TU graph-classification experiments (migrated to FrameworkMain).
## Splits are referenced directly from src/simplegnn/datasets/splits/fair/ ;
## the old Data/Splits copy helper has been removed.
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


def main_fair_real_world(num_threads=-1):
    for config_name in MAIN_CONFIGS:
        experiment = FrameworkMain(CONFIG_DIR / config_name)
        experiment.preprocessing(num_threads=1)
        experiment.run_configurations(num_threads=num_threads)
        experiment.evaluate_results()
        experiment.run_best_configuration(num_threads=num_threads)
        experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_fair_real_world(num_threads=num_threads)


if __name__ == '__main__':
    main()
