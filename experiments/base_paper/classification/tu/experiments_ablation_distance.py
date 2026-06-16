## Distance-ablation TU graph-classification experiment (migrated to FrameworkMain).
## Splits are referenced directly from src/simplegnn/datasets/splits/fair/ ;
## the old Data/Splits copy helper has been removed.
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain


def main_ablation_distance(num_threads=-1):
    ablation_experiment = FrameworkMain(
        Path('experiments/base_paper/classification/configs/ablation/distances/main_config_ablation_distances.yml'))
    ablation_experiment.preprocessing(num_threads=1)
    ablation_experiment.run_configurations(num_threads=num_threads)
    ablation_experiment.evaluate_results()
    ablation_experiment.run_best_configuration(num_threads=num_threads)
    ablation_experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_ablation_distance(num_threads)


if __name__ == '__main__':
    main()
