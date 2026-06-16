## ZINC graph-regression experiment (migrated to FrameworkMain).
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain


def main_ZINC(num_threads=-1):
    experiment = FrameworkMain(Path('experiments/base_paper/regression/ZINC/configs/main_config_ZINC.yml'))
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_ZINC(num_threads=num_threads)


if __name__ == '__main__':
    main()
