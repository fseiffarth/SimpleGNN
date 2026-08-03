## QM9 graph-regression experiment (ShareGNN multi-head).
##
## Reconstructed from the pre-refactor RuleGNN experiment
## `ReproduceExtended/main_QM.py` and migrated to FrameworkMain.
## See specs/19-qm9-migration.md.
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain


def main_QM9(num_threads=-1):
    experiment = FrameworkMain(Path('experiments/qm9/configs/main_config_QM9.yml'))
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_QM9(num_threads=num_threads)


if __name__ == '__main__':
    main()
