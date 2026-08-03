## Peptides-func graph-classification-as-regression experiment (ShareGNN
## multi-head), Long Range Graph Benchmark. See main_config_peptides_func.yml
## for the task=graph_regression + BCE workaround rationale.
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain


def main_peptides_func(num_threads=-1):
    experiment = FrameworkMain(Path('experiments/lrgb/configs/main_config_peptides_func.yml'))
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_peptides_func(num_threads=num_threads)


if __name__ == '__main__':
    main()
