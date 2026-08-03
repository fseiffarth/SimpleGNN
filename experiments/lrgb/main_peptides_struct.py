## Peptides-struct graph-regression experiment (ShareGNN multi-head), Long
## Range Graph Benchmark.
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain


def main_peptides_struct(num_threads=-1):
    experiment = FrameworkMain(Path('experiments/lrgb/configs/main_config_peptides_struct.yml'))
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_peptides_struct(num_threads=num_threads)


if __name__ == '__main__':
    main()
