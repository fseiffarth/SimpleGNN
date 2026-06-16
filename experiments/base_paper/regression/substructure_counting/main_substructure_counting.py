## Substructure-counting graph-regression experiment (migrated to FrameworkMain).
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain


def main_counting(num_threads=-1):
    experiment = FrameworkMain(
        Path('experiments/base_paper/regression/substructure_counting/configs/main_config_substructure_counting.yml'))
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)

    ### Additional evaluation of the experiment with dataset "multi":
    ### we optimized against the 6-dim output, now report the MAE for each entry.
    outputs, labels, accuracy = experiment.evaluate_model("multi", best=False)
    column_names = ['triangle', 'tri_tail', 'star', 'cycle4', 'cycle5', 'cycle6']
    for i in range(len(column_names)):
        mae = sum(abs(outputs[:, i] - labels[:, i])) / len(outputs)
        print(f'MAE for {column_names[i]}: {mae:.4f}')


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_counting(num_threads)


if __name__ == '__main__':
    main()
