## Real World Data
from pathlib import Path

import click
import joblib

from src.Experiment.ExperimentMain import ExperimentMain

def main_ZINC(num_threads=-1):
    experiment = ExperimentMain(Path(f'paper_experiments/regression/ZINC/configs/main_config_ZINC_test.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    ## run real world experiment
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)

@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    # parallelize over thresholds
    main_ZINC(num_threads=num_threads)



if __name__ == '__main__':
    main()