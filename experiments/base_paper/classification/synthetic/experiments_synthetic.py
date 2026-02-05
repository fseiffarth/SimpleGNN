from pathlib import Path
import click

from src.Experiment.ExperimentMain import ExperimentMain


def get_existing_splits():
    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("paper_experiments/Data/").mkdir(exist_ok=True)
    Path("paper_experiments/Data//Splits").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["CSL", "Snowflakes", "LongRings100", "EvenOddRings2_16", "EvenOddRingsCount16"]:
        source_path = Path("Data/Splits").joinpath(f"{split}_splits.json")
        target_path = Path("paper_experiments/Data//Splits").joinpath(f"{split}_splits.json")
        target_path.write_text(source_path.read_text())

def main_synthetic(num_threads=-1):
    ## Synthetic Data
    experiment_synthetic = ExperimentMain(Path('paper_experiments/classification/configs/main_config_fair_synthetic.yml'))
    experiment_synthetic.ExperimentPreprocessing(num_threads=1)

    ## run synthetic experiment
    experiment_synthetic.run_configurations(num_threads=num_threads)
    experiment_synthetic.evaluate_results()
    experiment_synthetic.run_best_configuration(num_threads=num_threads)
    experiment_synthetic.evaluate_results(evaluate_best_model=True)

    experiment_synthetic = ExperimentMain(Path('paper_experiments/classification/configs/main_config_fair_synthetic_random_variation.yml'))
    experiment_synthetic.ExperimentPreprocessing(num_threads=1)
    experiment_synthetic.run_configurations(num_threads=num_threads)
    experiment_synthetic.evaluate_results()
    experiment_synthetic.run_best_configuration(num_threads=num_threads)
    experiment_synthetic.evaluate_results(evaluate_best_model=True)

    experiment_synthetic = ExperimentMain(Path('paper_experiments/classification/configs/main_config_fair_synthetic_only_encoder.yml'))
    experiment_synthetic.ExperimentPreprocessing(num_threads=1)
    experiment_synthetic.run_configurations(num_threads=num_threads)
    experiment_synthetic.evaluate_results()
    experiment_synthetic.run_best_configuration(num_threads=num_threads)
    experiment_synthetic.evaluate_results(evaluate_best_model=True)

    experiment_synthetic = ExperimentMain(Path('paper_experiments/classification/configs/main_config_fair_synthetic_only_decoder.yml'))
    experiment_synthetic.ExperimentPreprocessing(num_threads=1)
    experiment_synthetic.run_configurations(num_threads=num_threads)
    experiment_synthetic.evaluate_results()
    experiment_synthetic.run_best_configuration(num_threads=num_threads)
    experiment_synthetic.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_synthetic(num_threads)




if __name__ == '__main__':
    main()