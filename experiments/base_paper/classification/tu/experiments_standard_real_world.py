from pathlib import Path
import click

from src.Experiment.ExperimentMain import ExperimentMain


def get_existing_splits():
    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("paper_experiments/Data//SplitsSimple").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]:
        source_path = Path("Data/SplitsSimple").joinpath(f"{split}_splits.json")
        target_path = Path("paper_experiments/Data//SplitsSimple").joinpath(f"{split}_splits.json")
        target_path.write_text(source_path.read_text())


def main_standard_real_world(num_threads=-1):
    get_existing_splits()

    experiment = ExperimentMain(Path('paper_experiments/classification/configs/main_config_sota_comparison.yml'))
    experiment.ExperimentPreprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results(evaluate_validation_only=True)

    experiment = ExperimentMain(Path('paper_experiments/classification/configs/main_config_sota_random_comparison.yml'))
    experiment.ExperimentPreprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results(evaluate_validation_only=True)

@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_standard_real_world(num_threads)



if __name__ == '__main__':
    main()
