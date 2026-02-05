from pathlib import Path

import click

from paper_experiments import  get_gnn_comparison_data, latex, latex_plots
from paper_experiments.classification.synthetic import experiments_synthetic
from paper_experiments.classification.tu import experiments_fair_real_world, experiments_standard_real_world, \
    experiments_ablation_distance, experiments_ablation_threshold


def get_existing_splits():
    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("paper_experiments/Data").mkdir(exist_ok=True)
    Path("paper_experiments/Data/Splits").mkdir(exist_ok=True)
    # copy all split files from "Data/Splits" to "paper_experiments/Data/Splits"
    for file in Path("Data/Splits").glob("*.json"):
        target_path = Path("paper_experiments/Data/Splits").joinpath(file.name)
        target_path.write_text(file.read_text())

    # copy all split files from "Data/SplitsSimple" to "paper_experiments/Data/SplitsSimple"
    Path("paper_experiments/Data/SplitsSimple").mkdir(exist_ok=True)
    for file in Path("Data/SplitsSimple").glob("*.json"):
        target_path = Path("paper_experiments/Data/SplitsSimple").joinpath(file.name)
        target_path.write_text(file.read_text())


# add arguments to the main function if needed using click
@click.command()
@click.option('--num_threads', default=-1, help='Number of tasks to run in parallel')
def main(num_threads):
    get_existing_splits()
    experiments_synthetic.main_synthetic(num_threads)
    experiments_fair_real_world.main_fair_real_world(num_threads)
    experiments_standard_real_world.main_standard_real_world(num_threads)
    experiments_ablation_distance.main_ablation_distance(num_threads)
    experiments_ablation_threshold.main_ablation_threshold(num_threads)
    #experiments_baseline.main_baseline(num_threads)
    get_gnn_comparison_data.main()
    latex.main()
    latex_plots.main()


if __name__ == '__main__':
    main()
