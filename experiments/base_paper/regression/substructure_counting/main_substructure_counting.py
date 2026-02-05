## Real World Data
from pathlib import Path

import click
from src.Experiment.ExperimentMain import ExperimentMain

def main_counting(num_threads=-1):
    experiment = ExperimentMain(Path('paper_experiments/regression/substructure_counting/configs/main_config_substructure_counting.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    ## run real world experiment
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


    ### Additional evaluation of the experiment with dataset substructure_counting
    ### Here we optimized against the 6-dim output, now calculate the MAE for each entry
    # load the models
    outputs, labels, accuracy = experiment.evaluate_model("multi", best=False)
    # get the MAE per column
    mae_per_column = [] * len(outputs[0])
    for i in range(len(outputs[0])):
        mae = sum(abs(outputs[:, i] - labels[:, i])) / len(outputs)
        mae_per_column.append(mae)
    # print the MAE (triangle, tri_tail, star, cycle4, cycle5, cycle6)
    column_names = ['triangle', 'tri_tail', 'star', 'cycle4', 'cycle5', 'cycle6']
    for i, mae in enumerate(mae_per_column):
        print(f'MAE for {column_names[i]}: {mae:.4f}')


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_counting(num_threads)



if __name__ == '__main__':
    main()