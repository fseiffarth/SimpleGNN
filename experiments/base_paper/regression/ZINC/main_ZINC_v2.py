## ZINC readout/architecture variants (specs/09-zinc-readout-and-architecture-improvements.md).
##
## Run a variant against the baseline of main_ZINC.py, which stays untouched:
##   python experiments/base_paper/regression/ZINC/main_ZINC_v2.py --variant small
##
##   small       Stage A + C: the readout shrunk via F (100 -> 32) and H (140 -> 56),
##               dropout before it, weight decay grid, AdamW. ~179k readout params.
##   labels      small + the invariants the baseline leaves off (WL depth 1/2,
##               combined conv labels, edge_label_distances cutoff 2, betweenness).
##   factorized  Stage B: baseline capacity, but the aggregation keeps its (H, F)
##               structure and the readout is a rank-32 CP factorization (~11k params).
##   attention   baseline capacity, unflattened aggregation, and the readout is
##               attention over the 140 head embeddings; grid-searches three
##               variants (gated pooling / PMA seed queries / transformer block).
##   attention_thresh5, attention_thresh10
##               attention, but rule_occurrence_threshold 2 -> 5 / 10, to shrink
##               the message-passing layer's 405,493 params (57-67% of the
##               model). Separate files, not a grid: run_configuration.py does
##               not treat rule_occurrence_threshold as a grid-searchable key.
##   attention_smoothl1
##               attention, but trained with SmoothL1 (Huber) loss instead of
##               MAE; ValidationMAE/TestMAE are still the task metric. Kept out
##               of the weight_decay grid so model selection stays valid (its
##               tie-break compares ValidationLoss, not comparable across loss
##               functions).
##
## All four base architectures (small/labels/factorized/attention) also had
## their first embedding layer shrunk from out_features 10 to 4.
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain

VARIANTS = ('attention', 'attention_thresh5', 'attention_thresh10', 'attention_smoothl1',
           'small', 'labels', 'factorized')


def main_ZINC_v2(variant='small', num_threads=-1, config_id=None):
    config = Path(f'experiments/base_paper/regression/ZINC/configs/main_config_ZINC_{variant}.yml')
    experiment = FrameworkMain(config)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads, config_ids=config_id)
    experiment.evaluate_results()
    if config_id is None:
        # only re-run the best configuration when the full grid was searched
        experiment.run_best_configuration(num_threads=1)
        experiment.evaluate_results(evaluate_best_model=True)


@click.command()
@click.option('--variant', type=click.Choice(VARIANTS), default='attention', help='Which architecture variant to run')
@click.option('--num_threads', default=-1, help='Number of threads to use')
@click.option('--config_id', type=int, default=None,
              help='Run only this single grid configuration index (0-based). '
                   'Omit to run the full hyperparameter grid.')
def main(variant, num_threads, config_id):
    main_ZINC_v2(variant=variant, num_threads=num_threads, config_id=config_id)


if __name__ == '__main__':
    main()
