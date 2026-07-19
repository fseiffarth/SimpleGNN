## Substructure -> ZINC hash-keyed transfer experiment (spec 18): train on
## synthetic, zero-shot on real-world. Pretrain a ShareGNN regressor on the
## SubstructureBenchmark "multi" target (6-dim: triangle/tri_tail/star/
## cycle4/cycle5/cycle6 counts) using the ripped-ZINC-v2 canonical backbone
## (see configs/network_transfer_Substructure_pretrain.yml), then "zero-shot"
## (linear-probe) warm-start a ZINC regressor from that checkpoint via the
## transfer: block in configs/parameters_transfer_ZINC_linear_probe.yml.
##
## Run order: pretrain once, then finetune. `--stage overlap` runs the spec 18
## B0 go/no-go label-vocabulary overlap check (against
## examples/transfer_learning/measure_overlap.py) BEFORE spending a full
## pretraining run.
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIGS = Path('experiments/base_paper/transfer/Substructure_to_ZINC/configs')
PRETRAIN_CONFIG = CONFIGS / 'main_config_Substructure_pretrain.yml'
FINETUNE_CONFIG = CONFIGS / 'main_config_ZINC_finetune.yml'


def run_pretrain(num_threads=-1):
    experiment = FrameworkMain(PRETRAIN_CONFIG)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


def run_finetune(num_threads=-1):
    experiment = FrameworkMain(FINETUNE_CONFIG)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


def run_overlap_report():
    import sys
    if str(REPO_ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / 'src'))
    sys.path.insert(0, str(REPO_ROOT / 'examples' / 'transfer_learning'))
    from measure_overlap import main as measure_overlap_main  # noqa: E402

    measure_overlap_main([
        '--source-labels', 'data/SubstructureBenchmark/labels/multi',
        '--target-labels', 'data/ZINC/labels/ZINC',
        '--label-types',
        'simple_cycles_5_5', 'simple_cycles_6_6', 'simple_cycles_7_7', 'simple_cycles_8_8',
        'simple_cycles_9_9', 'simple_cycles_10_10',
        'induced_cycles_5_5', 'induced_cycles_6_6', 'induced_cycles_7_7', 'induced_cycles_8_8',
        'induced_cycles_9_9', 'induced_cycles_10_10',
        'closed_walks_1_1', 'closed_walks_2_2', 'closed_walks_3_3', 'closed_walks_4_4',
        'closed_walks_5_5', 'closed_walks_6_6', 'closed_walks_7_7', 'closed_walks_8_8',
        'closed_walks_9_9', 'closed_walks_10_10',
        'wl_0',
    ])


@click.command()
@click.option('--stage', default='all',
              type=click.Choice(['overlap', 'pretrain', 'finetune', 'all']),
              help='overlap = spec 18 B0 go/no-go check only (needs preprocessing() already run on '
                   'both datasets); pretrain = SubstructureBenchmark only; finetune = ZINC only '
                   '(needs an existing SubstructureBenchmark checkpoint); all = pretrain then finetune.')
@click.option('--num_threads', default=-1, help='Number of threads to use for run_configurations')
def main(stage, num_threads):
    if stage == 'overlap':
        run_overlap_report()
        return

    if stage in ('pretrain', 'all'):
        run_pretrain(num_threads=num_threads)

    if stage in ('finetune', 'all'):
        run_finetune(num_threads=num_threads)


if __name__ == '__main__':
    main()
