## ZINC -> TU hash-keyed transfer experiment (spec 18): pretrain a ShareGNN
## regressor on ZINC using only canonical structural labels (closed_walks,
## induced_cycles, simple_cycles, wl [unlabeled]; see
## configs/network_transfer_ZINC_pretrain.yml), then "zero-shot" (linear-probe)
## warm-start a graph classifier on each TU target from that checkpoint via the
## transfer: block in configs/parameters_transfer_TU_linear_probe.yml.
##
## Run order: pretrain once, then finetune per target. `--stage overlap` runs
## the spec 18 B0 go/no-go label-vocabulary overlap check (against
## examples/transfer_learning/measure_overlap.py) BEFORE spending a full
## pretraining run -- if weighted coverage is single-digit percent for every
## label type, hash-keyed transfer will not help for that target.
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIGS = Path('experiments/base_paper/transfer/ZINC_to_TU/configs')
PRETRAIN_CONFIG = CONFIGS / 'main_config_ZINC_pretrain.yml'

TU_TARGETS = ['MUTAG', 'NCI1', 'NCI109', 'Mutagenicity', 'DHFR']


def _finetune_config(target):
    path = CONFIGS / f'main_config_{target}_finetune.yml'
    if not path.exists():
        raise click.BadParameter(
            f"no finetune config for target {target!r} at {path}. "
            f"Known targets: {TU_TARGETS} (add a main_config_<name>_finetune.yml "
            f"for any other TUDataset name following the same pattern).")
    return path


def run_pretrain(num_threads=-1):
    experiment = FrameworkMain(PRETRAIN_CONFIG)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


def run_finetune(target, num_threads=-1):
    experiment = FrameworkMain(_finetune_config(target))
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


def run_overlap_report(target):
    import sys
    if str(REPO_ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / 'src'))
    sys.path.insert(0, str(REPO_ROOT / 'examples' / 'transfer_learning'))
    from measure_overlap import main as measure_overlap_main  # noqa: E402

    measure_overlap_main([
        '--source-labels', 'data/ZINC/labels/ZINC',
        '--target-labels', f'data/TUDatasets/labels/{target}',
    ])


@click.command()
@click.option('--stage', default='all',
              type=click.Choice(['overlap', 'pretrain', 'finetune', 'all']),
              help='overlap = spec 18 B0 go/no-go check only (needs preprocessing() already run on '
                   'both datasets); pretrain = ZINC only; finetune = TU target(s) only (needs an '
                   'existing ZINC checkpoint); all = pretrain then finetune every target.')
@click.option('--target', default=None,
              help=f'TU dataset to finetune/check (one of {TU_TARGETS}); required for '
                   f'--stage finetune/overlap, ignored otherwise (finetune loops over all targets).')
@click.option('--num_threads', default=-1, help='Number of threads to use for run_configurations')
def main(stage, target, num_threads):
    if stage == 'overlap':
        if target is None:
            raise click.BadParameter('--target is required for --stage overlap')
        run_overlap_report(target)
        return

    if stage in ('pretrain', 'all'):
        run_pretrain(num_threads=num_threads)

    if stage in ('finetune', 'all'):
        targets = [target] if target else TU_TARGETS
        for t in targets:
            run_finetune(t, num_threads=num_threads)


if __name__ == '__main__':
    main()
