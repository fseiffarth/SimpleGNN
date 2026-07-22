## Substructure -> TU hash-keyed transfer experiment (spec 18): train on
## synthetic, zero-shot on real-world. Pretrain a ShareGNN regressor on the
## SubstructureBenchmark "multi" target (6-dim: triangle/tri_tail/star/
## cycle4/cycle5/cycle6 counts) using the ripped-ZINC-v2 canonical backbone
## (see configs/network_transfer_Substructure_pretrain.yml), then "zero-shot"
## (linear-probe) warm-start a TU graph classifier from that checkpoint via
## the transfer: block in configs/parameters_transfer_TU_linear_probe.yml.
##
## Run order: pretrain once, then finetune (all 5 TU targets run together,
## same as ZINC_to_TU's driver -- FrameworkMain does not loop them for us here
## since each target has its own main_config file; this driver loops
## explicitly). `--stage overlap` runs the spec 18 B0 go/no-go
## label-vocabulary overlap check (against
## examples/transfer_learning/measure_overlap.py) BEFORE spending a full
## pretraining run.
import importlib.util
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIGS = Path('experiments/base_paper/transfer/Substructure_to_TU/configs')
PRETRAIN_CONFIG = CONFIGS / 'main_config_Substructure_pretrain.yml'
RESULTS = Path('results/base_paper/transfer/Substructure_to_TU')

TU_TARGETS = ['MUTAG', 'NCI1', 'NCI109', 'Mutagenicity', 'DHFR']

# arm name -> results subdirectory, in the order they appear in the report table
ARMS = [('pretrained probe (transfer)', 'finetune'),
        ('random-init probe', 'baseline_random_probe'),
        ('trained from scratch', 'scratch')]

REPORT_NOTES = [
    'All three arms share `network_transfer_TU_finetune.yml`, the same splits and the same '
    'hyperparameters; they differ only in where the backbone weights come from and what is frozen.',
    '**pretrained probe**: backbone transferred hash-keyed from the SubstructureBenchmark '
    '("multi") pretraining run and frozen; only the re-initialized classification head trains.',
    '**random-init probe**: identical, but `transfer.random_init: true` -- the frozen backbone '
    'keeps its random init, isolating what the *pretrained* weights add over a random projection '
    'of the same invariants.',
    '**trained from scratch**: no transfer block, whole backbone trained. Upper reference point; '
    'the probes are the cheap arms, so read accuracy against the runtime columns.',
]


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


def _stage_config(target, suffix):
    path = CONFIGS / f'main_config_{target}_{suffix}.yml'
    if not path.exists():
        raise click.BadParameter(
            f"no {suffix} config for target {target!r} at {path}. "
            f"Add a main_config_<name>_{suffix}.yml following the NCI1 pattern.")
    return path


def run_experiment(config_path, num_threads=-1):
    experiment = FrameworkMain(config_path)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


def run_baseline(target, num_threads=-1):
    """Untrained backbone (random init, frozen), head-only training."""
    run_experiment(_stage_config(target, 'baseline'), num_threads=num_threads)


def run_scratch(target, num_threads=-1):
    """Whole backbone trained from a fresh init, no transfer."""
    run_experiment(_stage_config(target, 'scratch'), num_threads=num_threads)


def _transfer_report_module():
    """Load the shared report renderer (experiments/base_paper is not a package)."""
    path = REPO_ROOT / 'experiments/base_paper/src/transfer_report.py'
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report(target):
    """
    Render the cross-arm markdown report for ``target`` from whatever result
    directories exist; arms that have not been run yet are listed as *not run*.
    Called automatically at the end of every finetune/baseline/scratch stage.
    """
    transfer_report = _transfer_report_module()
    return transfer_report.write_report(
        title=f'Substructure -> {target} transfer report',
        arm_specs=[(name, RESULTS / subdir / target) for name, subdir in ARMS],
        output_path=RESULTS / f'report_{target}.md',
        task='graph_classification',
        pretrain_dir=(RESULTS / 'pretrain' / 'multi'),
        notes=REPORT_NOTES)


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
        '--source-labels', 'data/SubstructureBenchmark/labels/multi',
        '--target-labels', f'data/TUDatasets/labels/{target}',
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
              type=click.Choice(['overlap', 'pretrain', 'finetune', 'baseline', 'scratch',
                                 'report', 'all']),
              help='overlap = spec 18 B0 go/no-go check only (needs preprocessing() already run on '
                   'both datasets); pretrain = SubstructureBenchmark only; finetune = TU target(s) '
                   'only (needs an existing SubstructureBenchmark checkpoint); baseline = '
                   'untrained-backbone probe (random init, frozen, head-only training) as the '
                   'direct comparison point for the transfer runs; scratch = whole backbone '
                   'trained from a fresh init, no transfer; report = only re-render the markdown '
                   'report from existing results; all = pretrain then finetune every target. '
                   'Every result-producing stage refreshes results/.../report_<target>.md.')
@click.option('--target', default=None,
              type=click.Choice(TU_TARGETS),
              help=f'TU dataset to finetune/check (one of {TU_TARGETS}); required for '
                   f'--stage overlap, ignored otherwise (finetune loops over all targets unless '
                   f'--target is given).')
@click.option('--num_threads', default=-1, help='Number of threads to use for run_configurations')
def main(stage, target, num_threads):
    if stage == 'overlap':
        if target is None:
            raise click.BadParameter('--target is required for --stage overlap')
        run_overlap_report(target)
        return

    if stage in ('pretrain', 'all'):
        run_pretrain(num_threads=num_threads)

    targets = [target] if target else TU_TARGETS

    if stage in ('finetune', 'all'):
        for t in targets:
            run_finetune(t, num_threads=num_threads)

    if stage == 'baseline':
        for t in targets:
            run_baseline(t, num_threads=num_threads)

    if stage == 'scratch':
        for t in targets:
            run_scratch(t, num_threads=num_threads)

    # every stage that produced target results refreshes the markdown report
    if stage in ('finetune', 'baseline', 'scratch', 'all', 'report'):
        for t in targets:
            write_report(t)


if __name__ == '__main__':
    main()
