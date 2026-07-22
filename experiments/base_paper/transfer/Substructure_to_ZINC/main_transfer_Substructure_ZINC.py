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
## pretraining run. `--stage baseline`/`--stage scratch` run the two
## comparison arms (untrained-backbone probe, train-from-scratch) that the
## transfer run is judged against; `--stage report` re-renders the cross-arm
## markdown report from whatever result directories already exist.
import importlib.util
from pathlib import Path

import click

from simplegnn.framework.core import FrameworkMain

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIGS = Path('experiments/base_paper/transfer/Substructure_to_ZINC/configs')
PRETRAIN_CONFIG = CONFIGS / 'main_config_Substructure_pretrain.yml'
FINETUNE_CONFIG = CONFIGS / 'main_config_ZINC_finetune.yml'
RESULTS = Path('results/base_paper/transfer/Substructure_to_ZINC')

# arm name -> results subdirectory, in the order they appear in the report table
ARMS = [('pretrained probe (transfer)', 'finetune'),
        ('random-init probe', 'baseline_random_probe'),
        ('trained from scratch', 'scratch')]

REPORT_NOTES = [
    'All three arms share `network_transfer_ZINC_finetune.yml`, the same splits and the same '
    'hyperparameters; they differ only in where the backbone weights come from and what is frozen.',
    '**pretrained probe**: backbone transferred hash-keyed from the SubstructureBenchmark '
    '("multi") pretraining run and frozen; only the re-initialized regression head trains.',
    '**random-init probe**: identical, but `transfer.random_init: true` -- the frozen backbone '
    'keeps its random init, isolating what the *pretrained* weights add over a random projection '
    'of the same invariants.',
    '**trained from scratch**: no transfer block, whole backbone trained. Upper reference point; '
    'the probes are the cheap arms, so read MAE against the runtime columns.',
]


def _stage_config(suffix):
    path = CONFIGS / f'main_config_ZINC_{suffix}.yml'
    if not path.exists():
        raise click.BadParameter(f"no {suffix} config at {path}.")
    return path


def run_experiment(config_path, num_threads=-1):
    experiment = FrameworkMain(config_path)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


def run_pretrain(num_threads=-1):
    experiment = FrameworkMain(PRETRAIN_CONFIG)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)


def run_finetune(num_threads=-1):
    run_experiment(FINETUNE_CONFIG, num_threads=num_threads)


def run_baseline(num_threads=-1):
    """Untrained backbone (random init, frozen), head-only training."""
    run_experiment(_stage_config('baseline'), num_threads=num_threads)


def run_scratch(num_threads=-1):
    """Whole backbone trained from a fresh init, no transfer."""
    run_experiment(_stage_config('scratch'), num_threads=num_threads)


def _transfer_report_module():
    """Load the shared report renderer (experiments/base_paper is not a package)."""
    path = REPO_ROOT / 'experiments/base_paper/src/transfer_report.py'
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report():
    """
    Render the cross-arm markdown report from whatever result directories
    exist; arms that have not been run yet are listed as *not run*. Called
    automatically at the end of every finetune/baseline/scratch stage.
    """
    transfer_report = _transfer_report_module()
    return transfer_report.write_report(
        title='Substructure -> ZINC transfer report',
        arm_specs=[(name, RESULTS / subdir / 'ZINC') for name, subdir in ARMS],
        output_path=RESULTS / 'report_ZINC.md',
        task='graph_regression',
        # reused from the Substructure -> TU experiment (byte-identical pretrain
        # config); see the comment in parameters_transfer_ZINC_linear_probe.yml.
        pretrain_dir=Path('results/base_paper/transfer/Substructure_to_TU/pretrain/multi'),
        notes=REPORT_NOTES)


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
              type=click.Choice(['overlap', 'pretrain', 'finetune', 'baseline', 'scratch',
                                 'report', 'all']),
              help='overlap = spec 18 B0 go/no-go check only (needs preprocessing() already run on '
                   'both datasets); pretrain = SubstructureBenchmark only; finetune = ZINC only '
                   '(needs an existing SubstructureBenchmark checkpoint); baseline = '
                   'untrained-backbone probe (random init, frozen, head-only training) as the '
                   'direct comparison point for the transfer run; scratch = whole backbone '
                   'trained from a fresh init, no transfer; report = only re-render the markdown '
                   'report from existing results; all = pretrain then finetune. Every '
                   'result-producing stage refreshes results/.../report_ZINC.md.')
@click.option('--num_threads', default=-1, help='Number of threads to use for run_configurations')
def main(stage, num_threads):
    if stage == 'overlap':
        run_overlap_report()
        return

    if stage in ('pretrain', 'all'):
        run_pretrain(num_threads=num_threads)

    if stage in ('finetune', 'all'):
        run_finetune(num_threads=num_threads)

    if stage == 'baseline':
        run_baseline(num_threads=num_threads)

    if stage == 'scratch':
        run_scratch(num_threads=num_threads)

    # every stage that produced results refreshes the markdown report
    if stage in ('finetune', 'baseline', 'scratch', 'all', 'report'):
        write_report()


if __name__ == '__main__':
    main()
