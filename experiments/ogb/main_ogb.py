## OGB graph-property-prediction experiments (ShareGNN multi-head).
##
## Covers the single-task `ogbg-mol*` datasets. The multi-task members of the
## family (moltox21, molsider, molclintox, moltoxcast, molmuv, molpcba) are not
## included: their targets are NaN-masked matrices, and the framework's
## classification/regression heads take a single target per graph.
from pathlib import Path

import click
import pandas as pd

from simplegnn.framework.core import FrameworkMain
from simplegnn.framework.utils.seed_ensemble import write_seed_ensemble

CONFIG_DIR = Path('experiments/ogb/configs')
LADDER_DIR = Path('experiments/ogb/molhiv_ladder')
LADDER2_DIR = Path('experiments/ogb/molhiv_ladder2')
LADDER3_DIR = Path('experiments/ogb/molhiv_ladder3')
LADDER4_DIR = Path('experiments/ogb/molhiv_ladder4')

# short name -> main config, mirroring simplegnn.utils.ogb_splits.SUPPORTED_DATASETS
DATASETS = {
    'molhiv': CONFIG_DIR / 'main_molhiv.yml',
    'molbace': CONFIG_DIR / 'main_molbace.yml',
    'molbbbp': CONFIG_DIR / 'main_molbbbp.yml',
    'molesol': CONFIG_DIR / 'main_molesol.yml',
    'molfreesolv': CONFIG_DIR / 'main_molfreesolv.yml',
    'mollipo': CONFIG_DIR / 'main_mollipo.yml',
    # molhiv overfitting sweep: identical apart from rule_occurrence_threshold,
    # which is read as a scalar in the layer and so cannot be a grid axis.
    # Each arm writes to its own results root; see the config headers.
    'molhiv-t1': CONFIG_DIR / 'main_molhiv_t1.yml',
    'molhiv-t50': CONFIG_DIR / 'main_molhiv_t50.yml',
    # optimal arms: tuned models/parameters pairs (see OPTIMAL_CONFIG.md);
    # same datasets/splits, results land in results/ogb_optimal/. The
    # <dataset>-<variant> naming keeps run_ogb.sh's split generation working.
    'molhiv-optimal': CONFIG_DIR / 'main_molhiv_optimal.yml',
    'molbace-optimal': CONFIG_DIR / 'main_molbace_optimal.yml',
    'molbbbp-optimal': CONFIG_DIR / 'main_molbbbp_optimal.yml',
    'molesol-optimal': CONFIG_DIR / 'main_molesol_optimal.yml',
    'molfreesolv-optimal': CONFIG_DIR / 'main_molfreesolv_optimal.yml',
    'mollipo-optimal': CONFIG_DIR / 'main_mollipo_optimal.yml',
}

# sequential experiment ladders: (stage name, main config) in run order.
# molhiv-ladder stages are cumulative (molhiv_ladder/README.md); molhiv-ladder2
# stages are single-change arms off its t0 reference (molhiv_ladder2/README.md)
LADDERS = {
    'molhiv-ladder': [
        ('s0_protocol', LADDER_DIR / 'main_s0_protocol.yml'),
        ('s1_threshold', LADDER_DIR / 'main_s1_threshold.yml'),
        ('s2_trimmed', LADDER_DIR / 'main_s2_trimmed.yml'),
        ('s3_l1', LADDER_DIR / 'main_s3_l1.yml'),
        ('s4_auc_loss', LADDER_DIR / 'main_s4_auc_loss.yml'),
    ],
    'molhiv-ladder2': [
        ('t0_reference', LADDER2_DIR / 'main_t0_reference.yml'),
        ('t1_weight_decay', LADDER2_DIR / 'main_t1_weight_decay.yml'),
        ('t2_l1', LADDER2_DIR / 'main_t2_l1.yml'),
        ('t3_node_dropout', LADDER2_DIR / 'main_t3_node_dropout.yml'),
        ('t4_batch256', LADDER2_DIR / 'main_t4_batch256.yml'),
        ('t5_pool_only', LADDER2_DIR / 'main_t5_pool_only.yml'),
    ],
    # local-convolution confirmation; ladder-2 t0 is the reference (molhiv_ladder3/README.md)
    'molhiv-ladder3': [
        ('v1_local2', LADDER3_DIR / 'main_v1_local2.yml'),
        ('v2_local1', LADDER3_DIR / 'main_v2_local1.yml'),
        ('v3_local1_two_conv', LADDER3_DIR / 'main_v3_local1_two_conv.yml'),
        ('v4_local1_three_conv', LADDER3_DIR / 'main_v4_local1_three_conv.yml'),
    ],
    # residual convolutions and a longer budget; ladder-3 v3 is the reference (molhiv_ladder4/README.md)
    'molhiv-ladder4': [
        ('w1_two_conv_residual', LADDER4_DIR / 'main_w1_two_conv_residual.yml'),
        ('w2_two_conv_30_epochs', LADDER4_DIR / 'main_w2_two_conv_30_epochs.yml'),
        ('w3_three_conv_residual', LADDER4_DIR / 'main_w3_three_conv_residual.yml'),
    ],
}
LADDER_SUMMARY = {'molhiv-ladder': Path('results/ogb_molhiv_ladder/ladder_summary.csv'),
                  'molhiv-ladder2': Path('results/ogb_molhiv_ladder2/ladder_summary.csv'),
                  'molhiv-ladder3': Path('results/ogb_molhiv_ladder3/ladder_summary.csv'),
                  'molhiv-ladder4': Path('results/ogb_molhiv_ladder4/ladder_summary.csv')}


def run_pipeline(config_path, num_threads=-1):
    """
    Run the full grid-search -> select -> rerun-best -> test pipeline.

    The seed reruns of the best configuration run one at a time: in parallel,
    each process materializes its own ShareGNN rows on the GPU, and four
    molhiv workers at batch size 256 needed more than the 95 GiB of the card
    (out of memory in ladder 2, stage t4_batch256).
    """
    experiment = FrameworkMain(config_path)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=1)
    experiment.evaluate_results(evaluate_best_model=True)
    return experiment


def stage_summary(stage, experiment):
    """One ladder-summary row: seed mean/std AUCs and the seed-ensemble AUCs."""
    row = {'Stage': stage}
    for db_name, configurations in experiment.network_configurations.items():
        results_dir = Path(configurations[0]['paths']['results'])
        summary = pd.read_csv(results_dir / db_name / 'summary_best_mean.csv').iloc[0]
        for split in ('Validation', 'Test'):
            row[f'{split} AUC Mean'] = summary[f'{split} AUC Mean']
            row[f'{split} AUC Std'] = summary[f'{split} AUC Std']
        ensemble = write_seed_ensemble(results_dir, db_name)
        row['Runs'] = ensemble['num_runs']
        row['Ensemble Validation AUC'] = ensemble['validation_auc_ensemble_mean']
        row['Ensemble Test AUC'] = ensemble['test_auc_ensemble_mean']
        row['Rank Ensemble Test AUC'] = ensemble['test_auc_ensemble_rank']
    return row


def run_ladder(ladder, num_threads=-1, from_stage=0):
    """
    Run the stages of a ladder one after another, starting at ``from_stage``.

    The summary CSV is rewritten after every stage, so a partial ladder still
    has one. Rows of stages before ``from_stage`` are kept from the existing
    summary; a rerun stage replaces its row.
    """
    stages = LADDERS[ladder]
    summary_path = LADDER_SUMMARY[ladder]
    rows = {}
    if from_stage > 0 and summary_path.exists():
        previous = pd.read_csv(summary_path)
        rows = {row['Stage']: row for row in previous.to_dict('records')}
    for index, (stage, config_path) in enumerate(stages):
        if index < from_stage:
            continue
        print(f"=== {ladder}: stage {index} ({stage}), {config_path} ===")
        experiment = run_pipeline(config_path, num_threads=num_threads)
        rows[stage] = stage_summary(stage, experiment)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = [rows[name] for name, _ in stages if name in rows]
        pd.DataFrame(ordered).to_csv(summary_path, index=False)
        print(pd.DataFrame(ordered).to_string(index=False))


def main_ogb(dataset='molhiv', num_threads=-1, from_stage=0):
    """Run one dataset config, or every stage of a ladder in sequence."""
    if dataset in LADDERS:
        run_ladder(dataset, num_threads=num_threads, from_stage=from_stage)
        return
    if from_stage:
        raise click.UsageError('--from_stage only applies to ladders: ' + ', '.join(sorted(LADDERS)))
    run_pipeline(DATASETS[dataset], num_threads=num_threads)


@click.command()
@click.option('--dataset', default='molhiv', type=click.Choice(sorted(DATASETS) + sorted(LADDERS)),
              help='Which ogbg-mol* dataset or ladder to run (default: molhiv)')
@click.option('--num_threads', default=-1, help='Number of threads to use')
@click.option('--from_stage', default=0, type=click.IntRange(min=0),
              help='Ladders only: 0-based index of the first stage to run')
def main(dataset, num_threads, from_stage):
    main_ogb(dataset=dataset, num_threads=num_threads, from_stage=from_stage)


if __name__ == '__main__':
    main()
