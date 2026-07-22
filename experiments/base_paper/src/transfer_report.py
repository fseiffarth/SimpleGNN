"""Markdown result reports for the transfer-learning experiments (spec 18).

A transfer experiment is only interpretable as a *comparison of arms* --
pretrained probe vs. untrained-backbone probe vs. train-from-scratch -- and for
transfer the runtime matters as much as the metric: a probe that reaches the
same accuracy in a tenth of the epoch time is the actual result. The framework
writes per-epoch CSVs per arm but no cross-arm view, so this module reads the
result directories of several arms and renders one markdown table with both
quality and cost, plus the transfer coverage/freeze numbers from the
``TransferReports/*.json`` sidecars.

Used by the experiment drivers (``--stage report``), e.g.
``experiments/base_paper/transfer/Substructure_to_TU/main_transfer_Substructure_TU.py``.

Run standalone from the repository root:
    python experiments/base_paper/src/transfer_report.py \
        --arm "transfer=results/.../finetune/NCI1" \
        --arm "random-init probe=results/.../baseline_random_probe/NCI1" \
        --arm "scratch=results/.../scratch/NCI1" \
        --pretrain results/.../pretrain/multi \
        --task graph_classification --output results/.../report.md
"""
import json
import math
from pathlib import Path

import click
import pandas as pd

CSV_SEPARATOR = ';'


def _grid_files(results_dir: Path):
    search = Path(results_dir).joinpath('Results')
    if not search.is_dir():
        return []
    return sorted(f for f in search.glob('*_Configuration_*_Results_run_id_*.csv')
                  if 'Best_Configuration' not in f.name)


def _epoch_files(results_dir: Path):
    """Per-epoch CSVs of the ``Best_Configuration`` re-runs of one arm."""
    search = Path(results_dir).joinpath('Results')
    if not search.is_dir():
        return []
    return sorted(search.glob('*_Best_Configuration_*_Results_run_id_*.csv'))


def _mean_std(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return float('nan'), float('nan')
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _select_epoch(df, task):
    """Row of the epoch selected on validation, mirroring the framework's rule."""
    if task == 'graph_regression':
        return df.loc[df['ValidationLoss'].idxmin()]
    best = df[df['ValidationAccuracy'] == df['ValidationAccuracy'].max()]
    return best.loc[best['ValidationLoss'].idxmin()]


def collect_arm(name, results_dir, task):
    """
    Quality + runtime of one arm, aggregated over its run/validation files.

    Reads the ``Best_Configuration`` re-runs; if the arm has not reached
    ``run_best_configuration()`` yet (or was stopped early) it falls back to the
    grid-search runs and is flagged ``partial`` -- those runs carry no test
    metrics, so only the validation column is meaningful for them.
    """
    results_dir = Path(results_dir)
    files = _epoch_files(results_dir)
    partial = False
    if not files:
        files = _grid_files(results_dir)
        partial = bool(files)
    arm = {'name': name, 'results_dir': str(results_dir), 'runs': len(files),
           'task': task, 'missing': not files, 'partial': partial}
    if not files:
        return arm

    test_scores, validation_scores, epoch_times, run_times, epochs, selected_epochs = [], [], [], [], [], []
    for path in files:
        df = pd.read_csv(path, sep=CSV_SEPARATOR)
        if df.empty:
            continue
        row = _select_epoch(df, task)
        if task == 'graph_regression':
            test_scores.append(float(row['TestMAE'] if 'TestMAE' in row and not pd.isna(row['TestMAE'])
                                     else row['TestLoss']))
            validation_scores.append(float(row['ValidationMAE'] if 'ValidationMAE' in row
                                           and not pd.isna(row['ValidationMAE'])
                                           else row['ValidationLoss']))
        else:
            test_scores.append(float(row['TestAccuracy']))
            validation_scores.append(float(row['ValidationAccuracy']))
        epoch_times.extend(float(t) for t in df['EpochTime'])
        run_times.append(float(df['EpochTime'].sum()))
        epochs.append(int(len(df)))
        selected_epochs.append(int(row['Epoch']))

    arm['test_mean'], arm['test_std'] = _mean_std(test_scores)
    arm['validation_mean'], arm['validation_std'] = _mean_std(validation_scores)
    arm['epoch_time_mean'], arm['epoch_time_std'] = _mean_std(epoch_times)
    arm['run_time_mean'], arm['run_time_std'] = _mean_std(run_times)
    arm['total_time'] = sum(run_times)
    arm['epochs'] = str(epochs[0]) if len(set(epochs)) == 1 else f"{min(epochs)}-{max(epochs)}"
    arm['selected_epoch_mean'], _ = _mean_std(selected_epochs)
    arm['grid_time'] = sum(float(pd.read_csv(path, sep=CSV_SEPARATOR)['EpochTime'].sum())
                           for path in _grid_files(results_dir))
    arm.update(_transfer_facts(results_dir))
    return arm


def _transfer_facts(results_dir: Path):
    """Coverage/freeze numbers from the TransferReports sidecar, if the arm had one."""
    reports = sorted(Path(results_dir).joinpath('TransferReports').glob('transfer_report_*.json'))
    if not reports:
        return {'transferred': None}
    with open(reports[0]) as f:
        report = json.load(f)
    total = report.get('total_parameters', 0)
    matched = report.get('matched_parameters', 0)
    return {'transferred': matched,
            'transfer_total': total,
            'transfer_coverage': (matched / total) if total else float('nan'),
            'frozen_tensors': len(report.get('frozen_parameters', [])),
            'source_checkpoint': report.get('source_checkpoint', ''),
            'transfer_warnings': report.get('warnings', [])}


def _format_duration(seconds):
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return '-'
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _format_score(arm, key='test'):
    """Accuracies are already stored in percent by the framework; MAE is absolute."""
    if arm.get('missing'):
        return '-'
    if arm.get('partial') and key == 'test':
        return '*n/a (grid runs only)*'
    return f"{arm[f'{key}_mean']:.4f} ± {arm[f'{key}_std']:.4f}" \
        if arm['task'] == 'graph_regression' \
        else f"{arm[f'{key}_mean']:.2f} ± {arm[f'{key}_std']:.2f}"


def render(title, arms, pretrain=None, notes=(), task='graph_classification'):
    """Render the comparison of ``arms`` (list of collect_arm dicts) as markdown."""
    score_header = 'Test MAE' if task == 'graph_regression' else 'Test accuracy (%)'
    validation_header = 'Val MAE' if task == 'graph_regression' else 'Val accuracy (%)'
    lines = [f"# {title}", '',
             '## Results', '',
             f"| Arm | {score_header} | {validation_header} | Runs | Epochs | "
             f"Epoch time | Time / run | Total train time |",
             '|---|---|---|---|---|---|---|---|']
    for arm in arms:
        if arm.get('missing'):
            lines.append(f"| {arm['name']} | *not run* | - | - | - | - | - | - |")
            continue
        label = arm['name'] + (' *(in progress)*' if arm.get('partial') else '')
        lines.append(
            f"| {label} | {_format_score(arm)} | {_format_score(arm, 'validation')} | "
            f"{arm['runs']} | {arm['epochs']} | "
            f"{arm['epoch_time_mean']:.2f} ± {arm['epoch_time_std']:.2f} s | "
            f"{_format_duration(arm['run_time_mean'])} | {_format_duration(arm['total_time'])} |")

    lines += ['', 'Scores are taken at the epoch selected on validation, averaged over the '
                  'run/validation files of the best configuration (± std over those files). '
                  '"Total train time" sums `EpochTime` over those files; grid-search time is '
                  'listed separately below. Arms marked *(in progress)* have no '
                  '`Best_Configuration` runs yet, so their numbers come from the grid runs, '
                  'which carry no test metrics.', '']

    lines += ['## Cost', '',
              '| Arm | Grid-search time | Best-config time | Transferred params | Frozen tensors |',
              '|---|---|---|---|---|']
    for arm in arms:
        if arm.get('missing'):
            lines.append(f"| {arm['name']} | - | - | - | - |")
            continue
        if arm.get('transferred') is None:
            coverage = '- (no transfer block)'
        else:
            coverage = (f"{arm['transferred']:,}/{arm['transfer_total']:,} "
                        f"({arm['transfer_coverage']:.1%})")
        # for a partial arm total_time *is* the grid time -- don't count it twice
        best_config_time = '-' if arm.get('partial') else _format_duration(arm['total_time'])
        lines.append(f"| {arm['name']} | {_format_duration(arm.get('grid_time'))} | "
                     f"{best_config_time} | {coverage} | {arm.get('frozen_tensors', '-')} |")

    if pretrain and not pretrain.get('missing'):
        lines += ['', '## Pretraining (one-off, amortized over all targets)', '',
                  f"- Source run: `{pretrain['results_dir']}`",
                  f"- Epochs: {pretrain['epochs']}, runs: {pretrain['runs']}",
                  f"- Epoch time: {pretrain['epoch_time_mean']:.2f} ± "
                  f"{pretrain['epoch_time_std']:.2f} s",
                  f"- Best-config time: {_format_duration(pretrain['total_time'])}, "
                  f"grid-search time: {_format_duration(pretrain.get('grid_time'))}"]

    warnings = [(arm['name'], w) for arm in arms for w in arm.get('transfer_warnings', []) or []]
    if warnings:
        lines += ['', '## Transfer warnings', '']
        lines += [f"- **{name}**: {message}" for name, message in warnings]

    if notes:
        lines += ['', '## Notes', ''] + [f"- {note}" for note in notes]

    lines += ['', '## Source directories', '']
    lines += [f"- {arm['name']}: `{arm['results_dir']}`" for arm in arms]
    return '\n'.join(lines) + '\n'


def write_report(title, arm_specs, output_path, task='graph_classification',
                 pretrain_dir=None, notes=()):
    """
    Collect ``arm_specs`` (list of ``(name, results_dir)``) and write the markdown
    report to ``output_path``. Returns the path written.
    """
    arms = [collect_arm(name, results_dir, task) for name, results_dir in arm_specs]
    pretrain = collect_arm('pretrain', pretrain_dir, 'graph_regression') if pretrain_dir else None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(render(title, arms, pretrain=pretrain, notes=notes, task=task))
    print(f"transfer report written to {output_path}")
    return output_path


@click.command()
@click.option('--arm', 'arm_options', multiple=True, required=True,
              help='Arm as "name=results/dir" (repeat once per arm, in table order).')
@click.option('--pretrain', default=None, help='Result dir of the pretraining run (optional).')
@click.option('--task', default='graph_classification',
              type=click.Choice(['graph_classification', 'graph_regression']))
@click.option('--title', default='Transfer experiment report')
@click.option('--output', required=True, help='Path of the markdown file to write.')
def main(arm_options, pretrain, task, title, output):
    arm_specs = []
    for option in arm_options:
        if '=' not in option:
            raise click.BadParameter(f'--arm must look like "name=results/dir", got {option!r}')
        name, _, results_dir = option.partition('=')
        arm_specs.append((name.strip(), results_dir.strip()))
    write_report(title, arm_specs, output, task=task, pretrain_dir=pretrain)


if __name__ == '__main__':
    main()
