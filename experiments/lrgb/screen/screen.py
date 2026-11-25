## Fast LRGB screening: grid search over candidate architectures with a few
## seeds each, then a per-candidate MAE table. No best-configuration rerun;
## the winners go into a longer confirmation run (see README.md in this folder).
## Same layout as experiments/ogb/molhiv_screen/screen.py, for MAE (lower is
## better) instead of ROC-AUC.
import re
import sys
from pathlib import Path

import click
import pandas as pd
import yaml

from simplegnn.framework.core import FrameworkMain

CANDIDATE_RE = re.compile(r'^#\s+(\d+)\s+(\S+)')


def candidate_names(models_path):
    """Candidate names from the ``#   <id> <name>`` lines of the model file header."""
    names = {}
    for line in Path(models_path).read_text().splitlines():
        if not line.startswith('#'):
            break
        match = CANDIDATE_RE.match(line)
        if match:
            names[int(match.group(1))] = match.group(2)
    return names


def analyze(main_config):
    """
    One row per candidate: test MAE at the validation-selected epoch (mean and
    std over seeds), the oracle best test MAE over all epochs, best epochs,
    train MAE at the selected and at the final epoch, and seconds per epoch.
    """
    config = yaml.safe_load(Path(main_config).read_text())['datasets'][0]
    db_name = config['name']
    results_dir = Path(config['paths']['results']) / db_name
    names = candidate_names(config['paths']['models'])
    rows = []
    for config_id in sorted({int(m.group(1)) for m in (re.search(r'_Configuration_(\d+)_Results', p.name)
                                                       for p in (results_dir / 'Results').glob('*.csv')) if m}):
        runs = []
        for path in sorted((results_dir / 'Results').glob(f'{db_name}_Configuration_{config_id:06d}_Results_run_id_*.csv')):
            epochs = pd.read_csv(path, sep=';')
            if epochs.empty:
                continue
            best = epochs.ValidationMAE.idxmin()
            runs.append({'val': epochs.ValidationMAE[best], 'test': epochs.TestMAE[best],
                         'train': epochs.EpochMAE[best], 'epoch': int(epochs.Epoch[best]),
                         'last_epoch': int(epochs.Epoch.iloc[-1]), 'oracle': epochs.TestMAE.min(),
                         'final_train': epochs.EpochMAE.iloc[-1], 'epoch_time': epochs.EpochTime.median()})
        if not runs:
            continue
        runs = pd.DataFrame(runs)
        rows.append({'Id': config_id, 'Candidate': names.get(config_id, f'config_{config_id}'), 'Seeds': len(runs),
                     'Val': runs.val.mean(), 'Test': runs.test.mean(), 'Test Std': runs.test.std(ddof=0),
                     'Oracle Test': runs.oracle.mean(), 'Train@Best': runs.train.mean(),
                     'Final Train': runs.final_train.mean(), 'Best Epochs': runs.epoch.tolist(),
                     'Last Epochs': runs.last_epoch.tolist(), 'Sec/Epoch': runs.epoch_time.mean()})
    table = pd.DataFrame(rows)
    table.to_csv(results_dir.parent / 'screen_summary.csv', index=False)
    return table


@click.command()
@click.option('--config', 'main_configs', required=True, multiple=True, type=click.Path(exists=True),
              help='Screening main config, e.g. experiments/lrgb/screen/main_round1.yml; '
                   'repeat the option to run several rounds one after another')
@click.option('--num_threads', default=3, help='Parallel processes for the grid search')
@click.option('--analyze_only', is_flag=True, help='Only summarize existing results')
def main(main_configs, num_threads, analyze_only):
    for main_config in main_configs:
        print(f"=== screening {main_config} ===")
        if not analyze_only:
            experiment = FrameworkMain(Path(main_config))
            experiment.preprocessing(num_threads=1)
            experiment.run_configurations(num_threads=num_threads)
        # printed after every round, so a finished round is visible while the next runs
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            print(analyze(main_config).round(4).to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
