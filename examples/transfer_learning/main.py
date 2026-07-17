"""NCI1 -> DHFR hash-keyed transfer-learning example (spec 18 B5).

Runs, in order:

1. pretraining: the standard 5-step workflow on NCI1 with save_best_model +
   save_transfer_keys, producing checkpoints and .keys.pt sidecars,
2. overlap report: preprocesses DHFR and prints the canonical label-vocabulary
   overlap NCI1 -> DHFR per label type (spec 18 B0 go/no-go gate),
3. finetuning: the standard 5-step workflow on DHFR, where the transfer: block
   in finetune_parameters.yml warm-starts every model from the NCI1 checkpoint
   (invariant layers matched slot-by-slot via canonical hashes, standard layers
   by name+shape, classification head re-initialized).

Both datasets are TUDatasets and are downloaded automatically. Run from the
repo root:

    python examples/transfer_learning/main.py            # full run
    python examples/transfer_learning/main.py --fast     # smoke test

--fast caps the epochs and restricts both runs to a single validation fold
(patched copies of the configs are written to results/transfer_learning/fast/,
the original YAML files are never modified).
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from simplegnn.framework.core import FrameworkMain  # noqa: E402

import measure_overlap  # noqa: E402  (B0 driver, examples/transfer_learning/)

FAST_EPOCHS = 1
FAST_DIR = REPO_ROOT / 'results' / 'transfer_learning' / 'fast'


def _fast_configs(main_yml: Path, fast_results: Path) -> Path:
    """Patched copies of a main config + its hyperparameters for --fast:
    epochs capped, a single validation fold, results under fast/."""
    FAST_DIR.mkdir(parents=True, exist_ok=True)
    main_config = yaml.safe_load(main_yml.read_text())
    for dataset in main_config['datasets']:
        paths = dataset['paths']
        # cap the epochs in a patched hyperparameter file
        params = yaml.safe_load((REPO_ROOT / paths['hyperparameters']).read_text())
        params['epochs'] = [FAST_EPOCHS]
        if 'transfer' in params:
            # the fast finetune run transfers from the fast pretraining results
            params['transfer']['source']['results_path'] = str(
                FAST_DIR / 'pretrain')
        params_path = FAST_DIR / f"fast_{Path(paths['hyperparameters']).name}"
        params_path.write_text(yaml.safe_dump(params, sort_keys=False))
        paths['hyperparameters'] = str(params_path)
        # restrict to the first validation fold via a truncated splits file
        splits = json.loads((REPO_ROOT / paths['splits']).read_text())
        splits_path = FAST_DIR / f"fast_{dataset['name']}_splits.json"
        splits_path.write_text(json.dumps(splits[:1]))
        paths['splits'] = str(splits_path)
        paths['results'] = str(fast_results)
    patched_main = FAST_DIR / f'fast_{main_yml.name}'
    patched_main.write_text(yaml.safe_dump(main_config, sort_keys=False))
    return patched_main


def run_workflow(main_yml: Path, num_threads: int = -1,
                 preprocessed: FrameworkMain = None) -> FrameworkMain:
    """The standard 5-step workflow (see examples/share_gnn_basic/main.py)."""
    experiment = preprocessed
    if experiment is None:
        experiment = FrameworkMain(main_yml)
        experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast', action='store_true',
                        help='smoke test: 1 epoch, a single validation fold')
    parser.add_argument('--num-threads', type=int, default=-1,
                        help='parallel workers for the grid search (-1 = all CPUs)')
    args = parser.parse_args()

    pretrain_main = EXAMPLE_DIR / 'pretrain_main.yml'
    finetune_main = EXAMPLE_DIR / 'finetune_main.yml'
    if args.fast:
        pretrain_main = _fast_configs(pretrain_main, FAST_DIR / 'pretrain')
        finetune_main = _fast_configs(finetune_main, FAST_DIR / 'finetune')

    # 1. pretrain on NCI1 (writes checkpoints + .keys.pt sidecars)
    print('=' * 70)
    print('Step 1/3: pretraining on NCI1')
    print('=' * 70)
    run_workflow(pretrain_main, num_threads=args.num_threads)

    # 2. preprocess DHFR, then print the NCI1 -> DHFR overlap report (B0)
    print('=' * 70)
    print('Step 2/3: NCI1 -> DHFR label-vocabulary overlap (spec 18 B0)')
    print('=' * 70)
    finetune_experiment = FrameworkMain(finetune_main)
    finetune_experiment.preprocessing(num_threads=1)
    labels_root = REPO_ROOT / 'data' / 'TUDatasets' / 'labels'
    measure_overlap.main(['--source-labels', str(labels_root / 'NCI1'),
                          '--target-labels', str(labels_root / 'DHFR')])

    # 3. finetune on DHFR; the transfer: block in finetune_parameters.yml
    #    warm-starts every model from the NCI1 checkpoint
    print('=' * 70)
    print('Step 3/3: finetuning on DHFR (hash-keyed transfer from NCI1)')
    print('=' * 70)
    run_workflow(finetune_main, num_threads=args.num_threads,
                 preprocessed=finetune_experiment)
    results_root = FAST_DIR / 'finetune' if args.fast \
        else REPO_ROOT / 'results' / 'transfer_learning' / 'finetune'
    print(f"Done. Per-run transfer reports: {results_root / 'DHFR' / 'TransferReports'}")


if __name__ == '__main__':
    main()
