"""Generate split files for the OGB graph-property-prediction datasets (``ogbg-mol*``).

OGB ships one official scaffold split per dataset; reproducing a leaderboard
number means using exactly that split, so this module reads it from
``PygGraphPropPredDataset.get_idx_split()`` rather than resampling anything.

A split file has to exist *before* ``FrameworkMain`` is constructed — the
main-config check (``framework/utils/configuration_checks.py:87``) raises
``FileNotFoundError`` when the split JSON is missing, which happens well before
the ``split_function`` hook in ``framework/utils/preprocessing.py`` would ever
run. Generating the file up front (as the QM9 experiment does, see
``simplegnn/utils/qm_splits.py``) is therefore the only workable order.

The generated file follows the split format used everywhere in the framework
(see ``simplegnn/datasets/splits/fixed/ZINC_splits.json``)::

    [{"test": [...], "model_selection": [{"train": [...], "validation": [...]}]}]

Usage
-----
    python -m simplegnn.utils.ogb_splits --dataset ogbg-molhiv
    python -m simplegnn.utils.ogb_splits --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# default destination: the package's `fixed` split directory, next to the ZINC splits
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'datasets' / 'splits' / 'fixed'

# The single-task `ogbg-mol*` datasets. Multi-task members of the family
# (moltox21, molsider, molclintox, moltoxcast, molmuv, molpcba, molchembl) are
# deliberately absent: their targets are NaN-masked matrices, which the
# framework's single-target classification/regression heads cannot represent.
SINGLE_TASK_CLASSIFICATION = ('ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp')
SINGLE_TASK_REGRESSION = ('ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo')
SUPPORTED_DATASETS = SINGLE_TASK_CLASSIFICATION + SINGLE_TASK_REGRESSION


def build_ogb_splits(db_name: str, root: Path = Path('/tmp')) -> list[dict]:
    """
    Read OGB's official scaffold split and convert it to the framework format.

    Parameters
    ----------
    db_name : str
        OGB dataset name, e.g. ``'ogbg-molhiv'``.
    root : Path
        Download/cache root handed to ``PygGraphPropPredDataset``. Use the same
        root the experiment's preprocessing uses so both share one download.

    Returns
    -------
    list of dict
        One entry (OGB provides a single fold) in the framework's split format.
    """
    try:
        # shared loader: applies the PyTorch 2.6+ `weights_only` workaround and
        # reuses the same cache the experiment's preprocessing will read
        from simplegnn.datasets.graph_dataset_preprocessing import load_ogb_graphprop_dataset
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "The `ogb` package is required for OGB split generation. Install it with `pip install ogb`."
        ) from exc

    dataset = load_ogb_graphprop_dataset(db_name, root=root)
    split_idx = dataset.get_idx_split()
    train = split_idx['train'].tolist()
    validation = split_idx['valid'].tolist()
    test = split_idx['test'].tolist()
    return [{'test': sorted(test),
             'model_selection': [{'train': sorted(train), 'validation': sorted(validation)}]}]


def generate_ogb_splits(db_name: str = 'ogbg-molhiv',
                        output_dir: Path = DEFAULT_OUTPUT_DIR,
                        root: Path = Path('/tmp'),
                        overwrite: bool = False) -> Path:
    """Write ``<output_dir>/<db_name>_splits.json`` and return its path."""
    out_file = output_dir / f'{db_name}_splits.json'
    if out_file.is_file() and not overwrite:
        print(f'Split file {out_file} already exists. Use --overwrite to regenerate.')
        return out_file

    splits = build_ogb_splits(db_name, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(splits, f)
    fold = splits[0]['model_selection'][0]
    num_graphs = len(fold['train']) + len(fold['validation']) + len(splits[0]['test'])
    print(f'Created split file {out_file} '
          f'({num_graphs} graphs: {len(fold["train"])} train / '
          f'{len(fold["validation"])} validation / {len(splits[0]["test"])} test)')
    return out_file


def ogb_splits(output_path, db_name, *args, **kwargs) -> Path:
    """
    ``split_function`` entry point kept for main configs that use the hook.

    Note that the config check demands the split file already exist, so in
    practice the file is produced by :func:`generate_ogb_splits` (via the CLI or
    ``experiments/ogb/run_ogb.sh``) before the framework starts.

    Parameters
    ----------
    output_path : Path or str
        Either the directory the split file should be written to, or the full
        path of the ``.json`` file itself.
    db_name : str
        OGB dataset name.
    """
    output_path = Path(output_path)
    output_dir = output_path.parent if output_path.suffix == '.json' else output_path
    return generate_ogb_splits(db_name=db_name, output_dir=output_dir,
                               root=Path(kwargs.get('root', '/tmp')), overwrite=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='ogbg-molhiv',
                        help=f'OGB dataset name (default: ogbg-molhiv). Supported: {", ".join(SUPPORTED_DATASETS)}')
    parser.add_argument('--all', action='store_true',
                        help='Generate split files for every supported single-task ogbg-mol dataset')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory for the split file (default: {DEFAULT_OUTPUT_DIR})')
    # /tmp is the same root OGBGraphPropertyGraphDataPreprocessing uses, so
    # generating the splits and running the experiment share one download
    parser.add_argument('--root', type=Path, default=Path('/tmp'),
                        help='Download/cache root passed to ogb (default: /tmp)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite an existing split file')
    args = parser.parse_args(argv)

    datasets = SUPPORTED_DATASETS if args.all else (args.dataset,)
    for db_name in datasets:
        generate_ogb_splits(db_name=db_name, output_dir=args.output_dir,
                            root=args.root, overwrite=args.overwrite)
    return 0


if __name__ == '__main__':
    sys.exit(main())
