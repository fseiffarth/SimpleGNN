"""Generate the (unused but required) split file for a BREC dataset.

BREC is evaluated with RPC (Reliable Paired Comparisons), not with a
train/validation/test split: ``experiments/brec/main_brec.py`` trains one
siamese model per graph pair and decides with a Hotelling T-squared test (see
specs/21-brec-expressiveness-benchmark.md). The framework's config pipeline
nevertheless requires a split file for every dataset, so this generator emits a
*valid* single-fold partition that the RPC runner ignores: the first pair's
graphs go to validation, the second pair's to test, and everything else to
train.

The generated file follows the split format used everywhere in the framework
(see ``simplegnn/datasets/splits/fixed/ZINC_splits.json``)::

    [{"test": [...], "model_selection": [{"train": [...], "validation": [...]}]}]

Usage
-----
    python -m simplegnn.utils.brec_splits --dataset BREC
    python -m simplegnn.utils.brec_splits --dataset BREC-r4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simplegnn.datasets.graph_dataset_preprocessing import BREC_NUM_IDS, parse_brec_name

# default destination: the package's `fixed` split directory, next to the ZINC/QM9 splits
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'datasets' / 'splits' / 'fixed'


def build_brec_splits(num_relabel: int) -> list[dict]:
    """
    Build the single-fold placeholder split for a BREC dataset.

    Parameters
    ----------
    num_relabel : int
        Relabelings per graph in the dataset (32 for the official ``BREC``,
        *k* for ``BREC-r<k>``).

    Returns
    -------
    list of dict
        One entry (one fold) in the framework's split JSON format. The three
        index sets are disjoint and cover every graph in the dataset.
    """
    block = 2 * num_relabel  # graphs per pair id
    num_graphs = BREC_NUM_IDS * block
    validation = list(range(0, block))
    test = list(range(block, 2 * block))
    train = list(range(2 * block, num_graphs))
    return [{'test': test, 'model_selection': [{'train': train, 'validation': validation}]}]


def generate_brec_splits(dataset_name: str,
                         output_dir: Path = DEFAULT_OUTPUT_DIR,
                         overwrite: bool = False) -> Path:
    """Write ``<output_dir>/<dataset_name>_splits.json`` and return its path."""
    num_relabel = parse_brec_name(dataset_name)
    out_file = output_dir / f'{dataset_name}_splits.json'
    if out_file.is_file() and not overwrite:
        print(f'Split file {out_file} already exists. Use --overwrite to regenerate.')
        return out_file

    splits = build_brec_splits(num_relabel)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(splits, f)
    fold = splits[0]
    total = len(fold['test']) + len(fold['model_selection'][0]['train']) + len(fold['model_selection'][0]['validation'])
    print(f'Created split file {out_file} ({total} graphs, {num_relabel} relabelings per graph; '
          f'placeholder split — the RPC runner does not use it)')
    return out_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='BREC',
                        help="Dataset name: 'BREC' (32 relabelings) or 'BREC-r<k>' (default: BREC)")
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory for the split file (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite an existing split file')
    args = parser.parse_args(argv)

    generate_brec_splits(dataset_name=args.dataset, output_dir=args.output_dir, overwrite=args.overwrite)
    return 0


if __name__ == '__main__':
    sys.exit(main())
