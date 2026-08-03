"""Generate the fixed split file for the LRGB Peptides-func / Peptides-struct datasets.

Unlike QM9's random 80/10/10 split (see ``simplegnn.utils.qm_splits``), LRGB
ships an *official* train/val/test split. ``LRGBGraphDataPreprocessing``
concatenates the three official splits in train -> val -> test order, so the
split file only needs the three splits' lengths, expressed as contiguous
offsets into that concatenation -- no shuffling.

The generated file follows the split format used everywhere in the framework
(see ``simplegnn/datasets/splits/fixed/ZINC_splits.json``)::

    [{"test": [...], "model_selection": [{"train": [...], "validation": [...]}]}]

Usage
-----
    python -m simplegnn.utils.lrgb_splits --dataset peptides-func
    python -m simplegnn.utils.lrgb_splits --dataset peptides-struct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# default destination: the package's `fixed` split directory, next to the ZINC/QM9 splits
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'datasets' / 'splits' / 'fixed'

LRGB_NAMES = ('peptides-func', 'peptides-struct')


def _split_lengths(dataset_name: str, root: Path) -> tuple[int, int, int]:
    """Lengths of LRGB's official train/val/test splits (downloads on first call)."""
    try:
        from torch_geometric.datasets import LRGBDataset
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "torch-geometric is required for LRGB split generation. Install project dependencies first."
        ) from exc

    name = dataset_name.lower()
    if name not in LRGB_NAMES:
        raise ValueError(f"Unsupported LRGB dataset '{dataset_name}'. Use one of {LRGB_NAMES}.")

    train = LRGBDataset(root=str(root), name=name, split='train')
    validation = LRGBDataset(root=str(root), name=name, split='val')
    test = LRGBDataset(root=str(root), name=name, split='test')
    return len(train), len(validation), len(test)


def build_lrgb_splits(n_train: int, n_val: int, n_test: int) -> list[dict]:
    """
    Build the single-fold split from LRGB's official train/val/test sizes.

    Parameters
    ----------
    n_train, n_val, n_test : int
        Sizes of the official train/validation/test splits, in the order
        ``LRGBGraphDataPreprocessing`` concatenates them (train, then
        validation, then test).

    Returns
    -------
    list of dict
        One entry (one fold) in the framework's split JSON format.
    """
    train = list(range(0, n_train))
    validation = list(range(n_train, n_train + n_val))
    test = list(range(n_train + n_val, n_train + n_val + n_test))
    return [{'test': test, 'model_selection': [{'train': train, 'validation': validation}]}]


def generate_lrgb_splits(dataset_name: str,
                         output_dir: Path = DEFAULT_OUTPUT_DIR,
                         root: Path = Path('/tmp'),
                         overwrite: bool = False) -> Path:
    """Write ``<output_dir>/<dataset_name>_splits.json`` and return its path."""
    name = dataset_name.lower()
    out_file = output_dir / f'{name}_splits.json'
    if out_file.is_file() and not overwrite:
        print(f'Split file {out_file} already exists. Use --overwrite to regenerate.')
        return out_file

    n_train, n_val, n_test = _split_lengths(name, root)
    splits = build_lrgb_splits(n_train, n_val, n_test)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(splits, f)
    print(f'Created split file {out_file} '
          f'({n_train + n_val + n_test} graphs: {n_train} train / '
          f'{n_val} validation / {n_test} test)')
    return out_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', required=True, choices=list(LRGB_NAMES),
                        help='peptides-func or peptides-struct')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory for the split file (default: {DEFAULT_OUTPUT_DIR})')
    # /tmp is the same root LRGBGraphDataPreprocessing uses, so generating the
    # splits and running the experiment share one download
    parser.add_argument('--root', type=Path, default=Path('/tmp'),
                        help='Download/cache root passed to torch-geometric (default: /tmp)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite an existing split file')
    args = parser.parse_args(argv)

    generate_lrgb_splits(dataset_name=args.dataset, output_dir=args.output_dir,
                         root=args.root, overwrite=args.overwrite)
    return 0


if __name__ == '__main__':
    sys.exit(main())
