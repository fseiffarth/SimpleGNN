"""Generate the fixed 80/10/10 split file for the QM7/QM9 quantum-chemistry datasets.

The split reproduces the one used by the original RuleGNN experiments
(``src/Preprocessing/split_functions.py::qm_splits``): the graph indices are
shuffled once with Python's :mod:`random` seeded at 42 and cut at 80% / 90%.
Using the stdlib shuffle (rather than numpy) is what keeps the split identical
to the published runs, so do not "modernize" it to ``np.random``.

The generated file follows the split format used everywhere in the framework
(see ``simplegnn/datasets/splits/fixed/ZINC_splits.json``)::

    [{"test": [...], "model_selection": [{"train": [...], "validation": [...]}]}]

Usage
-----
    python -m simplegnn.utils.qm_splits --dataset QM9
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# default destination: the package's `fixed` split directory, next to the ZINC splits
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'datasets' / 'splits' / 'fixed'

QM9_NAMES = ('QM9', 'qm9', 'QM', 'qm')
QM7_NAMES = ('QM7', 'qm7', 'QM7b', 'qm7b')


def _num_graphs(dataset_name: str, root: Path) -> int:
    """Return the number of graphs in the (downloaded) QM dataset."""
    try:
        import torch_geometric.datasets as tg_datasets
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "torch-geometric is required for QM split generation. Install project dependencies first."
        ) from exc

    if dataset_name in QM9_NAMES:
        # shared loader: applies the rdkit/PyG SDMolSupplier workaround and reuses
        # the same cache the experiment's preprocessing will read
        from simplegnn.datasets.graph_dataset_preprocessing import load_qm9_dataset
        dataset = load_qm9_dataset(root)
    elif dataset_name in QM7_NAMES:
        dataset = tg_datasets.QM7b(root=str(root))
    else:
        raise ValueError(f"Unsupported QM dataset '{dataset_name}'. Use one of {QM9_NAMES + QM7_NAMES}.")
    return len(dataset)


def build_qm_splits(num_graphs: int, seed: int = 42) -> list[dict]:
    """
    Build the single-fold 80/10/10 split for a QM dataset.

    Parameters
    ----------
    num_graphs : int
        Number of graphs in the dataset.
    seed : int
        Seed for the stdlib shuffle. 42 reproduces the original RuleGNN split.

    Returns
    -------
    list of dict
        One entry (one fold) in the framework's split JSON format.
    """
    indices = list(range(num_graphs))
    random.seed(seed)
    random.shuffle(indices)
    train = indices[:int(0.8 * num_graphs)]
    validation = indices[int(0.8 * num_graphs):int(0.9 * num_graphs)]
    test = indices[int(0.9 * num_graphs):]
    return [{'test': sorted(test),
             'model_selection': [{'train': sorted(train), 'validation': sorted(validation)}]}]


def generate_qm_splits(dataset_name: str = 'QM9',
                       output_dir: Path = DEFAULT_OUTPUT_DIR,
                       root: Path = Path('/tmp'),
                       seed: int = 42,
                       overwrite: bool = False) -> Path:
    """Write ``<output_dir>/<dataset_name>_splits.json`` and return its path."""
    out_file = output_dir / f'{dataset_name}_splits.json'
    if out_file.is_file() and not overwrite:
        print(f'Split file {out_file} already exists. Use --overwrite to regenerate.')
        return out_file

    num_graphs = _num_graphs(dataset_name, root)
    splits = build_qm_splits(num_graphs, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(splits, f)
    fold = splits[0]['model_selection'][0]
    print(f'Created split file {out_file} '
          f'({num_graphs} graphs: {len(fold["train"])} train / '
          f'{len(fold["validation"])} validation / {len(splits[0]["test"])} test)')
    return out_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='QM9', help='QM9 or QM7 (default: QM9)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory for the split file (default: {DEFAULT_OUTPUT_DIR})')
    # /tmp is the same root QMGraphDataPreprocessing uses, so generating the
    # splits and running the experiment share one download
    parser.add_argument('--root', type=Path, default=Path('/tmp'),
                        help='Download/cache root passed to torch-geometric (default: /tmp)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Shuffle seed; 42 reproduces the original RuleGNN split (default: 42)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite an existing split file')
    args = parser.parse_args(argv)

    generate_qm_splits(dataset_name=args.dataset, output_dir=args.output_dir,
                       root=args.root, seed=args.seed, overwrite=args.overwrite)
    return 0


if __name__ == '__main__':
    sys.exit(main())
