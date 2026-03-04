"""Generate deterministic 10-fold split files for TU datasets.

The generated files follow the split format used by
`simplegnn/datasets/splits/standard/MUTAG_splits.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# TU dataset names taken from the official TU collection page:
# https://chrsmrrs.github.io/datasets/docs/datasets/
ALL_TU_DATASETS: tuple[str, ...] = (
    "AIDS",
    "alchemy_full",
    "aspirin",
    "benzene",
    "BZR",
    "BZR_MD",
    "COX2",
    "COX2_MD",
    "DHFR",
    "DHFR_MD",
    "ER_MD",
    "ethanol",
    "FRANKENSTEIN",
    "malonaldehyde",
    "MCF-7",
    "MCF-7H",
    "MOLT-4",
    "MOLT-4H",
    "Mutagenicity",
    "MUTAG",
    "naphthalene",
    "NCI1",
    "NCI109",
    "NCI-H23",
    "NCI-H23H",
    "OVCAR-8",
    "OVCAR-8H",
    "P388",
    "P388H",
    "PC-3",
    "PC-3H",
    "PTC_FM",
    "PTC_FR",
    "PTC_MM",
    "PTC_MR",
    "QM9",
    "salicylic_acid",
    "SF-295",
    "SF-295H",
    "SN12C",
    "SN12CH",
    "SW-620",
    "SW-620H",
    "toluene",
    "Tox21_AhR_training",
    "Tox21_AhR_testing",
    "Tox21_AhR_evaluation",
    "Tox21_AR_training",
    "Tox21_AR_testing",
    "Tox21_AR_evaluation",
    "Tox21_AR-LBD_training",
    "Tox21_AR-LBD_testing",
    "Tox21_AR-LBD_evaluation",
    "Tox21_ARE_training",
    "Tox21_ARE_testing",
    "Tox21_ARE_evaluation",
    "Tox21_aromatase_training",
    "Tox21_aromatase_testing",
    "Tox21_aromatase_evaluation",
    "Tox21_ATAD5_training",
    "Tox21_ATAD5_testing",
    "Tox21_ATAD5_evaluation",
    "Tox21_ER_training",
    "Tox21_ER_testing",
    "Tox21_ER_evaluation",
    "Tox21_ER-LBD_training",
    "Tox21_ER-LBD_testing",
    "Tox21_ER-LBD_evaluation",
    "Tox21_HSE_training",
    "Tox21_HSE_testing",
    "Tox21_HSE_evaluation",
    "Tox21_MMP_training",
    "Tox21_MMP_testing",
    "Tox21_MMP_evaluation",
    "Tox21_p53_training",
    "Tox21_p53_testing",
    "Tox21_p53_evaluation",
    "Tox21_PPAR-gamma_training",
    "Tox21_PPAR-gamma_testing",
    "Tox21_PPAR-gamma_evaluation",
    "UACC257",
    "UACC257H",
    "uracil",
    "Yeast",
    "YeastH",
    "ZINC_full",
    "ZINC_test",
    "ZINC_train",
    "ZINC_val",
    "DD",
    "ENZYMES",
    "KKI",
    "OHSU",
    "Peking_1",
    "PROTEINS",
    "PROTEINS_full",
    "COIL-DEL",
    "COIL-RAG",
    "Cuneiform",
    "Fingerprint",
    "FIRSTMM_DB",
    "Letter-high",
    "Letter-low",
    "Letter-med",
    "MSRC_9",
    "MSRC_21",
    "MSRC_21C",
    "COLLAB",
    "dblp_ct1",
    "dblp_ct2",
    "DBLP_v1",
    "deezer_ego_nets",
    "facebook_ct1",
    "facebook_ct2",
    "github_stargazers",
    "highschool_ct1",
    "highschool_ct2",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "infectious_ct1",
    "infectious_ct2",
    "mit_ct1",
    "mit_ct2",
    "REDDIT-BINARY",
    "REDDIT-MULTI-5K",
    "REDDIT-MULTI-12K",
    "reddit_threads",
    "tumblr_ct1",
    "tumblr_ct2",
    "twitch_egos",
    "TWITTER-Real-Graph-Partial",
    "COLORS-3",
    "SYNTHETIC",
    "SYNTHETICnew",
    "Synthie",
    "TRIANGLES",
)


def _import_splitters() -> tuple[Any, Any]:
    """Lazily import sklearn splitters with a clear error message."""
    try:
        from sklearn.model_selection import KFold, StratifiedKFold
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for TU split generation. Install it with 'pip install scikit-learn'."
        ) from exc
    return KFold, StratifiedKFold


def _load_tu_labels(dataset_name: str, root: Path) -> np.ndarray:
    """Load graph labels for a TU dataset using torch-geometric's TUDataset."""
    try:
        from torch_geometric.datasets import TUDataset
    except Exception as exc:
        raise RuntimeError(
            "torch-geometric is required for TU split generation. Install project dependencies first."
        ) from exc

    dataset = TUDataset(root=str(root), name=dataset_name, use_node_attr=True, use_edge_attr=True)
    labels = []
    for data in dataset:
        if data.y is None:
            raise ValueError(f"Dataset '{dataset_name}' does not provide graph labels (data.y is None).")
        value = data.y.detach().cpu().view(-1).numpy()
        if value.size == 0:
            raise ValueError(f"Dataset '{dataset_name}' has an empty label tensor for at least one graph.")
        labels.append(float(value[0]))

    if not labels:
        raise ValueError(f"Dataset '{dataset_name}' is empty.")

    return np.asarray(labels, dtype=float)


def _is_stratifiable(labels: np.ndarray, folds: int) -> bool:
    """Return True if labels support stable stratified k-fold splitting."""
    # Require integer-like labels and at least `folds` samples per class.
    rounded = np.rint(labels)
    if not np.allclose(labels, rounded, equal_nan=False):
        return False

    classes, counts = np.unique(rounded.astype(int), return_counts=True)
    return len(classes) >= 2 and np.min(counts) >= folds


def _build_splits(labels: np.ndarray, folds: int, seed: int) -> list[dict[str, Any]]:
    """Create split objects in the framework's expected JSON format."""
    KFold, StratifiedKFold = _import_splitters()

    indices = np.arange(labels.shape[0])
    if _is_stratifiable(labels, folds):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(indices, np.rint(labels).astype(int))
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(indices)

    result: list[dict[str, Any]] = []
    for train_idx, val_idx in split_iter:
        train = np.sort(train_idx).astype(int).tolist()
        validation = np.sort(val_idx).astype(int).tolist()
        result.append({"test": [], "model_selection": [{"train": train, "validation": validation}]})
    return result


def generate_tu_splits(
    output_dir: Path,
    root: Path,
    folds: int = 10,
    seed: int | None = None,
    overwrite: bool = False,
    standard_dir: Path | None = None,
    datasets: list[str] | None = None,
) -> tuple[int, int]:
    """Generate deterministic split files for TU datasets."""
    if folds <= 1:
        raise ValueError(f"folds must be > 1, got {folds}")

    requested_datasets = datasets if datasets is not None else list(ALL_TU_DATASETS)
    if not requested_datasets:
        raise ValueError("No datasets provided.")

    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_seed = 0 if seed is None else seed
    print(f"Using seed={resolved_seed} for generated (non-standard) datasets")

    labels_by_dataset: dict[str, np.ndarray] = {}
    loading_failures: dict[str, str] = {}
    copy_failures: dict[str, str] = {}
    generated_count = 0
    copied_count = 0
    processing_failures: dict[str, str] = {}

    for dataset_name in requested_datasets:
        target_path = output_dir / f"{dataset_name}_splits.json"
        source_path = None if standard_dir is None else standard_dir / f"{dataset_name}_splits.json"

        if target_path.exists() and not overwrite:
            processing_failures[dataset_name] = (
                f"Refusing to overwrite existing split file: {target_path}. Use --overwrite to replace it."
            )
            continue

        # Prefer copying standard splits when available.
        if source_path is not None and source_path.exists():
            try:
                target_path.write_bytes(source_path.read_bytes())
                copied_count += 1
            except Exception as exc:  # noqa: BLE001 - keep processing remaining datasets
                copy_failures[dataset_name] = str(exc)
            continue

        # Otherwise try to load TU data and generate splits.
        try:
            labels_by_dataset[dataset_name] = _load_tu_labels(dataset_name=dataset_name, root=root)
        except Exception as exc:  # noqa: BLE001 - collect/report all dataset failures
            loading_failures[dataset_name] = str(exc)
            continue

        try:
            generated = _build_splits(labels=labels_by_dataset[dataset_name], folds=folds, seed=resolved_seed)
            target_path.write_text(json.dumps(generated), encoding="utf-8")
            generated_count += 1
        except Exception as exc:  # noqa: BLE001 - keep processing remaining datasets
            processing_failures[dataset_name] = str(exc)

    success_count = copied_count + generated_count
    failed_count = len(copy_failures) + len(loading_failures) + len(processing_failures)
    print(
        f"Split generation summary: requested={len(requested_datasets)}, "
        f"copied={copied_count}, generated={generated_count}, failed={failed_count}"
    )

    if copy_failures:
        print("Copy failures:")
        for name, msg in sorted(copy_failures.items()):
            print(f"- {name}: {msg}")

    if loading_failures:
        print("Load failures:")
        for name, msg in sorted(loading_failures.items()):
            print(f"- {name}: {msg}")

    if processing_failures:
        print("Processing failures:")
        for name, msg in sorted(processing_failures.items()):
            print(f"- {name}: {msg}")

    if success_count == 0:
        raise RuntimeError("No split file was copied or generated successfully.")

    return success_count, failed_count


def _parse_args() -> argparse.Namespace:
    src_root = Path(__file__).resolve().parents[2]
    repo_root = src_root.parent

    parser = argparse.ArgumentParser(description="Generate fixed 10-fold TU dataset splits.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=src_root / "simplegnn" / "datasets" / "splits" / "tu_splits",
        help="Output directory for <dataset>_splits.json files.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "data" / "TUDataset",
        help="Root directory used by torch-geometric TUDataset.",
    )
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=None, help="Fixed RNG seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files in output directory.",
    )
    parser.add_argument(
        "--match-standard-dir",
        type=Path,
        default=src_root / "simplegnn" / "datasets" / "splits" / "standard",
        help="Source split directory to copy from when <dataset>_splits.json exists.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names. Defaults to full TU list.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    datasets = None
    if args.datasets:
        datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    try:
        generate_tu_splits(
            output_dir=args.output_dir,
            root=args.root,
            folds=args.folds,
            seed=args.seed,
            overwrite=args.overwrite,
            standard_dir=args.match_standard_dir,
            datasets=datasets,
        )
    except Exception as exc:  # noqa: BLE001 - user-facing CLI
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
