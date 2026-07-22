"""Hash-keyed cross-dataset transfer (spec 18, Part B).

This module contains the dataset-boundary machinery built on the canonical
label hashes of Part A (``datasets/utils/label_hashing.py``):

- B0: overlap measurement between two independently preprocessed datasets
  (:func:`measure_label_overlap`, :func:`measure_pair_overlap`) — the go/no-go
  gate for any transfer attempt,
- B3: the remap engine (:func:`apply_transfer`) that copies a source
  checkpoint into a freshly built target network, matching invariant-layer
  weight slots by ``(src_hash, tgt_hash, property_key)`` instead of by
  dataset-relative label ids, plus checkpoint/sidecar resolution and the
  freeze/linear-probe strategies.

The reserved hash values (``RESERVED_INVALID``, ``RESERVED_CAPPED``) are data,
not identity: any row containing one is excluded from every join.
"""
from __future__ import annotations

import fnmatch
import gzip
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from simplegnn.datasets.utils.NodeLabels import NodeLabels
from simplegnn.datasets.utils.label_hashing import RESERVED_CAPPED, RESERVED_INVALID
from simplegnn.datasets.utils.node_labeling import load_labels

MISSING_SIDECAR_HINT = (
    "re-save the source model with save_transfer_keys enabled (and regenerate "
    "the source labels for v2 hashes if they predate spec 18 Part A)"
)
MISSING_HASHES_HINT = (
    "delete the labels directory of the dataset (paths['labels']/<dataset>) and "
    "rerun preprocessing() to regenerate v2 label files with canonical hashes"
)


# --------------------------------------------------------------------- helpers
def label_ids_to_hashes(ids: torch.Tensor, node_labels: NodeLabels) -> torch.Tensor:
    """
    Map frequency-sorted label ids to their canonical int64 hashes.

    Ids outside ``[0, len(label_hashes))`` (notably the ``-1`` invalid label)
    and ids of labels without a hash vocabulary map to ``RESERVED_INVALID``.
    """
    ids = ids.detach().cpu().long()
    out = torch.full(ids.shape, RESERVED_INVALID, dtype=torch.int64)
    hashes = node_labels.label_hashes
    if hashes is None:
        return out
    hashes = hashes.detach().cpu().long()
    valid = (ids >= 0) & (ids < hashes.numel())
    out[valid] = hashes[ids[valid]]
    return out


def _as_2d_int64(columns: Sequence) -> np.ndarray:
    """Stack 1-D int64 columns (tensors or arrays) into an (n, c) numpy array."""
    cols = []
    for column in columns:
        if isinstance(column, torch.Tensor):
            column = column.detach().cpu().numpy()
        cols.append(np.asarray(column, dtype=np.int64).reshape(-1))
    return np.stack(cols, axis=1) if cols else np.empty((0, 0), dtype=np.int64)


def _reserved_row_mask(rows: np.ndarray) -> np.ndarray:
    """True for rows containing a reserved hash value in any column."""
    return ((rows == RESERVED_INVALID) | (rows == RESERVED_CAPPED)).any(axis=1)


def hash_join(source_columns: Sequence, target_columns: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized equi-join of two hash-keyed row sets.

    ``source_columns`` / ``target_columns`` are sequences of equal-length 1-D
    int64 tensors/arrays (e.g. ``(src_hash, tgt_hash)``). Returns
    ``(src_rows, tgt_rows)``: for every target row whose full column tuple
    also occurs in the source, the index of one matching source row and the
    target row index. Rows containing ``RESERVED_INVALID`` or
    ``RESERVED_CAPPED`` in any column NEVER match (reserved values are
    placeholders, not identity).
    """
    s = _as_2d_int64(source_columns)
    t = _as_2d_int64(target_columns)
    if s.size == 0 or t.size == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty
    s_rows = np.nonzero(~_reserved_row_mask(s))[0]
    t_rows = np.nonzero(~_reserved_row_mask(t))[0]
    if s_rows.size == 0 or t_rows.size == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty
    stacked = np.concatenate([s[s_rows], t[t_rows]], axis=0)
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    group_to_source = np.full(int(inverse.max()) + 1, -1, dtype=np.int64)
    # source vocabularies are unique per key, so at most one source row per group
    group_to_source[inverse[: s_rows.size]] = s_rows
    matched_source = group_to_source[inverse[s_rows.size:]]
    hit = matched_source >= 0
    return (torch.from_numpy(matched_source[hit]),
            torch.from_numpy(t_rows[hit].astype(np.int64)))


def _as_node_labels(labels: Union[NodeLabels, str, Path]) -> NodeLabels:
    if isinstance(labels, NodeLabels):
        return labels
    return load_labels(path=labels)


def _require_hash_vocabulary(node_labels: NodeLabels, role: str) -> None:
    if node_labels.label_hashes is None:
        raise ValueError(
            f"{role} label file '{node_labels.label_name}' of dataset "
            f"'{node_labels.dataset_name}' has no hash vocabulary (legacy v1 format): "
            f"{MISSING_HASHES_HINT}."
        )


def _per_node_hashes(node_labels: NodeLabels) -> np.ndarray:
    return label_ids_to_hashes(node_labels.node_labels, node_labels).numpy()


def _valid_vocabulary(node_labels: NodeLabels) -> np.ndarray:
    hashes = node_labels.label_hashes.detach().cpu().numpy().astype(np.int64)
    return np.unique(hashes[(hashes != RESERVED_INVALID) & (hashes != RESERVED_CAPPED)])


def load_property_pairs(path: Union[str, Path]) -> Dict:
    """
    Load a pairwise-property file (``<db>_properties_<name>.pt``): a gzipped
    pickle of ``(valid_properties, pairs_dict, slices_dict)`` where
    ``pairs_dict`` maps property value -> (num_pairs, 2) global node-index
    tensor. Returns ``pairs_dict``.
    """
    with open(path, 'rb') as f:
        _, pairs_dict, _ = pickle.loads(gzip.decompress(f.read()))
    return pairs_dict


# ------------------------------------------------------------- B0: overlap gate
@dataclass
class OverlapReport:
    """Vocabulary/occurrence overlap between a source and a target dataset."""
    kind: str                     # 'labels' (unique label hashes) or 'pairs' ((src, tgt, value) triples)
    source_name: str
    target_name: str
    label_name: str
    source_unique: int            # non-reserved unique keys in the source
    target_unique: int
    shared_unique: int
    target_occurrences: int       # every target occurrence, reserved ones included
    covered_occurrences: int      # target occurrences whose key exists in the source
    target_reserved_occurrences: int  # occurrences excluded from matching (invalid/capped)
    source_canonical: bool
    target_canonical: bool
    details: dict = field(default_factory=dict)

    @property
    def unique_overlap(self) -> float:
        return self.shared_unique / self.target_unique if self.target_unique else 0.0

    @property
    def weighted_coverage(self) -> float:
        return self.covered_occurrences / self.target_occurrences if self.target_occurrences else 0.0


def measure_label_overlap(source_label_file: Union[NodeLabels, str, Path],
                          target_label_file: Union[NodeLabels, str, Path]) -> OverlapReport:
    """
    Unique-label overlap and occurrence-weighted coverage of a target
    dataset's label vocabulary by a source dataset's (spec 18 B0).

    Both arguments are v2 label ``.pt`` files (or loaded ``NodeLabels``);
    files without a hash vocabulary raise with a migration hint.
    """
    source = _as_node_labels(source_label_file)
    target = _as_node_labels(target_label_file)
    _require_hash_vocabulary(source, 'Source')
    _require_hash_vocabulary(target, 'Target')

    source_vocab = _valid_vocabulary(source)
    target_vocab = _valid_vocabulary(target)
    shared = np.intersect1d(source_vocab, target_vocab, assume_unique=True)

    target_node_hashes = _per_node_hashes(target)
    reserved = (target_node_hashes == RESERVED_INVALID) | (target_node_hashes == RESERVED_CAPPED)
    covered = np.isin(target_node_hashes, shared) & ~reserved

    return OverlapReport(
        kind='labels',
        source_name=source.dataset_name,
        target_name=target.dataset_name,
        label_name=target.label_name,
        source_unique=int(source_vocab.size),
        target_unique=int(target_vocab.size),
        shared_unique=int(shared.size),
        target_occurrences=int(target_node_hashes.size),
        covered_occurrences=int(covered.sum()),
        target_reserved_occurrences=int(reserved.sum()),
        source_canonical=source.has_canonical_hashes,
        target_canonical=target.has_canonical_hashes,
    )


def _pair_triples(node_labels: NodeLabels, pairs_dict: Dict, property_values) -> Dict:
    """
    Per property value: unique non-reserved (src_hash, tgt_hash) rows plus
    per-row occurrence counts and the number of reserved/total occurrences.
    """
    node_hashes = _per_node_hashes(node_labels)
    out = {}
    for key in property_values:
        pairs = pairs_dict[key]
        if isinstance(pairs, torch.Tensor):
            pairs = pairs.detach().cpu().numpy()
        pairs = np.asarray(pairs, dtype=np.int64)
        rows = np.stack([node_hashes[pairs[:, 0]], node_hashes[pairs[:, 1]]], axis=1)
        reserved = _reserved_row_mask(rows)
        unique_rows, counts = (np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64))
        if (~reserved).any():
            unique_rows, counts = np.unique(rows[~reserved], axis=0, return_counts=True)
        out[key] = {
            'unique_rows': unique_rows,
            'counts': counts,
            'total_occurrences': int(rows.shape[0]),
            'reserved_occurrences': int(reserved.sum()),
        }
    return out


def measure_pair_overlap(source_labels: Union[NodeLabels, str, Path],
                         source_props: Union[Dict, str, Path],
                         target_labels: Union[NodeLabels, str, Path],
                         target_props: Union[Dict, str, Path],
                         property_values: Optional[Sequence] = None) -> OverlapReport:
    """
    Occurrence-weighted coverage of the target's
    ``(src_hash, tgt_hash, property_value)`` triples by the source (spec 18
    B0) — the quantity that upper-bounds what :func:`apply_transfer` can
    transfer for a conv head using these labels/properties.

    ``source_props`` / ``target_props`` are property files
    (``<db>_properties_<name>.pt``) or already-loaded ``{value: (n, 2) pairs}``
    dicts. ``property_values`` restricts the property values considered
    (default: the values present in both datasets).
    """
    source = _as_node_labels(source_labels)
    target = _as_node_labels(target_labels)
    _require_hash_vocabulary(source, 'Source')
    _require_hash_vocabulary(target, 'Target')
    if not isinstance(source_props, dict):
        source_props = load_property_pairs(source_props)
    if not isinstance(target_props, dict):
        target_props = load_property_pairs(target_props)

    if property_values is None:
        property_values = sorted(set(source_props) & set(target_props), key=str)
    else:
        property_values = [v for v in property_values
                           if v in source_props and v in target_props]

    source_triples = _pair_triples(source, source_props, property_values)
    target_triples = _pair_triples(target, target_props, property_values)

    source_unique = target_unique = shared_unique = 0
    total_occurrences = covered_occurrences = reserved_occurrences = 0
    per_value = {}
    for key in property_values:
        s, t = source_triples[key], target_triples[key]
        src_rows, tgt_rows = hash_join(
            (s['unique_rows'][:, 0], s['unique_rows'][:, 1]),
            (t['unique_rows'][:, 0], t['unique_rows'][:, 1]))
        covered = int(t['counts'][tgt_rows.numpy()].sum()) if tgt_rows.numel() else 0
        source_unique += int(s['unique_rows'].shape[0])
        target_unique += int(t['unique_rows'].shape[0])
        shared_unique += int(tgt_rows.numel())
        total_occurrences += t['total_occurrences']
        covered_occurrences += covered
        reserved_occurrences += t['reserved_occurrences']
        per_value[str(key)] = {
            'target_unique': int(t['unique_rows'].shape[0]),
            'shared_unique': int(tgt_rows.numel()),
            'target_occurrences': t['total_occurrences'],
            'covered_occurrences': covered,
        }

    return OverlapReport(
        kind='pairs',
        source_name=source.dataset_name,
        target_name=target.dataset_name,
        label_name=target.label_name,
        source_unique=source_unique,
        target_unique=target_unique,
        shared_unique=shared_unique,
        target_occurrences=total_occurrences,
        covered_occurrences=covered_occurrences,
        target_reserved_occurrences=reserved_occurrences,
        source_canonical=source.has_canonical_hashes,
        target_canonical=target.has_canonical_hashes,
        details={'per_value': per_value},
    )


# --------------------------------------------------- B2/B3: sidecar + transfer
def load_transfer_sidecar(sidecar_path: Union[str, Path]) -> dict:
    """Load a ``<model>.keys.pt`` sidecar; a missing file raises with the fix."""
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Transfer-key sidecar {sidecar_path} not found: {MISSING_SIDECAR_HINT}.")
    payload = torch.load(str(sidecar_path), weights_only=False)
    if not isinstance(payload, dict) or 'layers' not in payload:
        raise ValueError(f"Transfer-key sidecar {sidecar_path} has an unexpected format")
    if payload.get('schema') != 1:
        raise ValueError(
            f"Transfer-key sidecar {sidecar_path} has schema {payload.get('schema')!r}, "
            f"expected 1: {MISSING_SIDECAR_HINT}.")
    return payload


def sidecar_path_for(checkpoint_path: Union[str, Path]) -> Path:
    """``.../model_x.pt`` -> ``.../model_x.keys.pt``."""
    return Path(checkpoint_path).with_suffix('.keys.pt')


def _validation_score(csv_path: Path):
    """
    (metric, higher_is_better) of one per-epoch result CSV at its best epoch.

    Classification runs record ``ValidationAccuracy`` in percent; regression
    runs leave it at 0 and carry the error in ``ValidationLoss``/
    ``ValidationMAE``, so the column that actually varies decides the direction.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, sep=';')
    if df.empty:
        return None
    accuracy = df['ValidationAccuracy'] if 'ValidationAccuracy' in df.columns else None
    if accuracy is not None and float(accuracy.abs().max()) > 0.0:
        return float(accuracy.max()), True
    return float(df['ValidationLoss'].min()), False


def _best_validation_checkpoint(models_dir: Path, results_dir: Path) -> Path:
    """
    Checkpoint of the (config, run, validation step) with the best validation
    score, mirroring how the framework selects a model within a run.
    """
    result_files = sorted(results_dir.glob('*_Best_Configuration_*_Results_run_id_*.csv')) or \
        sorted(f for f in results_dir.glob('*_Configuration_*_Results_run_id_*.csv')
               if 'Best_Configuration' not in f.name)
    scored = []
    for csv_path in result_files:
        # <db>_[Best_]Configuration_<id>_Results_run_id_<r>_validation_step_<v>.csv
        stem = csv_path.stem
        best = '_Best_Configuration_' in stem
        config_id = stem.split('Configuration_')[1].split('_')[0]
        run_id = stem.split('run_id_')[1].split('_')[0]
        validation_id = stem.split('validation_step_')[1]
        prefix = 'model_Best_Configuration_' if best else 'model_Configuration_'
        checkpoint = models_dir / f'{prefix}{config_id}_run_{run_id}_val_step_{validation_id}.pt'
        if not checkpoint.exists():
            continue
        score = _validation_score(csv_path)
        if score is not None:
            scored.append((score[0], score[1], checkpoint))
    if not scored:
        raise FileNotFoundError(
            f"transfer.source.select: 'best_validation' found no result CSV in {results_dir} "
            f"with a matching checkpoint in {models_dir}.")
    higher_is_better = scored[0][1]
    best_score, _, checkpoint = (max if higher_is_better else min)(scored, key=lambda item: item[0])
    print(f"transfer.source.select: 'best_validation' picked {checkpoint.name} "
          f"(validation {'accuracy' if higher_is_better else 'loss'} {best_score:.4f} "
          f"of {len(scored)} checkpoints)")
    return checkpoint


def resolve_source_checkpoint(results_path: Union[str, Path], dataset: str,
                              select: Union[str, dict, None] = 'best') -> Path:
    """
    Resolve the source checkpoint of a ``transfer.source`` block.

    ``select == 'best'`` (default) prefers the best-configuration re-run
    checkpoints (``model_Best_Configuration_*``, run 0 / validation step 0,
    lowest config id), falling back to the grid-search checkpoints — note that
    this is positional: it takes run 0, not the run that scored best.
    ``select == 'best_validation'`` instead compares the validation scores of
    every checkpoint of the source run and takes the winner. A dict
    ``{config_id, run_id, validation_id}`` addresses one checkpoint exactly.
    """
    models_dir = Path(results_path).joinpath(dataset, 'Models')
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Source model directory {models_dir} not found — run the pretraining "
            f"experiment with save_best_model/best_model enabled first.")

    if isinstance(select, dict):
        config_id = int(select.get('config_id', 0))
        run_id = int(select.get('run_id', 0))
        validation_id = int(select.get('validation_id', 0))
        candidates = [
            models_dir / f'model_Best_Configuration_{str(config_id).zfill(6)}_run_{run_id}_val_step_{validation_id}.pt',
            models_dir / f'model_Configuration_{str(config_id).zfill(6)}_run_{run_id}_val_step_{validation_id}.pt',
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No source checkpoint for config_id={config_id}, run_id={run_id}, "
            f"validation_id={validation_id} in {models_dir} "
            f"(tried {', '.join(c.name for c in candidates)})")

    if select == 'best_validation':
        return _best_validation_checkpoint(models_dir, Path(results_path).joinpath(dataset, 'Results'))

    if select in (None, 'best', 'Best'):
        for pattern in ('model_Best_Configuration_*_run_0_val_step_0.pt',
                        'model_Best_Configuration_*.pt',
                        'model_*_run_0_val_step_0.pt',
                        'model_*.pt'):
            matches = sorted(p for p in models_dir.glob(pattern)
                             if not p.name.endswith('.keys.pt'))
            if matches:
                if len(matches) > 1:
                    print(f"transfer.source.select: 'best' matched {len(matches)} checkpoints "
                          f"in {models_dir}; using {matches[0].name}")
                return matches[0]
        raise FileNotFoundError(f"No source checkpoints found in {models_dir}")

    raise ValueError(
        f"transfer.source.select must be 'best', 'best_validation' or a dict with "
        f"config_id/run_id/validation_id, got {select!r}")


@dataclass
class TransferReport:
    """Per-layer outcome of one :func:`apply_transfer` call."""
    layers: List[dict] = field(default_factory=list)     # see _layer_entry for keys
    warnings: List[str] = field(default_factory=list)
    frozen_parameters: List[str] = field(default_factory=list)
    source_checkpoint: str = ''

    @property
    def matched_parameters(self) -> int:
        return sum(entry['matched'] for entry in self.layers)

    @property
    def total_parameters(self) -> int:
        return sum(entry['total'] for entry in self.layers)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"⚠ transfer: {message}")

    def to_dict(self) -> dict:
        return {
            'source_checkpoint': self.source_checkpoint,
            'matched_parameters': self.matched_parameters,
            'total_parameters': self.total_parameters,
            'layers': self.layers,
            'warnings': self.warnings,
            'frozen_parameters': self.frozen_parameters,
        }

    def format(self) -> str:
        lines = [f"Transfer report (source: {self.source_checkpoint or 'in-memory'})",
                 f"{'layer':<24}{'type':<40}{'action':<24}{'matched/total':>18}{'coverage':>10}"]
        for entry in self.layers:
            total = entry['total']
            coverage = f"{entry['matched'] / total:6.1%}" if total else '     -'
            lines.append(f"{entry['layer']:<24}{entry['layer_type']:<40}{entry['action']:<24}"
                         f"{entry['matched']:>9}/{total:<8}{coverage:>10}")
        lines.append(f"total matched parameters: {self.matched_parameters}/{self.total_parameters}")
        for message in self.warnings:
            lines.append(f"warning: {message}")
        if self.frozen_parameters:
            lines.append(f"frozen parameters: {len(self.frozen_parameters)}")
        return '\n'.join(lines)


def _layer_entry(prefix: str, layer_type: str, action: str, matched: int, total: int,
                 transferred: bool, notes: Optional[List[str]] = None) -> dict:
    return {'layer': prefix, 'layer_type': layer_type, 'action': action,
            'matched': int(matched), 'total': int(total),
            'transferred': bool(transferred), 'notes': notes or []}


def _head_usable(head: dict, allow_non_canonical: bool, prefix: str,
                 report: TransferReport, side: str) -> bool:
    """Canonical-hash gating for one exported head (source or target side)."""
    if not head.get('has_hashes', True):
        raise ValueError(
            f"{side} head {head.get('head_id')} of {prefix} uses label "
            f"'{head.get('source_label', head.get('label', head.get('bias_label')))}' without a "
            f"hash vocabulary (legacy v1 label file): {MISSING_HASHES_HINT}.")
    if not head.get('canonical', False) and not allow_non_canonical:
        report.warn(
            f"{prefix} head {head.get('head_id')}: {side.lower()} labels are not canonical "
            f"(dataset-relative); skipped (set invariant_transfer.allow_non_canonical: True "
            f"to match them anyway)")
        return False
    if not _property_canonical(head) and not allow_non_canonical:
        report.warn(
            f"{prefix} head {head.get('head_id')}: {side.lower()} property "
            f"'{head.get('property')}' has dataset-relative keys; skipped (set "
            f"invariant_transfer.allow_non_canonical: True to match them anyway)")
        return False
    return True


def _property_canonical(head: dict) -> bool:
    """Whether the head's property keys are dataset-independent.

    Sidecars written before the flag existed derive it from the property
    description instead of assuming canonicality."""
    flag = head.get('property_canonical')
    if flag is not None:
        return bool(flag)
    return not str(head.get('property', '')).startswith('edge_label_distances')


def _match_conv_layer(prefix: str, layer_keys: dict, source_layer_keys: dict,
                      target_sd: dict, source_sd: dict, on_missing: str,
                      allow_non_canonical: bool, report: TransferReport) -> Tuple[int, int]:
    """Hash-join transfer for an invariant message-passing layer."""
    matched = 0
    weight = target_sd[f'{prefix}.Param_W']
    source_weight = source_sd.get(f'{prefix}.Param_W')
    total = weight.numel()
    if source_weight is None:
        report.warn(f"{prefix}: source state dict has no Param_W; layer left on fresh init")
        return 0, total
    if on_missing == 'zero':
        weight.zero_()
    source_heads = {h['head_id']: h for h in source_layer_keys.get('heads', [])}
    for t_head in layer_keys.get('heads', []):
        s_head = source_heads.get(t_head['head_id'])
        if s_head is None:
            report.warn(f"{prefix} head {t_head['head_id']}: no matching source head")
            continue
        if (s_head.get('source_label') != t_head.get('source_label')
                or s_head.get('target_label') != t_head.get('target_label')
                or s_head.get('property') != t_head.get('property')):
            report.warn(f"{prefix} head {t_head['head_id']}: source/target head "
                        f"configurations differ; skipped")
            continue
        if not (_head_usable(t_head, allow_non_canonical, prefix, report, 'Target')
                and _head_usable(s_head, allow_non_canonical, prefix, report, 'Source')):
            continue
        replicas = min(int(s_head['num_replicas']), int(t_head['num_replicas']))
        source_keys_by_value = {k['property_key']: k for k in s_head['keys']}
        for t_key in t_head['keys']:
            s_key = source_keys_by_value.get(t_key['property_key'])
            if s_key is None:
                continue
            src_rows, tgt_rows = hash_join((s_key['src_hash'], s_key['tgt_hash']),
                                           (t_key['src_hash'], t_key['tgt_hash']))
            if tgt_rows.numel() == 0:
                continue
            for n in range(replicas):
                t_idx = t_key['param_offset'] + n * t_key['num_weights'] + tgt_rows
                s_idx = s_key['param_offset'] + n * s_key['num_weights'] + src_rows
                weight[t_idx] = source_weight[s_idx].to(weight.dtype)
            matched += int(tgt_rows.numel()) * replicas

    bias = target_sd.get(f'{prefix}.Param_b')
    source_bias = source_sd.get(f'{prefix}.Param_b')
    if bias is not None:
        total += bias.numel()
        if source_bias is None:
            report.warn(f"{prefix}: source state dict has no Param_b; bias left on fresh init")
        else:
            if on_missing == 'zero':
                bias.zero_()
            source_bias_heads = {b['head_id']: b for b in source_layer_keys.get('bias', [])}
            for t_bias in layer_keys.get('bias', []):
                s_bias = source_bias_heads.get(t_bias['head_id'])
                if s_bias is None or s_bias.get('bias_label') != t_bias.get('bias_label'):
                    continue
                if not (_head_usable(t_bias, allow_non_canonical, prefix, report, 'Target')
                        and _head_usable(s_bias, allow_non_canonical, prefix, report, 'Source')):
                    continue
                features = int(t_bias['in_features'])
                if int(s_bias['in_features']) != features:
                    report.warn(f"{prefix} bias head {t_bias['head_id']}: in_features differ; skipped")
                    continue
                src_rows, tgt_rows = hash_join((s_bias['bias_hash'],), (t_bias['bias_hash'],))
                if tgt_rows.numel() == 0:
                    continue
                replicas = min(int(s_bias['num_replicas']), int(t_bias['num_replicas']))
                for n in range(replicas):
                    for f in range(features):
                        t_idx = (t_bias['bias_base'] + n * features * t_bias['n_bias']
                                 + f * t_bias['n_bias'] + tgt_rows)
                        s_idx = (s_bias['bias_base'] + n * features * s_bias['n_bias']
                                 + f * s_bias['n_bias'] + src_rows)
                        bias[t_idx] = source_bias[s_idx].to(bias.dtype)
                matched += int(tgt_rows.numel()) * replicas * features
    return matched, total


def _match_table_layer(prefix: str, layer_keys: dict, source_layer_keys: dict,
                       target_sd: dict, source_sd: dict, on_missing: str,
                       allow_non_canonical: bool, report: TransferReport) -> Tuple[int, int]:
    """Hash-join transfer for aggregation/positional-encoding label tables."""
    matched = 0
    weight = target_sd[f'{prefix}.Param_W']
    source_weight = source_sd.get(f'{prefix}.Param_W')
    total = weight.numel()
    if source_weight is None:
        report.warn(f"{prefix}: source state dict has no Param_W; layer left on fresh init")
        return 0, total
    if on_missing == 'zero':
        weight.zero_()
    source_heads = {h['head_id']: h for h in source_layer_keys.get('heads', [])}
    for t_head in layer_keys.get('heads', []):
        s_head = source_heads.get(t_head['head_id'])
        if s_head is None or s_head.get('label') != t_head.get('label'):
            report.warn(f"{prefix} head {t_head['head_id']}: no matching source head")
            continue
        if not (_head_usable(t_head, allow_non_canonical, prefix, report, 'Target')
                and _head_usable(s_head, allow_non_canonical, prefix, report, 'Source')):
            continue
        src_rows, tgt_rows = hash_join((s_head['label_hash'],), (t_head['label_hash'],))
        if tgt_rows.numel() == 0:
            continue
        if 'weight_base' in t_head:                       # aggregation layout
            replicas = min(int(s_head['num_replicas']), int(t_head['num_replicas']))
            for n in range(replicas):
                t_idx = t_head['weight_base'] + n * t_head['n_labels'] + tgt_rows
                s_idx = s_head['weight_base'] + n * s_head['n_labels'] + src_rows
                weight[t_idx] = source_weight[s_idx].to(weight.dtype)
            matched += int(tgt_rows.numel()) * replicas
        else:                                             # positional-encoding layout
            entries = min(int(s_head['num_entries']), int(t_head['num_entries']))
            k = torch.arange(entries, dtype=torch.int64)
            t_idx = (t_head['offset'] + tgt_rows[:, None] * t_head['num_entries'] + k[None, :]).reshape(-1)
            s_idx = (s_head['offset'] + src_rows[:, None] * s_head['num_entries'] + k[None, :]).reshape(-1)
            weight[t_idx] = source_weight[s_idx].to(weight.dtype)
            matched += int(tgt_rows.numel()) * entries

    bias = target_sd.get(f'{prefix}.Param_b')
    source_bias = source_sd.get(f'{prefix}.Param_b')
    if bias is not None:
        total += bias.numel()
        # the aggregation bias is (num_heads, in_features): config-shaped, not
        # dataset-shaped, so it transfers positionally when shapes agree
        if source_bias is not None and tuple(source_bias.shape) == tuple(bias.shape):
            bias.copy_(source_bias.to(bias.dtype))
            matched += bias.numel()
        else:
            report.warn(f"{prefix}: bias shape mismatch or missing; left on fresh init")
    return matched, total


def apply_transfer(target_net, source_state_dict: dict, source_keys: dict,
                   cfg: Optional[dict]) -> TransferReport:
    """
    Copy a source checkpoint into a freshly initialized target network
    (spec 18 B3).

    - Standard layers transfer by state-dict name when shapes match; the head
      (last linear layer) keeps its fresh init unless ``head.reinit: never``.
    - Invariant layers transfer per weight slot through a vectorized hash join
      of the exported ``(src_hash, tgt_hash, property_key)`` keys; unmatched
      slots keep their fresh init (``invariant_transfer.on_missing: reinit``)
      or are zeroed (``zero``). Reserved hashes never match.
    - ``random_init: true`` skips every weight copy (standard *and*
      invariant layers keep their fresh initialization) while still
      resolving/validating the source checkpoint and sidecar and producing a
      report — the "untrained backbone" counterpart of a real transfer run,
      for use as a sanity-check baseline (paired with
      ``strategy: linear_probe`` so only the head trains).

    ``cfg`` is the ``transfer:`` block of the hyperparameter config (the
    ``source``/``strategy``/``freeze`` keys are consumed by the callers, not
    here). Modifies ``target_net`` in place and returns a
    :class:`TransferReport`.
    """
    cfg = cfg or {}
    invariant_cfg = cfg.get('invariant_transfer') or {}
    match = invariant_cfg.get('match', 'hashes')
    on_missing = invariant_cfg.get('on_missing', 'reinit')
    allow_non_canonical = bool(invariant_cfg.get('allow_non_canonical', False))
    min_overlap_warn = float(invariant_cfg.get('min_overlap_warn', 0.10))
    head_reinit = (cfg.get('head') or {}).get('reinit', 'always')
    random_init = bool(cfg.get('random_init', False))
    if match not in ('hashes', 'none'):
        raise ValueError(f"invariant_transfer.match must be 'hashes' or 'none', got {match!r}")
    if on_missing not in ('reinit', 'zero'):
        raise ValueError(f"invariant_transfer.on_missing must be 'reinit' or 'zero', got {on_missing!r}")
    if head_reinit not in ('always', 'never'):
        raise ValueError(f"head.reinit must be 'always' or 'never', got {head_reinit!r}")

    report = TransferReport()
    target_sd = target_net.state_dict()

    net_layers = getattr(target_net, 'net_layers', None) or []
    invariant_prefixes = {f'net_layers.{i}': layer for i, layer in enumerate(net_layers)
                          if hasattr(layer, 'export_weight_keys')}
    head_prefix = None
    if head_reinit == 'always':
        linear_ids = [i for i, layer in enumerate(net_layers)
                      if type(layer).__name__ == 'LinearLayer']
        if linear_ids:
            head_prefix = f'net_layers.{linear_ids[-1]}'

    with torch.no_grad():
        # ---- standard layers: copy by name + shape --------------------------
        standard: Dict[str, List[str]] = {}
        for name in target_sd:
            top = '.'.join(name.split('.')[:2])
            if top in invariant_prefixes:
                continue
            standard.setdefault(top, []).append(name)
        for top, names in standard.items():
            layer_type = (type(net_layers[int(top.split('.')[1])]).__name__
                          if top.startswith('net_layers.') and top.split('.')[1].isdigit()
                          and int(top.split('.')[1]) < len(net_layers) else 'unknown')
            if top == head_prefix:
                report.layers.append(_layer_entry(
                    top, layer_type, 'head_reinit', 0,
                    sum(target_sd[n].numel() for n in names), False))
                continue
            if random_init:
                report.layers.append(_layer_entry(
                    top, layer_type, 'random_init', 0,
                    sum(target_sd[n].numel() for n in names), False))
                continue
            copied = 0
            total = 0
            notes = []
            for name in names:
                tensor = target_sd[name]
                total += tensor.numel()
                source_tensor = source_state_dict.get(name)
                if source_tensor is None:
                    notes.append(f'{name}: missing in source')
                elif tuple(source_tensor.shape) != tuple(tensor.shape):
                    notes.append(f'{name}: shape {tuple(source_tensor.shape)} != {tuple(tensor.shape)}')
                    report.warn(f"{name}: shape mismatch "
                                f"{tuple(source_tensor.shape)} vs {tuple(tensor.shape)}; kept fresh init "
                                f"(use input_features: {{name: constant}} for transferable widths)")
                else:
                    tensor.copy_(source_tensor.to(tensor.dtype))
                    copied += tensor.numel()
            action = 'copied' if copied == total and total else ('partial_copy' if copied else 'reinit')
            report.layers.append(_layer_entry(top, layer_type, action, copied, total,
                                              copied > 0, notes))

        # ---- invariant layers: hash-keyed join ------------------------------
        for prefix, layer in invariant_prefixes.items():
            layer_type = type(layer).__name__
            total_estimate = sum(target_sd[n].numel() for n in target_sd
                                 if n.startswith(prefix + '.'))
            if random_init:
                report.layers.append(_layer_entry(prefix, layer_type, 'random_init',
                                                  0, total_estimate, False))
                continue
            if match == 'none':
                report.layers.append(_layer_entry(prefix, layer_type, 'reinit',
                                                  0, total_estimate, False))
                continue
            source_layer_keys = (source_keys or {}).get('layers', {}).get(prefix)
            if source_layer_keys is None:
                report.warn(f"{prefix}: source sidecar has no keys for this layer "
                            f"({MISSING_SIDECAR_HINT}); layer left on fresh init")
                report.layers.append(_layer_entry(prefix, layer_type, 'reinit',
                                                  0, total_estimate, False))
                continue
            layer_keys = layer.export_weight_keys()
            if source_layer_keys.get('layer_type') != layer_keys.get('layer_type'):
                report.warn(f"{prefix}: source layer type "
                            f"{source_layer_keys.get('layer_type')!r} != target "
                            f"{layer_keys.get('layer_type')!r}; layer left on fresh init")
                report.layers.append(_layer_entry(prefix, layer_type, 'reinit',
                                                  0, total_estimate, False))
                continue
            if layer_keys.get('layer_type') == 'invariant_based_convolution':
                matched, hashed_total = _match_conv_layer(
                    prefix, layer_keys, source_layer_keys, target_sd, source_state_dict,
                    on_missing, allow_non_canonical, report)
            else:
                matched, hashed_total = _match_table_layer(
                    prefix, layer_keys, source_layer_keys, target_sd, source_state_dict,
                    on_missing, allow_non_canonical, report)
            coverage = matched / hashed_total if hashed_total else 0.0
            if hashed_total and coverage < min_overlap_warn:
                report.warn(f"{prefix}: only {coverage:.1%} of the invariant parameters "
                            f"matched (min_overlap_warn={min_overlap_warn:.0%})")
            # config-shaped sub-parameters of the invariant layer (e.g. the
            # pre_norm LayerNorm) are not hash-keyed; copy them by name+shape
            for name in target_sd:
                if not name.startswith(prefix + '.') or name[len(prefix) + 1:] in ('Param_W', 'Param_b'):
                    continue
                tensor = target_sd[name]
                source_tensor = source_state_dict.get(name)
                if source_tensor is not None and tuple(source_tensor.shape) == tuple(tensor.shape):
                    tensor.copy_(source_tensor.to(tensor.dtype))
                    matched += tensor.numel()
            report.layers.append(_layer_entry(prefix, layer_type, 'hash_join',
                                              matched, total_estimate, matched > 0))
    return report


def apply_transfer_strategy(target_net, cfg: Optional[dict], report: TransferReport) -> List[str]:
    """
    Apply ``strategy`` / ``freeze`` of a ``transfer:`` block (spec 18 B3.5).

    - ``strategy: finetune`` (default): nothing frozen beyond ``freeze``.
    - ``strategy: linear_probe``: every layer that received transferred
      weights is frozen; the (re-initialized) head stays trainable. Under
      ``random_init: true`` nothing is ever "transferred", so instead every
      layer that isn't the (re-initialized) head is frozen at its random
      init — the untrained-backbone baseline.
    - ``freeze``: state-dict prefix globs, e.g. ``['net_layers.0*']``.

    Freezing sets ``requires_grad = False``; ``set_optimizer`` builds its
    parameter groups from the remaining trainable parameters.
    """
    cfg = cfg or {}
    strategy = cfg.get('strategy', 'finetune')
    if strategy not in ('finetune', 'linear_probe'):
        raise ValueError(f"transfer.strategy must be 'finetune' or 'linear_probe', got {strategy!r}")
    patterns = [str(p) for p in (cfg.get('freeze') or [])]
    if strategy == 'linear_probe':
        if cfg.get('random_init', False):
            patterns += [entry['layer'] for entry in report.layers if entry['action'] != 'head_reinit']
        else:
            patterns += [entry['layer'] for entry in report.layers if entry['transferred']]

    frozen = []
    if patterns:
        for name, param in target_net.named_parameters():
            if any(fnmatch.fnmatchcase(name, pattern)
                   or fnmatch.fnmatchcase(name, pattern.rstrip('.') + '.*')
                   for pattern in patterns):
                param.requires_grad_(False)
                frozen.append(name)
    report.frozen_parameters = frozen
    if frozen:
        print(f"transfer: froze {len(frozen)} parameter tensors "
              f"(strategy={strategy}, freeze={cfg.get('freeze') or []})")
    return frozen
