"""Cross-dataset label/pair overlap measurement (spec 18 B0) — the go/no-go
gate for hash-keyed transfer.

Compares the canonical hash vocabularies of two independently preprocessed
datasets and prints, per label type, the unique-label overlap and the
occurrence-weighted coverage of the target by the source. With property files
it additionally reports the coverage of the target's
(src_hash, tgt_hash, property_value) triples — the quantity that upper-bounds
what apply_transfer can transfer for a conv head.

Both datasets must have been preprocessed with spec 18 Part A in place (v2
label files with hash vocabularies); delete the labels directory and rerun
preprocessing() otherwise.

Usage (after preprocessing both datasets):

    python examples/transfer_learning/measure_overlap.py \
        --source-labels data/TUDatasets/labels/NCI1 \
        --target-labels data/TUDatasets/labels/DHFR

    python examples/transfer_learning/measure_overlap.py \
        --source-labels data/TUDatasets/labels/NCI1 \
        --target-labels data/TUDatasets/labels/DHFR \
        --label-types wl_2 wl_3 closed_walks_2_8 induced_cycles_5_6 \
        --source-properties data/TUDatasets/properties/NCI1/NCI1_properties_distances.pt \
        --target-properties data/TUDatasets/properties/DHFR/DHFR_properties_distances.pt

Decision point (spec 18 B0): if the weighted coverage is single-digit percent
for every usable (canonical) labeling, hash-keyed transfer cannot help — stop.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))

from simplegnn.datasets.utils.node_labeling import load_labels  # noqa: E402
from simplegnn.framework.utils.transfer import (  # noqa: E402
    load_property_pairs, measure_label_overlap, measure_pair_overlap)


def dataset_name_of(labels_dir: Path) -> str:
    return labels_dir.name


def discover_label_types(labels_dir: Path, dataset_name: str) -> set:
    prefix = f'{dataset_name}_labels_'
    return {f.name[len(prefix):-3] for f in labels_dir.glob(f'{prefix}*.pt')}


def label_file(labels_dir: Path, dataset_name: str, label_type: str) -> Path:
    return labels_dir / f'{dataset_name}_labels_{label_type}.pt'


def print_table(header, rows):
    widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
    line = ' | '.join(str(h).ljust(w) for h, w in zip(header, widths))
    print(line)
    print('-+-'.join('-' * w for w in widths))
    for row in rows:
        print(' | '.join(str(v).ljust(w) for v, w in zip(row, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-labels', type=Path, required=True,
                        help="labels directory of the source dataset, e.g. data/TUDatasets/labels/NCI1")
    parser.add_argument('--target-labels', type=Path, required=True,
                        help="labels directory of the target dataset")
    parser.add_argument('--source-name', default=None,
                        help="source dataset name (default: labels directory name)")
    parser.add_argument('--target-name', default=None,
                        help="target dataset name (default: labels directory name)")
    parser.add_argument('--label-types', nargs='*', default=None,
                        help="label types to compare (default: every type present in both datasets)")
    parser.add_argument('--source-properties', type=Path, default=None,
                        help="source property file (<db>_properties_<name>.pt) for pair overlap")
    parser.add_argument('--target-properties', type=Path, default=None,
                        help="target property file for pair overlap")
    parser.add_argument('--property-values', nargs='*', type=int, default=None,
                        help="property values to consider (default: values present in both)")
    args = parser.parse_args(argv)

    source_name = args.source_name or dataset_name_of(args.source_labels)
    target_name = args.target_name or dataset_name_of(args.target_labels)

    if args.label_types:
        label_types = list(args.label_types)
    else:
        label_types = sorted(discover_label_types(args.source_labels, source_name)
                             & discover_label_types(args.target_labels, target_name))
        if not label_types:
            print(f"No label types found in both {args.source_labels} and {args.target_labels} "
                  f"— run preprocessing() for both datasets first.")
            return 1

    print(f"\nLabel-vocabulary overlap {source_name} -> {target_name}\n")
    rows = []
    reports = {}
    for label_type in label_types:
        source_file = label_file(args.source_labels, source_name, label_type)
        target_file = label_file(args.target_labels, target_name, label_type)
        if not source_file.exists() or not target_file.exists():
            rows.append([label_type, '-', '-', '-', '-', '-', 'missing label file'])
            continue
        try:
            report = measure_label_overlap(source_file, target_file)
        except ValueError as e:
            rows.append([label_type, '-', '-', '-', '-', '-', str(e)])
            continue
        reports[label_type] = report
        canonical = 'yes' if (report.source_canonical and report.target_canonical) else 'NO'
        rows.append([label_type, report.source_unique, report.target_unique,
                     report.shared_unique, f'{report.unique_overlap:.1%}',
                     f'{report.weighted_coverage:.1%}', canonical])
    print_table(['label type', 'src unique', 'tgt unique', 'shared',
                 'unique overlap', 'weighted coverage', 'canonical'], rows)
    print("\n(unique overlap = shared / target unique; weighted coverage = fraction of the "
          "target's node occurrences whose hash exists in the source; non-canonical "
          "labelings are dataset-relative and excluded from transfer by default)")

    if args.source_properties and args.target_properties:
        print(f"\n(src_hash, tgt_hash, property_value) triple coverage "
              f"{source_name} -> {target_name} via {args.target_properties.name}\n")
        source_props = load_property_pairs(args.source_properties)
        target_props = load_property_pairs(args.target_properties)
        rows = []
        for label_type in label_types:
            source_file = label_file(args.source_labels, source_name, label_type)
            target_file = label_file(args.target_labels, target_name, label_type)
            if not source_file.exists() or not target_file.exists():
                continue
            try:
                report = measure_pair_overlap(load_labels(path=source_file), source_props,
                                              load_labels(path=target_file), target_props,
                                              property_values=args.property_values)
            except ValueError as e:
                rows.append([label_type, '-', '-', '-', '-', str(e)])
                continue
            rows.append([label_type, report.target_unique, report.shared_unique,
                         f'{report.unique_overlap:.1%}', f'{report.weighted_coverage:.1%}',
                         'yes' if (report.source_canonical and report.target_canonical) else 'NO'])
        print_table(['label type', 'tgt unique triples', 'shared', 'unique overlap',
                     'weighted coverage', 'canonical'], rows)
        print("\n(weighted coverage upper-bounds the occurrence mass a conv head using this "
              "labeling + property can transfer — spec 18 B0 decision point)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
