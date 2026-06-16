"""Migrate the 69 ablation-threshold main configs to the three-tier schema.

For each {lower, lower_upper, upper}/main_config_ablation_threshold_<N>.yml
(N in 1..20, 30, 40, 50) this:

  1. Extracts the rule-occurrence thresholds (rule_occurrence_threshold and/or
     rule_occurrence_upper_threshold) from the *existing* config. If the config
     has already been migrated (thresholds moved into the parameters file), the
     values are recovered from the sibling parameters file instead, so the script
     is idempotent.
  2. Writes a per-threshold parameters yml next to the main config.
  3. Overwrites the main config in place with the new schema: 6 TU datasets,
     fair splits, per-dataset paths, pointing at the 6 shared model configs and
     the per-threshold parameters file.

The naming (.../threshold/<type>/main_config_ablation_threshold_<N>.yml) is
preserved because experiments_ablation_threshold.py constructs these paths.

Run from the repo root:  python experiments/base_paper/tools/migrate_threshold_configs.py
"""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
THRESH_DIR = REPO / "experiments/base_paper/classification/configs/ablation/threshold"
CFG_REL = "experiments/base_paper/classification/configs/ablation/threshold"
SPLITS_REL = "src/simplegnn/datasets/splits/fair"
DATA_REL = "data/TUDatasets"

TYPES = ["lower", "lower_upper", "upper"]
TYPE_TO_RESULTS = {"lower": "Lower", "lower_upper": "LowerUpper", "upper": "Upper"}
THRESHOLDS = list(range(1, 21)) + [30, 40, 50]

# dataset name -> shared model config (note: model files use underscores)
DATASETS = [
    ("IMDB-BINARY", "config_ablation_threshold_IMDB_BINARY.yml"),
    ("NCI1", "config_ablation_threshold_NCI1.yml"),
    ("IMDB-MULTI", "config_ablation_threshold_IMDB_MULTI.yml"),
    ("NCI109", "config_ablation_threshold_NCI109.yml"),
    ("Mutagenicity", "config_ablation_threshold_Mutagenicity.yml"),
    ("DHFR", "config_ablation_threshold_DHFR.yml"),
]

WEIGHT_INIT = (
    "weight_initialization: { convolution: { type: 'constant', value: 0.001 },\n"
    "                         convolution_bias: { type: 'constant', value: 0.0 },\n"
    "                         aggregation: { type: 'constant', value: 0.001 },\n"
    "                         aggregation_bias: { type: 'constant', value: 0.0 } }"
)


def extract_thresholds(main_path: Path, params_path: Path):
    """Recover (rule_occurrence_threshold, rule_occurrence_upper_threshold)."""
    data = yaml.safe_load(main_path.read_text()) if main_path.is_file() else {}
    rot = data.get("rule_occurrence_threshold")
    rout = data.get("rule_occurrence_upper_threshold")
    if rot is None and rout is None and params_path.is_file():
        p = yaml.safe_load(params_path.read_text())
        rot = p.get("rule_occurrence_threshold")
        rout = p.get("rule_occurrence_upper_threshold")
    return rot, rout


def write_parameters(params_path: Path, rot, rout):
    lines = [
        f"# Ablation (threshold) hyperparameters for {params_path.parent.name}/{params_path.stem.split('_')[-1]}",
        "# (migrated from old monolithic main config).",
        "device: cpu",
        "mode: experiments",
        "precision: double",
        "",
        "optimizer:",
        "  - Adam",
        "",
        "loss:",
        "  - CrossEntropyLoss",
        "",
        "convolution_grad: True",
        "aggregation_grad: True",
        "",
    ]
    if rot is not None:
        lines.append(f"rule_occurrence_threshold: {rot}")
    if rout is not None:
        lines.append(f"rule_occurrence_upper_threshold: {rout}")
    lines += [
        "",
        WEIGHT_INIT,
        "",
        "batch_size:",
        "  - 64",
        "learning_rate:",
        "  - 0.01",
        "epochs:",
        "  - 200",
        "",
        "early_stopping:",
        "  enabled: True",
        "  patience: 25",
        "",
        "num_workers: 30",
        "input_features: { name: constant, value: 1.0 }",
        "",
    ]
    params_path.write_text("\n".join(lines))


def write_main(main_path: Path, ttype: str, threshold: int, params_name: str):
    results = f"results/base_paper/classification/Ablation/Threshold/{TYPE_TO_RESULTS[ttype]}/{threshold}/"
    out = [f"# Ablation (threshold) experiment {ttype}/{threshold} (migrated to three-tier schema).",
           "datasets:"]
    for name, model in DATASETS:
        out.append(
            f'  - {{ name: "{name}", source: "TUDataset", task: "graph_classification", validation_folds: 10,\n'
            f'      paths: {{ data: "{DATA_REL}/", labels: "{DATA_REL}/labels/", properties: "{DATA_REL}/properties/",\n'
            f'               results: "{results}", models: "{CFG_REL}/{model}",\n'
            f'               hyperparameters: "{CFG_REL}/{ttype}/{params_name}", splits: "{SPLITS_REL}/{name}_splits.json" }} }}'
        )
    main_path.write_text("\n".join(out) + "\n")


def main():
    count = 0
    for ttype in TYPES:
        for t in THRESHOLDS:
            main_path = THRESH_DIR / ttype / f"main_config_ablation_threshold_{t}.yml"
            params_name = f"parameters_ablation_threshold_{t}.yml"
            params_path = THRESH_DIR / ttype / params_name
            rot, rout = extract_thresholds(main_path, params_path)
            if rot is None and rout is None:
                raise SystemExit(f"Could not determine thresholds for {main_path}")
            write_parameters(params_path, rot, rout)
            write_main(main_path, ttype, t, params_name)
            count += 1
    print(f"Migrated {count} threshold configs (+ {count} parameters files).")


if __name__ == "__main__":
    main()
