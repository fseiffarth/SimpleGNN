"""Speed benchmark: batched vs per-graph ShareGNN forward on NCI1 (CPU and GPU).

Runs **one hyperparameter configuration** (one split, one seed) of the full
production training loop -- ``FrameworkMain.run_configuration`` ->
``ModelConfiguration.train_configuration`` -- once per (device, forward mode)
combination:

    cpu  unbatched | cpu  batched | cuda unbatched | cuda batched

and reports epoch time *and* the resulting train/validation/test accuracies.
The batched forward (``share_gnn_forward: {batched: true}``, see
specs/07-batched-share-gnn-forward.md) must be a pure speed optimization: all
four runs use the same seed, the same batches and the same inputs, so the
accuracy columns are expected to agree (up to floating-point summation order,
which differs between the block-diagonal sparse matmul and the per-graph
matmuls, and between CPU and GPU kernels).

Run it (the ``speed`` marker is excluded from ``pytest tests`` by default)::

    pytest tests/test_speed_share_gnn_nci1.py -m speed -s

The GPU rows need a CUDA/ROCm build of torch; on the Radeon iGPU setup the
venv has no pytest, so the module doubles as a standalone script::

    HSA_OVERRIDE_GFX_VERSION=11.0.0 venv-rocm/bin/python \
        tests/test_speed_share_gnn_nci1.py --device cuda

    python tests/test_speed_share_gnn_nci1.py --device cpu --device cuda \
        --epochs 10 --batch-size 64
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "share_gnn_nci1"
DATASET = "NCI1"
SPLITS = ROOT / "src/simplegnn/datasets/splits/standard/NCI1_splits.json"
DATA = ROOT / "data" / "TUDatasets"

# accuracy tolerance (percentage points) between the batched and the unbatched
# run of the same configuration -- they train the identical model on identical
# batches, only the summation order inside the invariant layers differs
ACCURACY_TOLERANCE = 1.0


def _write_configs(work_dir: Path, device: str, batched: bool, epochs: int,
                   batch_size: int, precision: str) -> Path:
    """Write a main.yml + parameters.yml for one (device, mode) run."""
    parameters = yaml.safe_load((FIXTURES / "parameters.yml").read_text())
    parameters["device"] = device
    parameters["precision"] = precision
    parameters["epochs"] = [epochs]
    parameters["batch_size"] = [batch_size]
    parameters["share_gnn_forward"] = {"batched": bool(batched)}
    parameters_path = work_dir / "parameters.yml"
    parameters_path.write_text(yaml.safe_dump(parameters))

    main_config = {
        "datasets": [
            {
                "name": DATASET,
                "source": "TUDataset",
                "task": "graph_classification",
                "paths": {
                    # data/labels/properties are shared so the (expensive)
                    # ShareGNN preprocessing is computed once and cached
                    "data": str(DATA),
                    "labels": str(DATA / "labels"),
                    "properties": str(DATA / "properties"),
                    "results": str(work_dir / "results"),
                    "models": str(FIXTURES / "models_ShareGNN.yml"),
                    "hyperparameters": str(parameters_path),
                    "splits": str(SPLITS),
                },
            }
        ]
    }
    main_path = work_dir / "main.yml"
    main_path.write_text(yaml.safe_dump(main_config))
    return main_path


def _read_epoch_rows(results_dir: Path) -> list[dict]:
    csv_files = list(results_dir.glob("*_Results_run_id_*.csv"))
    assert len(csv_files) == 1, f"expected exactly one results csv, got {csv_files}"
    with open(csv_files[0], newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def run_once(device: str, batched: bool, epochs: int, batch_size: int,
             precision: str = "double") -> dict:
    """Train one configuration on one split and return timings + accuracies."""
    from simplegnn.framework.core import FrameworkMain, preprocess_graph_data
    from simplegnn.framework.run_configuration import get_run_configs

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        main_path = _write_configs(work_dir, device, batched, epochs, batch_size, precision)

        experiment = FrameworkMain(main_path)
        experiment.preprocessing(num_threads=1)
        configuration = experiment.network_configurations[DATASET][0]
        graph_data = preprocess_graph_data(configuration)
        run_config = get_run_configs(configuration)[0]

        start = time.perf_counter()
        experiment.run_configuration(graph_data, run_config,
                                     validation_id=0, run_id=0, config_id=0)
        wall_time = time.perf_counter() - start

        rows = _read_epoch_rows(work_dir / "results" / DATASET / "Results")

    epoch_times = [float(row["EpochTime"]) for row in rows]
    last = rows[-1]
    return {
        "device": device,
        "mode": "batched" if batched else "unbatched",
        "num_graphs": len(graph_data),
        "epochs": len(rows),
        "wall_time": wall_time,
        "epoch_times": epoch_times,
        # first epoch pays lazy allocation / kernel compilation, so it is warmup
        "epoch_time_mean": sum(epoch_times[1:]) / max(1, len(epoch_times) - 1),
        "first_epoch_loss": float(rows[0][_loss_key(rows[0])]),
        "final_epoch_loss": float(last[_loss_key(last)]),
        "train_accuracy": float(last["EpochAccuracy"]),
        "validation_accuracy": float(last["ValidationAccuracy"]),
        # the standard NCI1 splits have empty test lists (10-fold CV over
        # train/validation only), so there is usually no test accuracy
        "test_size": int(last["TestSize"]),
        "test_accuracy": float(last["TestAccuracy"]),
    }


def _loss_key(row: dict) -> str:
    """The epoch-loss column name embeds the loss function, e.g.
    ``EpochLoss  (CrossEntropyLoss)``."""
    for key in row:
        if key.startswith("EpochLoss"):
            return key
    raise KeyError(f"no epoch loss column in {list(row)}")


def cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _accuracy_keys(results: list[dict]) -> tuple[str, ...]:
    keys = ["train_accuracy", "validation_accuracy"]
    if any(r["test_size"] for r in results):
        keys.append("test_accuracy")
    return tuple(keys)


def benchmark(devices: list[str], epochs: int, batch_size: int,
              precision: str = "double") -> list[dict]:
    results = []
    for device in devices:
        for batched in (False, True):
            result = run_once(device, batched, epochs, batch_size, precision)
            results.append(result)
            print(f"  {device:>4} {result['mode']:>9}: "
                  f"{result['epoch_time_mean']:6.2f} s/epoch  "
                  f"train {result['train_accuracy']:5.2f}  "
                  f"val {result['validation_accuracy']:5.2f}", flush=True)
    return results


def format_table(results: list[dict]) -> str:
    with_test = "test_accuracy" in _accuracy_keys(results)
    header = (f"{'device':<8}{'mode':<11}{'s/epoch':>9}{'speedup':>9}"
              f"{'train acc':>11}{'val acc':>9}")
    if with_test:
        header += f"{'test acc':>10}"
    header += f"{'final loss':>12}"
    lines = [
        "",
        f"=== {DATASET}: batched vs per-graph ShareGNN forward "
        f"({results[0]['num_graphs']} graphs, {results[0]['epochs']} epochs, one split) ===",
        "",
        header,
        "-" * len(header),
    ]
    baseline = {r["device"]: r["epoch_time_mean"] for r in results if r["mode"] == "unbatched"}
    for r in results:
        speedup = baseline.get(r["device"], float("nan")) / r["epoch_time_mean"]
        line = (f"{r['device']:<8}{r['mode']:<11}{r['epoch_time_mean']:9.2f}{speedup:8.2f}x"
                f"{r['train_accuracy']:11.2f}{r['validation_accuracy']:9.2f}")
        if with_test:
            line += f"{r['test_accuracy']:10.2f}"
        line += f"{r['final_epoch_loss']:12.4f}"
        lines.append(line)
    if not with_test:
        lines.append("")
        lines.append("(no test column: the standard NCI1 splits define train/validation folds only)")
    return "\n".join(lines) + "\n"


def accuracy_deltas(results: list[dict]) -> dict:
    """Max |batched - unbatched| accuracy difference per device (percentage points)."""
    keys = _accuracy_keys(results)
    deltas = {}
    for device in {r["device"] for r in results}:
        by_mode = {r["mode"]: r for r in results if r["device"] == device}
        if len(by_mode) != 2:
            continue
        deltas[device] = max(
            abs(by_mode["batched"][key] - by_mode["unbatched"][key]) for key in keys
        )
    return deltas


# --------------------------------------------------------------------------- #
# pytest entry point
# --------------------------------------------------------------------------- #

try:
    import pytest
except ImportError:  # standalone script mode (e.g. the ROCm venv has no pytest)
    pytest = None

if pytest is not None:

    @pytest.mark.speed
    @pytest.mark.integration
    def test_batched_speed_and_accuracy_nci1(capsys):
        """Benchmark all available devices; the batched forward must not change
        the accuracies of the configuration it speeds up."""
        devices = ["cpu"] + (["cuda"] if cuda_available() else [])
        results = benchmark(devices, epochs=10, batch_size=64)

        with capsys.disabled():
            print(format_table(results))
            for device, delta in accuracy_deltas(results).items():
                print(f"max |batched - unbatched| accuracy difference on {device}: "
                      f"{delta:.4f} percentage points")
            if "cuda" not in devices:
                print("cuda rows skipped: torch.cuda.is_available() is False")

        for device, delta in accuracy_deltas(results).items():
            assert delta <= ACCURACY_TOLERANCE, (
                f"batched and unbatched accuracies disagree on {device} by "
                f"{delta:.4f} percentage points"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="append", choices=["cpu", "cuda"],
                        help="device to benchmark (repeatable; default: cpu + cuda if available)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--precision", default="double", choices=["double", "float"])
    args = parser.parse_args()

    devices = args.device or (["cpu"] + (["cuda"] if cuda_available() else []))
    if "cuda" in devices and not cuda_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")

    results = benchmark(devices, args.epochs, args.batch_size, args.precision)
    print(format_table(results))
    for device, delta in accuracy_deltas(results).items():
        print(f"max |batched - unbatched| accuracy difference on {device}: "
              f"{delta:.4f} percentage points")


if __name__ == "__main__":
    main()
