"""End-to-end smoke tests: every classical MPNN architecture on real MUTAG data.

One test per architecture registered in LayerTypes (GCN, GAT, GATv2, GIN,
SAGE), each running the full FrameworkMain pipeline against real MUTAG. This
catches breakage in a layer wrapper (wrong PyG argument, shape mismatch after
pooling, ...) that unit tests over synthetic tensors would miss.

See tests/test_share_gnn_mutag_integration.py for the ShareGNN equivalent.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "classical_gnn_mutag"
PARAMETERS = FIXTURES / "parameters.yml"

ARCHITECTURES = ["GCN", "GAT", "GATv2", "GIN", "SAGE"]


@pytest.mark.integration
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_classical_gnn_trains_and_evaluates_on_mutag(
    architecture, tmp_path, mutag_main_config
):
    from simplegnn.framework.core import FrameworkMain

    main_config_path = mutag_main_config(
        models=FIXTURES / f"models_{architecture}.yml",
        hyperparameters=PARAMETERS,
    )

    experiment = FrameworkMain(main_config_path)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=-1)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=-1)
    experiment.evaluate_results(evaluate_best_model=True)

    summary_path = tmp_path / "results" / "MUTAG" / "summary_best_mean.csv"
    assert summary_path.is_file(), (
        f"{architecture}: expected summary_best_mean.csv to be written"
    )

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    validation_accuracy = float(rows[0]["Validation Accuracy Mean"])
    assert 0.0 <= validation_accuracy <= 100.0

    models_dir = tmp_path / "results" / "MUTAG" / "Models"
    assert any(models_dir.glob("*.pt")), (
        f"{architecture}: expected at least one saved model checkpoint"
    )
