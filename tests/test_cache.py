"""Preprocessing is idempotent and safe to re-run (label/property caching).

FrameworkMain.preprocessing() generates node labels and edge properties and
writes them under data/TUDatasets/{labels,properties}/. Running it a second
time must reuse those artifacts rather than regenerate or corrupt them, since
every experiment re-runs preprocessing before training.

The original version of this file timed two runs and printed a speedup against
an example config that no longer exists; it asserted nothing. This version runs
preprocessing twice against the lightweight ShareGNN test fixture and asserts
the generated artifacts are present and unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "share_gnn_mutag"
LABELS_DIR = ROOT / "data" / "TUDatasets" / "labels" / "MUTAG"


@pytest.mark.integration
def test_preprocessing_is_idempotent(mutag_main_config):
    from simplegnn.framework.core import FrameworkMain

    main_config_path = mutag_main_config(
        models=FIXTURES / "models_ShareGNN.yml",
        hyperparameters=FIXTURES / "parameters.yml",
    )

    FrameworkMain(main_config_path).preprocessing(num_threads=1)

    generated = sorted(p.name for p in LABELS_DIR.glob("*.pt"))
    assert generated, "preprocessing should have generated node label files"
    fingerprint = {p.name: p.stat().st_size for p in LABELS_DIR.glob("*.pt")}

    # A second run must reuse the existing artifacts, not regenerate or drop them.
    FrameworkMain(main_config_path).preprocessing(num_threads=1)

    assert sorted(p.name for p in LABELS_DIR.glob("*.pt")) == generated
    assert {p.name: p.stat().st_size for p in LABELS_DIR.glob("*.pt")} == fingerprint
