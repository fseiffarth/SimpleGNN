from __future__ import annotations

from pathlib import Path

import torch

from simplegnn.framework.utils.parameters import Parameters


def test_set_data_param_populates_fields(tmp_path: Path):
    p = Parameters()
    run_config = type(
        "RunConfig",
        (),
        {
            "config": {
                "paths": {"data": tmp_path / "d", "results": tmp_path / "r", "splits": tmp_path / "s"},
                "batch_size": 8,
            }
        },
    )()

    p.set_data_param("MUTAG", max_coding=3, layers=["L"], node_features=7, run_config=run_config)

    assert p.db == "MUTAG"
    assert p.node_features == 7
    assert p.batch_size == 8


def test_set_print_param_no_print_overrides_flags():
    p = Parameters()
    p.set_print_param(
        no_print=True,
        print_results=True,
        net_print_weights=True,
        print_number=5,
        draw=True,
        save_weights=True,
        save_prediction_values=True,
        plot_graphs=True,
        print_layer_init=True,
    )

    assert p.print_results is False
    assert p.draw is False
    assert p.print_number == 0


def test_set_file_index_picks_next_highest_index(tmp_path: Path):
    (tmp_path / "results_000003_summary.txt").write_text("x")
    (tmp_path / "results_000010_summary.txt").write_text("x")

    p = Parameters()
    p.results_path = str(tmp_path)
    p.set_file_index(size=6)

    assert p.new_file_index == "000011"


def test_save_predictions_writes_expected_format(tmp_path: Path):
    p = Parameters()
    p.results_path = str(tmp_path) + "/"
    p.save_file_name = "preds.txt"

    p.save_predictions(torch.tensor([0.1234, 0.9876]), torch.tensor([1.0, 0.0]))

    content = (tmp_path / "preds.txt").read_text()
    assert "Labels" in content
    assert "Prediction" in content
