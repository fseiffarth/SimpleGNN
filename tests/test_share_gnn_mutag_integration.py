"""End-to-end smoke tests: ShareGNN on real MUTAG data.

Unlike the unit tests in this suite, these exercise the real pipeline against
the real MUTAG dataset (downloaded via TUDataset on first run, cached under
data/TUDatasets/ afterwards). They are the project's sanity check that "the
whole thing still works", not tests of any single component.

Two things are covered:

1. The full FrameworkMain pipeline runs and writes the expected results.
2. The ShareGNN invariant-based layers actually learn -- their shared weight
   vectors receive gradients and are changed by an optimizer step. These
   layers index into a shared Param_W via a precomputed weight distribution
   rather than using a plain nn.Linear, so a bug in that indexing can silently
   detach them from the graph and leave the weights frozen while training
   still appears to "work".
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "share_gnn_mutag"
MODELS = FIXTURES / "models_ShareGNN.yml"
PARAMETERS = FIXTURES / "parameters.yml"


def _invariant_layers(net):
    from simplegnn.models.ShareGNN.layers.inv_based_message_passing import (
        InvariantBasedMessagePassingLayer,
    )
    from simplegnn.models.ShareGNN.layers.inv_based_pooling import (
        InvariantBasedAggregationLayer,
    )

    invariant_types = (InvariantBasedMessagePassingLayer, InvariantBasedAggregationLayer)
    return [
        (name, module)
        for name, module in net.named_modules()
        if isinstance(module, invariant_types)
    ]


@pytest.mark.integration
def test_share_gnn_trains_and_evaluates_on_mutag(tmp_path, mutag_main_config):
    from simplegnn.framework.core import FrameworkMain

    main_config_path = mutag_main_config(models=MODELS, hyperparameters=PARAMETERS)

    experiment = FrameworkMain(main_config_path)
    experiment.preprocessing(num_threads=1)
    experiment.run_configurations(num_threads=-1)
    experiment.evaluate_results()
    experiment.run_best_configuration(num_threads=-1)
    experiment.evaluate_results(evaluate_best_model=True)

    summary_path = tmp_path / "results" / "MUTAG" / "summary_best_mean.csv"
    assert summary_path.is_file(), "expected summary_best_mean.csv to be written"

    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    validation_accuracy = float(rows[0]["Validation Accuracy Mean"])
    assert 0.0 <= validation_accuracy <= 100.0

    models_dir = tmp_path / "results" / "MUTAG" / "Models"
    assert any(models_dir.glob("*.pt")), "expected at least one saved model checkpoint"


@pytest.mark.integration
def test_invariant_layer_weights_are_updated_by_training(mutag_main_config):
    """A training step must actually move the invariant layers' shared weights."""
    from simplegnn.framework.core import FrameworkMain, preprocess_graph_data
    from simplegnn.framework.run_configuration import get_run_configs
    from simplegnn.framework.utils.parameters import Parameters
    from simplegnn.framework.utils.preprocessing import (
        load_preprocessed_data_and_parameters,
    )
    from simplegnn.models.model import GraphModel

    main_config_path = mutag_main_config(models=MODELS, hyperparameters=PARAMETERS)

    experiment = FrameworkMain(main_config_path)
    experiment.preprocessing(num_threads=1)

    dataset_key = next(iter(experiment.network_configurations))
    configuration = experiment.network_configurations[dataset_key][0]
    graph_data = preprocess_graph_data(configuration)
    run_config = get_run_configs(configuration)[0]

    para = Parameters()
    load_preprocessed_data_and_parameters(
        config_id=0,
        run_id=0,
        validation_id=0,
        validation_folds=run_config.config.get("validation_folds", 10),
        graph_data=graph_data,
        run_config=run_config,
        para=para,
    )

    net = GraphModel(graph_data=graph_data, para=para, seed=42, device="cpu")

    invariant_layers = _invariant_layers(net)
    assert invariant_layers, "fixture model should contain invariant-based layers"

    weights_before = {
        name: module.Param_W.detach().clone() for name, module in invariant_layers
    }

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    graph_ids = list(range(16))
    optimizer.zero_grad()
    outputs = torch.stack([net(graph_data[i], pos=i) for i in graph_ids])
    labels = torch.stack([graph_data[i].y for i in graph_ids]).squeeze()
    loss = criterion(outputs, labels)
    loss.backward()

    for name, module in invariant_layers:
        gradient = module.Param_W.grad
        assert gradient is not None, f"{name}.Param_W got no gradient"
        assert torch.any(gradient != 0), f"{name}.Param_W gradient is all zeros"

    optimizer.step()

    for name, module in invariant_layers:
        assert not torch.equal(module.Param_W.detach(), weights_before[name]), (
            f"{name}.Param_W did not change after an optimizer step"
        )
