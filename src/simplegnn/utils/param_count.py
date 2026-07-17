"""Per-layer parameter counts for a model built from an experiment config.

Answers "where are the parameters?" without training anything::

    python -m simplegnn.utils.param_count --config experiments/base_paper/regression/ZINC/configs/main_config_ZINC.yml

On the ZINC baseline this shows that the single dense readout after the
aggregation holds ~80% of the model's weights, which is the starting point of
specs/09-zinc-readout-and-architecture-improvements.md. Use it to check a
config against the ~100k parameter budget of the ZINC-subset protocol before
spending GPU time on it.
"""
from pathlib import Path

import click


def parameter_summary(net):
    """
    Per-layer parameter counts of a GraphModel.

    Parameters
    ----------
    net : torch.nn.Module
        A model with a ``net_layers`` ModuleList (i.e. a GraphModel).

    Returns
    -------
    list of dict
        One entry per layer with ``index``, ``name``, ``in_features``,
        ``out_features``, ``out_channels``, ``params`` and ``share`` (the
        fraction of the model's total parameters that the layer holds).
    """
    counts = []
    for index, layer in enumerate(net.net_layers):
        counts.append({
            'index': index,
            'name': getattr(layer, 'name', type(layer).__name__),
            'in_features': getattr(layer, 'in_features', None),
            'out_features': getattr(layer, 'out_features', None),
            'out_channels': getattr(layer, 'out_channels', None),
            'params': sum(p.numel() for p in layer.parameters()),
        })
    total = sum(entry['params'] for entry in counts)
    for entry in counts:
        entry['share'] = entry['params'] / total if total else 0.0
    return counts


def format_parameter_summary(net):
    """Render :func:`parameter_summary` as a table, heaviest layer marked."""
    counts = parameter_summary(net)
    total = sum(entry['params'] for entry in counts)
    lines = [f"{'#':>2}  {'layer':<34} {'in':>7} {'out':>7} {'ch':>5} {'params':>12} {'share':>7}"]
    lines.append('-' * 80)
    heaviest = max(counts, key=lambda entry: entry['params'])['index'] if counts else None
    for entry in counts:
        marker = ' <-' if entry['index'] == heaviest and entry['params'] else ''
        lines.append(
            f"{entry['index']:>2}  {entry['name']:<34} "
            f"{entry['in_features'] if entry['in_features'] is not None else '-':>7} "
            f"{entry['out_features'] if entry['out_features'] is not None else '-':>7} "
            f"{entry['out_channels'] if entry['out_channels'] is not None else '-':>5} "
            f"{entry['params']:>12,} {entry['share'] * 100:>6.1f}%{marker}"
        )
    lines.append('-' * 80)
    lines.append(f"{'':>2}  {'total':<34} {'':>7} {'':>7} {'':>5} {total:>12,}")
    return '\n'.join(lines)


def build_models(main_config_path: Path, device='cpu', seed=0):
    """
    Build (without training) one GraphModel per run configuration of a config.

    Runs the experiment's preprocessing -- the invariant layers need the
    generated labels and properties to know how many weights they have -- and
    then instantiates the model for every point of the hyperparameter grid.

    Parameters
    ----------
    main_config_path : Path
        Path to a main experiment config (the three-tier schema entry point).
    device : str, default 'cpu'
        Device to build the models on.
    seed : int, default 0
        Seed passed to GraphModel.

    Returns
    -------
    list of tuple
        ``(run_config, GraphModel)`` per grid point.
    """
    from simplegnn.framework.core import FrameworkMain, preprocess_graph_data
    from simplegnn.framework.run_configuration import get_run_configs
    from simplegnn.framework.utils.parameters import Parameters
    from simplegnn.framework.utils.preprocessing import load_preprocessed_data_and_parameters
    from simplegnn.models.model import GraphModel

    experiment = FrameworkMain(Path(main_config_path))
    experiment.preprocessing(num_threads=1)

    models = []
    for dataset_key in experiment.network_configurations:
        configuration = experiment.network_configurations[dataset_key][0]
        # counting parameters needs no accelerator, and the experiment configs
        # ask for cuda, which a CPU-only install cannot even move data to
        configuration['device'] = device
        graph_data = preprocess_graph_data(configuration)
        for run_config in get_run_configs(configuration):
            para = Parameters()
            load_preprocessed_data_and_parameters(
                config_id=0,
                run_id=0,
                validation_id=0,
                validation_folds=run_config.config.get('validation_folds', 10),
                graph_data=graph_data,
                run_config=run_config,
                para=para,
            )
            models.append((run_config, GraphModel(graph_data=graph_data, para=para, seed=seed, device=device)))
    return models


@click.command()
@click.option('--config', required=True, help='Path to the main experiment config')
@click.option('--all_configs', is_flag=True, help='Print every grid point instead of only the first')
def main(config, all_configs):
    models = build_models(Path(config))
    if not all_configs:
        models = models[:1]
    for index, (run_config, net) in enumerate(models):
        print(f"\n== run config {index}: lr={run_config.lr}, batch_size={run_config.batch_size}, "
              f"optimizer={run_config.optimizer}, weight_decay={run_config.weight_decay}")
        print(format_parameter_summary(net))


if __name__ == '__main__':
    main()
