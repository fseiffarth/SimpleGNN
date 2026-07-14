"""Benchmark: batched vs per-graph ShareGNN forward on MUTAG.

Compares the classical per-graph ShareGNN training loop against the batched
forward (specs/07-batched-share-gnn-forward.md), where all graphs of a batch
are concatenated and processed jointly (one block-diagonal sparse matmul per
invariant layer, one device transfer per batch).

Usage:
    python examples/share_gnn_basic/benchmark_batched.py --device cpu
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python examples/share_gnn_basic/benchmark_batched.py --device cuda

Both modes run the same batches from the same seed, so their losses must agree
(up to floating point summation order); the script checks this.
"""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from simplegnn.framework.core import FrameworkMain, preprocess_graph_data
from simplegnn.framework.run_configuration import get_run_configs
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.framework.utils.preprocessing import load_preprocessed_data_and_parameters
from simplegnn.models.model import GraphModel


def setup(main_yml: str):
    experiment = FrameworkMain(Path(main_yml))
    experiment.preprocessing(num_threads=1)
    dataset_key = next(iter(experiment.network_configurations))
    configuration = experiment.network_configurations[dataset_key][0]
    graph_data = preprocess_graph_data(configuration)
    run_config = get_run_configs(configuration)[0]
    para = Parameters()
    load_preprocessed_data_and_parameters(
        config_id=0, run_id=0, validation_id=0,
        validation_folds=run_config.config.get("validation_folds", 10),
        graph_data=graph_data, run_config=run_config, para=para)
    # the benchmark drives its own loop; disable input noise for comparability
    para.run_config.config.get('input_features', {}).pop('random_variation', None)
    return graph_data, para


def assemble_batch(graph_data, positions, device):
    """All graphs of the batch are concatenated and moved to the device together."""
    slices = graph_data.slices['x']
    x = torch.cat([graph_data.x[int(slices[p]):int(slices[p + 1])] for p in positions])
    if x.device != device:
        x = x.to(device)
    return SimpleNamespace(x=x)


def run_training(graph_data, para, device, batched: bool, epochs: int, warmup: int,
                 batch_size: int, seed: int = 42, lr: float = 0.01):
    net = GraphModel(graph_data=graph_data, para=para, seed=seed, device=device)
    net.to(device)
    net.random_variation_bool = None
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    ids = np.arange(len(graph_data))
    times, losses = [], []
    for epoch in range(epochs + warmup):
        order = np.random.RandomState(1000 + epoch).permutation(ids)
        batches = np.array_split(order, max(1, len(order) // batch_size))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_loss = 0.0
        for batch_ids in batches:
            optimizer.zero_grad(set_to_none=True)
            labels = graph_data.y[batch_ids]
            if labels.device != device:
                labels = labels.to(device)
            if batched:
                positions = [int(g) for g in batch_ids]
                outputs = net(assemble_batch(graph_data, positions, device), pos=positions)
            else:
                outputs = torch.stack([net(graph_data[int(g)], pos=int(g)) for g in batch_ids])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if epoch >= warmup:
            times.append(elapsed)
            losses.append(total_loss / len(ids))
    return np.array(times), np.array(losses), net


def run_inference(net, graph_data, device, batched: bool, repeats: int = 5):
    net.eval()
    ids = list(range(len(graph_data)))
    times = []
    with torch.no_grad():
        for _ in range(repeats + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if batched:
                outputs = net(assemble_batch(graph_data, ids, device), pos=ids)
            else:
                outputs = torch.stack([net(graph_data[g], pos=g) for g in ids])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return np.array(times[1:]), outputs  # drop warmup repeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='examples/share_gnn_basic/main.yml')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA/ROCm device requested but torch.cuda.is_available() is False')

    graph_data, para = setup(args.config)
    print(f"\n=== Benchmark: MUTAG, {len(graph_data)} graphs, device={device}, "
          f"dtype={'double' if para.run_config.config.get('precision') == 'double' else 'float'}, "
          f"batch_size={args.batch_size}, epochs={args.epochs} (+{args.warmup} warmup) ===")
    # the framework moves the collated dataset to the execution device up front
    graph_data.to(device)

    results = {}
    for mode, batched in [('unbatched', False), ('batched', True)]:
        train_times, losses, net = run_training(
            graph_data, para, device, batched=batched,
            epochs=args.epochs, warmup=args.warmup, batch_size=args.batch_size)
        infer_times, outputs = run_inference(net, graph_data, device, batched=batched)
        results[mode] = (train_times, losses, infer_times, outputs)
        print(f"\n[{mode}]")
        print(f"  train epoch: {train_times.mean()*1000:8.1f} ms  ± {train_times.std()*1000:.1f} ms")
        print(f"  inference (all {len(graph_data)} graphs): {infer_times.mean()*1000:8.1f} ms  ± {infer_times.std()*1000:.1f} ms")
        print(f"  final epoch loss: {losses[-1]:.6f}")

    u_train, u_losses, u_infer, u_out = results['unbatched']
    b_train, b_losses, b_infer, b_out = results['batched']
    print(f"\n=== Speedup (batched vs unbatched) ===")
    print(f"  training:  {u_train.mean() / b_train.mean():5.2f}x")
    print(f"  inference: {u_infer.mean() / b_infer.mean():5.2f}x")
    max_loss_diff = np.abs(u_losses - b_losses).max()
    max_out_diff = (u_out - b_out).abs().max().item()
    print(f"  max epoch-loss difference:   {max_loss_diff:.2e}")
    print(f"  max final-output difference: {max_out_diff:.2e}")
    if max_loss_diff > 1e-4:
        print("  WARNING: losses diverged beyond floating-point noise!")
    else:
        print("  losses agree: both modes compute the same training trajectory")


if __name__ == '__main__':
    main()
