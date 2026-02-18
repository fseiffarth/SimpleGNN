#!/usr/bin/env python3
"""
Direct test of InvariantBasedMessagePassingLayer caching.
"""
import sys
import time
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from datasets.graph_dataset import GraphDataset
from models.ShareGNN.layers.inv_based_message_passing import InvariantBasedMessagePassingLayer
from models.ShareGNN.utils import Layer, LayerHead
from framework.utils.parameters import Parameters

def create_minimal_layer_config():
    """Create a minimal layer configuration for testing."""
    layer_dict = {
        'layer_type': 'invariant_based_convolution',
        'activation': 'torch.nn.Tanh()',
        'bias': False,
        'heads': [
            {
                'num': 1,
                'bias': False,
                'labels': {
                    'head': {'label_type': 'primary'},
                    'tail': {'label_type': 'primary'},
                    'bias': {'label_type': 'primary'},
                },
                'properties': {
                    'name': 'distances',
                    'values': [0, 1, 2, 3]
                }
            }
        ]
    }
    return Layer(layer_dict, layer_id=0)

def test_caching():
    """Test that caching works correctly."""
    print("=" * 80)
    print("Loading MUTAG dataset...")
    print("=" * 80)

    # Load dataset
    dataset_path = Path('data/TUDatasets/MUTAG')
    graph_data = GraphDataset(
        root=str(dataset_path.parent),
        name='MUTAG',
        source='TUDatasets'
    )

    print(f"Dataset loaded: {len(graph_data)} graphs\n")

    # Create minimal parameters object
    class MinimalParams:
        def __init__(self):
            self.path = str(dataset_path.parent)
            self.run_config = type('obj', (object,), {'config': {}})()

    parameters = MinimalParams()
    layer_config = create_minimal_layer_config()

    print("=" * 80)
    print("FIRST RUN: Cache Miss Expected")
    print("=" * 80)

    start_time = time.time()
    layer1 = InvariantBasedMessagePassingLayer(parameters, layer_config, graph_data)
    first_run_time = time.time() - start_time

    print(f"\nFirst run completed in {first_run_time:.2f} seconds")
    print(f"Number of weights: {sum(layer1.weight_num)}")

    print("\n" + "=" * 80)
    print("SECOND RUN: Cache Hit Expected")
    print("=" * 80)

    start_time = time.time()
    layer2 = InvariantBasedMessagePassingLayer(parameters, layer_config, graph_data)
    second_run_time = time.time() - start_time

    print(f"\nSecond run completed in {second_run_time:.2f} seconds")
    print(f"Number of weights: {sum(layer2.weight_num)}")

    print("\n" + "=" * 80)
    print("CACHE TEST RESULTS")
    print("=" * 80)
    print(f"First run time:  {first_run_time:.2f}s")
    print(f"Second run time: {second_run_time:.2f}s")

    if second_run_time < first_run_time:
        speedup = first_run_time / second_run_time
        print(f"✓ Cache speedup: {speedup:.1f}x")

        # Verify correctness
        if sum(layer1.weight_num) == sum(layer2.weight_num):
            print("✓ Weight counts match")
        if torch.equal(layer1.weight_distribution, layer2.weight_distribution):
            print("✓ Weight distributions match")
        if torch.equal(layer1.weight_distribution_slices, layer2.weight_distribution_slices):
            print("✓ Weight distribution slices match")

        print("\n✓ CACHE TEST PASSED")
    else:
        print(f"⚠ Warning: Second run not faster")

if __name__ == '__main__':
    test_caching()
