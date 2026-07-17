"""
ShareGNN node classification on the Cora citation network.

Trains a ShareGNN (invariant-based message passing + linear readout) on the
single Cora graph. The train/validation/test splits contain *node* indices and
are generated from the standard Planetoid masks (140 train / 500 validation /
1000 test nodes).
"""
import json
from pathlib import Path

from simplegnn.framework.core import FrameworkMain


def create_split_file_from_planetoid_masks(split_file: Path, name: str = 'Cora'):
    """Write the framework's split JSON from the standard Planetoid masks."""
    if split_file.exists():
        return
    import torch
    from torch_geometric.datasets import Planetoid
    data = Planetoid(root='tmp/', name=name)[0]
    splits = [{
        'test': torch.where(data.test_mask)[0].tolist(),
        'model_selection': [{
            'train': torch.where(data.train_mask)[0].tolist(),
            'validation': torch.where(data.val_mask)[0].tolist(),
        }],
    }]
    split_file.write_text(json.dumps(splits))
    print(f'Created split file {split_file}')


def main():
    create_split_file_from_planetoid_masks(Path('examples/node_classification/Cora_splits.json'))

    experiment = FrameworkMain(Path('examples/node_classification/main.yml'))
    # dataset download, label/property generation
    experiment.preprocessing(num_threads=1)
    # train all hyperparameter configurations
    experiment.run_configurations(num_threads=1)
    # find the best configuration on the validation set
    experiment.evaluate_results()
    # retrain the best configuration and evaluate it on the test set
    experiment.run_best_configuration(num_threads=1)
    experiment.evaluate_results(evaluate_best_model=True)


if __name__ == '__main__':
    main()
