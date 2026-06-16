from __future__ import annotations

import numpy as np
import torch

from simplegnn.framework.utils.data_sampling import curriculum_sampling, no_curriculum_sampling


class StubGraphData:
    def __init__(self):
        self.slices = {
            "x": torch.tensor([0, 2, 5, 9, 10], dtype=torch.long),
            "edge_index": torch.tensor([0, 4, 10, 16, 20], dtype=torch.long),
        }
        self.y = torch.tensor([0, 1, 0, 1], dtype=torch.long)


def test_no_curriculum_sampling_shape_and_domain():
    training_data = np.array([0, 1, 2])
    out = no_curriculum_sampling(training_data, num_batches=3, batch_size=4)

    assert out.shape == (3, 4)
    assert set(np.unique(out)).issubset({0, 1, 2})


def test_curriculum_sampling_exclusive_and_nonexclusive_shapes():
    gd = StubGraphData()
    training_data = np.array([0, 1, 2, 3])

    out_exclusive = curriculum_sampling(gd, training_data, bucket_num=2, num_batches=2, batch_size=3, total_epochs=10, epoch=1, exclusive=True)
    out_nonexclusive = curriculum_sampling(gd, training_data, bucket_num=2, num_batches=2, batch_size=3, total_epochs=10, epoch=1, exclusive=False)

    assert out_exclusive.shape == (2, 3)
    assert out_nonexclusive.shape == (2, 3)


def test_curriculum_sampling_anti_and_use_edges_paths():
    gd = StubGraphData()
    training_data = np.array([0, 1, 2, 3])

    out = curriculum_sampling(
        gd,
        training_data,
        bucket_num=2,
        num_batches=1,
        batch_size=2,
        total_epochs=10,
        epoch=5,
        anti=True,
        use_edges=True,
    )

    assert out.shape == (1, 2)
    assert set(np.unique(out)).issubset({0, 1, 2, 3})
