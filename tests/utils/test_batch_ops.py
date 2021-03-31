import pytest

import torch

from nequip.utils import batch_ops


@pytest.mark.parametrize("n_class", [1, 5])
@pytest.mark.parametrize("n_batch", [1, 10])
@pytest.mark.parametrize("n_max_nodes", [1, 4])
def test_bincount(n_class, n_batch, n_max_nodes):
    n_nodes = torch.randint(1, n_max_nodes + 1, size=(n_batch,))
    total_n_nodes = n_nodes.sum()
    input = torch.randint(0, n_class, size=(total_n_nodes,))
    batch = torch.LongTensor(sum(([i] * n for i, n in enumerate(n_nodes)), start=[]))

    truth = []
    for b in range(n_batch):
        truth.append(torch.bincount(input[batch == b], minlength=n_class))
    truth = torch.stack(truth)

    res = batch_ops.bincount(input, batch, minlength=n_class)

    assert torch.all(res == truth)
