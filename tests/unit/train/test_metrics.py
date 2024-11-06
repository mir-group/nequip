import torch
from nequip.train import HuberLoss

import pytest


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("delta", [0.5, 1.2, 3.4])
def test_huber(reduction, delta):
    """
    Tests that nequip's custom HuberLoss (using torchmetrics-style accumulation) agrees with torch.nn.HuberLoss.
    """
    torch_huber = torch.nn.HuberLoss(reduction=reduction, delta=delta)
    nequip_huber = HuberLoss(reduction=reduction, delta=delta)

    shape = (27, 23)
    xs = []
    ys = []
    for _ in range(10):
        datax = torch.randn(shape, dtype=torch.float64)
        datay = torch.randn(shape, dtype=torch.float64)
        nequip_huber.update(datax, datay)
        xs.append(datax)
        ys.append(datay)
    assert torch.allclose(
        nequip_huber.compute(), torch_huber(torch.cat(xs, 0), torch.cat(ys, 0))
    )
