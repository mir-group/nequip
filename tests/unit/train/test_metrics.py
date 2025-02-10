import torch
from nequip.train import HuberLoss, StratifiedHuberForceLoss

import pytest


LARGE_FORCE_MAGNITUDE_SCALING = 200
EXTREMELY_LARGE_FORCE_MAGNITUDE_MASK = 15000


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


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize(
    "delta_dict",
    [
        {0: 0.01, 100: 0.007, 200: 0.004, 300: 0.001},
        {0: 0.01, 150: 0.005},
        {0: 0.01, EXTREMELY_LARGE_FORCE_MAGNITUDE_MASK: 1e-7},
        {150: 0.005},
    ],
)
def test_strat_huber(reduction, delta_dict):
    """
    Tests nequip's custom StratifiedHuberForceLoss
    """
    nequip_strat_huber = StratifiedHuberForceLoss(
        reduction=reduction, delta_dict=delta_dict
    )
    # test limiting behaviour:
    nequip_min_huber = HuberLoss(
        reduction=reduction, delta=min(nequip_strat_huber.delta_dict.values())
    )
    nequip_max_huber = HuberLoss(
        reduction=reduction, delta=max(nequip_strat_huber.delta_dict.values())
    )
    torch_hubers = {
        lower_bound: torch.nn.HuberLoss(
            reduction="none", delta=delta
        )  # reduction to applied afterward
        for lower_bound, delta in nequip_strat_huber.delta_dict.items()
    }

    shape = (27, 23, 3)
    xs = []
    ys = []
    for _ in range(10):
        # scale by a large force magnitude -> mean = 0, variance = 200^2
        datax = torch.randn(shape, dtype=torch.float64) * LARGE_FORCE_MAGNITUDE_SCALING
        datay = torch.randn(shape, dtype=torch.float64) * LARGE_FORCE_MAGNITUDE_SCALING
        nequip_strat_huber.update(datax, datay)
        nequip_min_huber.update(datax, datay)
        nequip_max_huber.update(datax, datay)
        xs.append(datax)
        ys.append(datay)

    bounds = list(nequip_strat_huber.delta_dict.keys())
    x_vals = torch.cat(xs, 0)
    y_vals = torch.cat(ys, 0)

    stratified_losses = torch.zeros_like(y_vals)
    for lower_bound, torch_huber in torch_hubers.items():
        next_lower_bound_idx = bounds.index(lower_bound) + 1
        mask = (torch.norm(y_vals, dim=-1) >= lower_bound) & (
            torch.norm(y_vals, dim=-1) < bounds[next_lower_bound_idx]
            if next_lower_bound_idx < len(bounds)
            else True
        )
        stratified_losses[mask] = torch_huber(x_vals[mask], y_vals[mask])

    torch_huber_val = (
        stratified_losses.mean() if reduction == "mean" else stratified_losses.sum()
    )

    assert torch.allclose(nequip_strat_huber.compute(), torch_huber_val)
    assert (
        nequip_min_huber.compute()
        <= nequip_strat_huber.compute()
        <= nequip_max_huber.compute()
    )

    if EXTREMELY_LARGE_FORCE_MAGNITUDE_MASK in nequip_strat_huber.delta_dict:
        # then reduces to basically just normal Huber behaviour, large force magnitude mask in last
        # stratum isn't being applied:
        assert torch.allclose(nequip_strat_huber.compute(), nequip_max_huber.compute())
