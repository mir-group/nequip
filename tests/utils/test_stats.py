import pytest

import random

import torch

from nequip.utils.stats import RunningStats, Reduction


class StatsTruth:
    """Inefficient ground truth for RunningStats."""

    def __init__(self, dim=1, reduction: Reduction = Reduction.MEAN):
        self._dim = dim
        self._reduction = reduction
        self.reset()

    def accumulate_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "_state"):
            self._state = torch.cat((self._state, batch), dim=0)
        else:
            self._state = batch.clone()
        if self._reduction == Reduction.MEAN:
            return batch.mean(dim=0)
        elif self._reduction == Reduction.RMS:
            return batch.square().mean(dim=0).sqrt()

    def reset(self) -> None:
        if hasattr(self, "_state"):
            delattr(self, "_state")

    def current_result(self):
        if not hasattr(self, "_state"):
            return torch.zeros(self._dim)
        if self._reduction == Reduction.MEAN:
            return self._state.mean(dim=0)
        elif self._reduction == Reduction.RMS:
            return self._state.square().mean(dim=0).sqrt()


@pytest.mark.parametrize("dim", [1, 3, (2, 3), torch.Size((1, 2, 1))])
@pytest.mark.parametrize("reduction", [Reduction.MEAN, Reduction.RMS])
def test_runstats(dim, reduction):
    n_batchs = (random.randint(1, 4), random.randint(1, 4))
    truth_obj = StatsTruth(dim=dim, reduction=reduction)
    runstats = RunningStats(dim=dim, reduction=reduction)

    for n_batch in n_batchs:
        for _ in range(n_batch):
            batch = torch.randn((random.randint(1, 10),) + runstats.dim)
            truth = truth_obj.accumulate_batch(batch)
            res = runstats.accumulate_batch(batch)
            assert torch.allclose(truth, res)
        assert torch.allclose(truth_obj.current_result(), runstats.current_result())
        truth_obj.reset()
        runstats.reset()


@pytest.mark.parametrize("reduction", [Reduction.MEAN, Reduction.RMS])
def test_zeros(reduction):
    dim = (4,)
    runstats = RunningStats(dim=dim, reduction=reduction)
    assert torch.allclose(runstats.current_result(), torch.zeros(dim))
    runstats.accumulate_batch(torch.randn((3,) + dim))
    runstats.reset()
    assert torch.allclose(runstats.current_result(), torch.zeros(dim))


def test_raises():
    runstats = RunningStats(dim=4, reduction=Reduction.MEAN)
    with pytest.raises(AssertionError):
        runstats.accumulate_batch(torch.zeros(10, 2))
