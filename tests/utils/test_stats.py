from typing import Optional

import pytest

import random

import torch
from torch_scatter import scatter

from nequip.utils.stats import RunningStats, Reduction


class StatsTruth:
    """Inefficient ground truth for RunningStats."""

    def __init__(self, dim=1, reduction: Reduction = Reduction.MEAN):
        self._dim = dim
        self._reduction = reduction
        self.reset()

    def accumulate_batch(
        self, batch: torch.Tensor, accumulate_by: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if accumulate_by is None:
            accumulate_by = torch.zeros(len(batch), dtype=torch.long)
        if hasattr(self, "_state"):
            self._state = torch.cat((self._state, batch), dim=0)
            self._acc = torch.cat((self._acc, accumulate_by), dim=0)
        else:
            self._state = batch.clone()
            self._acc = accumulate_by.clone()
        if self._reduction == Reduction.MEAN:
            return batch.mean(dim=0)
        elif self._reduction == Reduction.RMS:
            return batch.square().mean(dim=0).sqrt()

    def reset(self) -> None:
        if hasattr(self, "_state"):
            delattr(self, "_state")
            delattr(self, "_acc")

    def current_result(self):
        if not hasattr(self, "_state"):
            return torch.zeros(self._dim)
        if torch.max(self._acc) > 0:
            if self._reduction == Reduction.MEAN:
                return scatter(self._state, dim=0, reduce="mean")
            elif self._reduction == Reduction.RMS:
                return scatter(torch.square(self._state), dim=0, reduce="mean").sqrt()
        else:
            if self._reduction == Reduction.MEAN:
                return self._state.mean(dim=0)
            elif self._reduction == Reduction.RMS:
                return self._state.square().mean(dim=0).sqrt()


@pytest.mark.parametrize("dim", [1, 3, (2, 3), torch.Size((1, 2, 1))])
@pytest.mark.parametrize("reduction", [Reduction.MEAN, Reduction.RMS])
@pytest.mark.parametrize("do_accumulate_by", [True, False])
def test_runstats(dim, reduction, do_accumulate_by):
    n_batchs = (random.randint(1, 4), random.randint(1, 4))
    truth_obj = StatsTruth(dim=dim, reduction=reduction)
    runstats = RunningStats(dim=dim, reduction=reduction)

    for n_batch in n_batchs:
        for _ in range(n_batch):
            batch = torch.randn((random.randint(1, 10),) + runstats.dim)
            if do_accumulate_by and random.choice((True, False)):
                accumulate_by = torch.randint(
                    0, random.randint(1, 5), size=(batch.shape[0],)
                )
                print("batch", batch.shape, accumulate_by)
                truth = truth_obj.accumulate_batch(batch, accumulate_by=accumulate_by)
                res = runstats.accumulate_batch(batch, accumulate_by=accumulate_by)
            else:
                truth = truth_obj.accumulate_batch(batch)
                res = runstats.accumulate_batch(batch)
            print(dim, reduction, truth, res)
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
