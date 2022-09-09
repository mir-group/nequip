import torch
import pytest

from nequip.utils.regressor import solver


# @pytest.mark.parametrize("full_rank", [True, False])
@pytest.mark.parametrize("full_rank", [False])
@pytest.mark.parametrize("alpha", [0, 1e-3, 0.1, 1])
def test_random(full_rank, alpha):

    if alpha == 0 and not full_rank:
        return

    torch.manual_seed(0)
    n_samples = 10
    n_dim = 3

    if full_rank:
        X = torch.randint(low=1, high=10, size=(n_samples, n_dim))
    else:
        X = torch.randint(low=1, high=10, size=(n_samples, 1)) * torch.ones(
            (n_samples, n_dim)
        )

    ref_mean = torch.rand((n_dim, 1))
    y = torch.matmul(X, ref_mean)

    mean, std = solver(
        X, y, alpha=0.1
    )

    if full_rank:
        assert torch.allclose(ref_mean, mean, rtol=0.5)
    else:
        assert torch.allclose(mean, mean[0], rtol=1e-3)
