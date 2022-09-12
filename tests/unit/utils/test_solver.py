import torch
import pytest

from nequip.utils.regressor import solver


@pytest.mark.parametrize("full_rank", [True, False])
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

    ref_mean, ref_std, y = generate_E(X, 100.0, 1000.0, 20.0)

    mean, std = solver(X, y, alpha=alpha)

    if full_rank:
        assert torch.allclose(ref_mean, mean, atol=(2*torch.max(ref_std)))
    else:
        assert torch.allclose(mean, mean[0], rtol=1e-3)

def generate_E(X, mean_min, mean_max, std):
    torch.manual_seed(0)
    ref_mean = torch.rand((X.shape[1])) * (mean_max - mean_min) + mean_min
    t_mean = torch.ones((X.shape[0], 1)) * ref_mean.reshape([1, -1])
    ref_std = torch.rand((X.shape[1])) * std
    t_std = torch.ones((X.shape[0], 1)) * ref_std.reshape([1, -1])
    E = torch.normal(t_mean, t_std)
    return ref_mean, ref_std, (X * E).sum(axis=-1)
