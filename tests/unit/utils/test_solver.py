import torch
import pytest

from nequip.utils.regressor import solver


@pytest.mark.parametrize("full_rank", [True, False])
@pytest.mark.parametrize("alpha", [0, 1e-3, 0.1, 1])
def test_random(full_rank, alpha, per_species_set):

    if alpha == 0 and not full_rank:
        return

    torch.manual_seed(0)

    ref_mean, ref_std, E, n_samples, n_dim = per_species_set

    dtype = torch.get_default_dtype()

    X = torch.randint(low=1, high=10, size=(n_samples, n_dim)).to(dtype)
    if not full_rank:
        X[:, n_dim - 2] = X[:, n_dim - 1]
    y = (X * E).sum(axis=-1)

    mean, std = solver(X, y, alpha=alpha)

    tolerance = 2 * torch.max(ref_std)

    if full_rank:
        assert torch.allclose(ref_mean, mean, atol=tolerance)
    else:
        assert torch.allclose(mean[n_dim - 1], mean[n_dim - 2], rtol=1e-3)

    assert torch.max(std) < tolerance
