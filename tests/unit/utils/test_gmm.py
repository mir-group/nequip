import torch
import numpy as np
from nequip.utils import gmm
from sklearn import mixture

rng = np.random.RandomState(678912345)
fit_data = rng.rand(10, 3)
gmm_sklearn = mixture.GaussianMixture(
    n_components=3, covariance_type="full", random_state=rng.rand(1)
)

gmm_torch = gmm.GaussianMixture(n_components=3, n_features=3)
gmm_torch = torch.jit.script(gmm_torch)


class TestGMM:
    def test_fit_forward_simple(self):
        gmm_sklearn.fit(fit_data)
        gmm_torch.fit(torch.from_numpy(fit_data))

        assert torch.allclose(torch.from_numpy(gmm_sklearn.means_), gmm_torch.means)

        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.covariances_), gmm_torch.covariances
        )

        assert torch.allclose(torch.from_numpy(gmm_sklearn.weights_), gmm_torch.weights)

        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.precisions_cholesky_),
            gmm_torch.precisions_cholesky,
        )

        test_data = rng.rand(20, 3)

        sklearn_nll = gmm_sklearn.score_samples(test_data)
        torch_nll = gmm_torch(torch.from_numpy(test_data))

        print(f"sklearn_nll shape: {sklearn_nll.shape}")
        print(f"torch_nll shape: {torch_nll.size()}")

        assert torch.allclose(
            torch.from_numpy(sklearn_nll),
            torch_nll,
        )

    def test_forward(self):

        assert True


# TODO:  test consistancy of GaussianMixture NLLs with pure sklearn solution
