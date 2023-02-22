import torch
import numpy as np
from nequip.utils import gmm
from sklearn import mixture

fit_data = np.random.rand(10, 3)
gmm_sklearn = mixture.GaussianMixture(
    n_components=3, covariance_type="full", random_state=123
)

gmm_torch = gmm.GaussianMixture(n_components=3, seed=123, n_features=3)
# gmm_torch = torch.jit.script(gmm_torch)


class TestGMM:
    def test_fit_forward_simple(self):
        gmm_sklearn.fit(fit_data)
        gmm_torch.fit(torch.from_numpy(fit_data))

        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.means_), gmm_torch.state_dict()["means"]
        )

        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.covariances_),
            gmm_torch.state_dict()["covariances"],
        )

        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.weights_), gmm_torch.state_dict()["weights"]
        )

        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.precisions_cholesky_),
            gmm_torch.state_dict()["precisions_cholesky"],
        )

        test_data = np.random.rand(20, 3)

        sklearn_nll = gmm_sklearn.score_samples(test_data)
        torch_nll = gmm_torch.forward(torch.from_numpy(test_data))

        print(f"sklearn_nll shape: {sklearn_nll.shape}")
        print(f"torch_nll shape: {torch_nll.size()}")

        assert torch.allclose(
            torch.from_numpy(sklearn_nll),
            torch_nll,
        )

    def test_forward(self):

        assert True


# TODO:  test consistancy of GaussianMixture NLLs with pure sklearn solution
