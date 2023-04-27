import torch
import pytest
import numpy as np
from nequip.utils import gmm
from sklearn import mixture
from e3nn.util.test import assert_auto_jitable


class TestGMM:
    # Seed for tests
    @pytest.fixture
    def seed(self):
        return 678912345

    # Data sets for fitting GMMs and scoring NLLs
    @pytest.fixture(params=[[10, 8], [500, 32]])
    def feature_data(self, seed, request):
        fit_data = 2 * (
            torch.randn(
                request.param[0],
                request.param[1],
                generator=torch.Generator().manual_seed(seed),
            )
            - 0.5
        )
        test_data = 2 * (
            torch.randn(
                request.param[0] * 2,
                request.param[1],
                generator=torch.Generator().manual_seed(seed - 123456789),
            )
            - 0.5
        )
        return {"fit_data": fit_data, "test_data": test_data}

    # Sklearn GMM for tests
    @pytest.fixture
    def gmm_sklearn(self, seed):
        return mixture.GaussianMixture(
            n_components=8, covariance_type="full", random_state=seed
        )

    # Torch GMM for small data set tests
    @pytest.fixture
    def gmm_torch(self, feature_data):
        return gmm.GaussianMixture(
            n_features=feature_data["fit_data"].size(dim=1), n_components=8
        )

    # Test compilation
    def test_compile(self, gmm_torch):
        assert_auto_jitable(gmm_torch)

    # Test agreement between sklearn and torch GMMs
    def test_fit_forward(self, seed, gmm_sklearn, gmm_torch, feature_data):
        gmm_sklearn.fit(feature_data["fit_data"].numpy())
        gmm_torch.fit(feature_data["fit_data"], rng=seed)

        assert torch.allclose(torch.from_numpy(gmm_sklearn.means_), gmm_torch.means)
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.covariances_), gmm_torch.covariances
        )
        assert torch.allclose(torch.from_numpy(gmm_sklearn.weights_), gmm_torch.weights)
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.precisions_cholesky_),
            gmm_torch.precisions_cholesky,
        )

        sklearn_nll = gmm_sklearn.score_samples(feature_data["test_data"].numpy())
        torch_nll = gmm_torch(feature_data["test_data"])

        assert torch.allclose(-torch.from_numpy(sklearn_nll), torch_nll)

    # Test agreement between sklearn and torch using BIC
    def test_fit_forward_bic(self, seed, feature_data):
        components = range(1, min(50, feature_data["fit_data"].size(dim=0)))
        gmms = [
            mixture.GaussianMixture(
                n_components=n, covariance_type="full", random_state=seed
            )
            for n in components
        ]
        bics = [
            model.fit(feature_data["fit_data"]).bic(feature_data["fit_data"])
            for model in gmms
        ]

        gmm_sklearn = mixture.GaussianMixture(
            n_components=np.argmin(bics), covariance_type="full", random_state=seed
        )
        gmm_torch = gmm.GaussianMixture(n_features=feature_data["fit_data"].size(dim=1))

        gmm_sklearn.fit(feature_data["fit_data"].numpy())
        gmm_torch.fit(feature_data["fit_data"], rng=seed)

        assert torch.allclose(torch.from_numpy(gmm_sklearn.means_), gmm_torch.means)
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.covariances_), gmm_torch.covariances
        )
        assert torch.allclose(torch.from_numpy(gmm_sklearn.weights_), gmm_torch.weights)
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.precisions_cholesky_),
            gmm_torch.precisions_cholesky,
        )

        sklearn_nll = gmm_sklearn.score_samples(feature_data["test_data"].numpy())
        torch_nll = gmm_torch(feature_data["test_data"])

        assert torch.allclose(-torch.from_numpy(sklearn_nll), torch_nll)

    # Test assertion error for covariance type other than "full"
    def test_full_cov(self):
        with pytest.raises(AssertionError) as excinfo:
            _ = gmm.GaussianMixture(n_features=2, covariance_type="tied")
        assert "covariance type was tied, should be full" in str(excinfo.value)

    # Test assertion error for evaluating unfitted GMM
    def test_unfitted_gmm(self, gmm_torch, feature_data):
        with pytest.raises(AssertionError) as excinfo:
            _ = gmm_torch(feature_data["test_data"])
        assert "model has not been fitted" in str(excinfo.value)
