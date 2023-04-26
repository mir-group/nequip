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

    # Small data set for fitting GMMs in tests
    @pytest.fixture
    def fit_data_small(self, seed):
        return 2 * (
            torch.randn(10, 8, generator=torch.Generator().manual_seed(seed)) - 0.5
        )

    # Large data set for fitting GMMs in tests
    @pytest.fixture
    def fit_data_large(self, seed):
        return 20 * (
            torch.randn(500, 32, generator=torch.Generator().manual_seed(seed)) - 0.5
        )

    # Small data set for scoring NLLs in test
    @pytest.fixture
    def test_data_small(self, seed):
        return 2 * (
            torch.randn(
                100, 8, generator=torch.Generator().manual_seed(seed - 123456789)
            )
            - 0.5
        )

    # Sklearn GMM for tests
    @pytest.fixture
    def gmm_sklearn(self, seed):
        return mixture.GaussianMixture(
            n_components=8, covariance_type="full", random_state=seed
        )

    # Torch GMM for small data set tests
    @pytest.fixture
    def gmm_torch_small(self, fit_data_small):
        return gmm.GaussianMixture(
            n_features=fit_data_small.size(dim=1), n_components=8
        )

    # Torch GMM for large data set tests
    @pytest.fixture
    def gmm_torch_large(self, fit_data_large):
        return gmm.GaussianMixture(
            n_features=fit_data_large.size(dim=1), n_components=8
        )

    # Test compilation
    def test_compile(self, gmm_torch_small):
        assert_auto_jitable(gmm_torch_small)

    # Test agreement between sklearn and torch using small dataset
    def test_fit_forward_small(
        self, seed, gmm_sklearn, gmm_torch_small, fit_data_small, test_data_small
    ):
        gmm_sklearn.fit(fit_data_small.numpy())
        gmm_torch_small.fit(fit_data_small, rng=seed)

        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.means_), gmm_torch_small.means
        )
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.covariances_), gmm_torch_small.covariances
        )
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.weights_), gmm_torch_small.weights
        )
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.precisions_cholesky_),
            gmm_torch_small.precisions_cholesky,
        )

        sklearn_nll = gmm_sklearn.score_samples(test_data_small.numpy())
        torch_nll = gmm_torch_small(test_data_small)

        assert torch.allclose(-torch.from_numpy(sklearn_nll), torch_nll)

    # Test agreement between sklearn and torch using large dataset
    def test_fit_forward_large(
        self, seed, gmm_sklearn, gmm_torch_large, fit_data_large
    ):
        gmm_sklearn.fit(fit_data_large.numpy())
        gmm_torch_large.fit(fit_data_large, rng=seed)

        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.means_), gmm_torch_large.means
        )
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.covariances_), gmm_torch_large.covariances
        )
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.weights_), gmm_torch_large.weights
        )
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.precisions_cholesky_),
            gmm_torch_large.precisions_cholesky,
        )

        test_data_large = 20 * (
            torch.randn(
                500, 32, generator=torch.Generator().manual_seed(seed - 123456789)
            )
            - 0.5
        )

        sklearn_nll = gmm_sklearn.score_samples(test_data_large.numpy())
        torch_nll = gmm_torch_large(test_data_large)

        assert torch.allclose(-torch.from_numpy(sklearn_nll), torch_nll)

    # Test agreement between sklearn and torch using small dataset and BIC
    def test_fit_forward_bic(self, seed, fit_data_small, test_data_small):
        components = range(1, min(50, fit_data_small.size(dim=0)))
        gmms = [
            mixture.GaussianMixture(
                n_components=n, covariance_type="full", random_state=seed
            )
            for n in components
        ]
        bics = [model.fit(fit_data_small).bic(fit_data_small) for model in gmms]

        gmm_sklearn = mixture.GaussianMixture(
            n_components=np.argmin(bics), covariance_type="full", random_state=seed
        )
        gmm_torch = gmm.GaussianMixture(n_features=fit_data_small.size(dim=1))

        gmm_sklearn.fit(fit_data_small.numpy())
        gmm_torch.fit(fit_data_small, rng=seed)

        assert torch.allclose(torch.from_numpy(gmm_sklearn.means_), gmm_torch.means)
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.covariances_), gmm_torch.covariances
        )
        assert torch.allclose(torch.from_numpy(gmm_sklearn.weights_), gmm_torch.weights)
        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.precisions_cholesky_),
            gmm_torch.precisions_cholesky,
        )

        sklearn_nll = gmm_sklearn.score_samples(test_data_small.numpy())
        torch_nll = gmm_torch(test_data_small)

        assert torch.allclose(-torch.from_numpy(sklearn_nll), torch_nll)

    # Test assertion error for covariance type other than "full"
    def test_full_cov(self):
        with pytest.raises(AssertionError) as excinfo:
            _ = gmm.GaussianMixture(n_features=2, covariance_type="tied")
        assert "covariance type was tied, should be full" in str(excinfo.value)

    # Test assertion error for evaluating unfitted GMM
    def test_unfitted_gmm(self, gmm_torch_small, test_data_small):
        with pytest.raises(AssertionError) as excinfo:
            _ = gmm_torch_small(test_data_small)
        assert "model has not been fitted" in str(excinfo.value)
