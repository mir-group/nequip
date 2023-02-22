from typing import Optional

import torch
import numpy as np
from sklearn import mixture


@torch.jit.script
def _compute_log_det_cholesky(
    matrix_chol: torch.Tensor, covariance_type: str, n_features: int
):
    """Compute the log-det of the cholesky decomposition of matrices."""

    assert covariance_type in ("full",)
    n_components = matrix_chol.size(dim=0)
    log_det_chol = torch.sum(
        torch.log(matrix_chol.view(n_components, -1)[:, :: n_features + 1]), dim=1
    )

    return log_det_chol


@torch.jit.script
def _estimate_log_gaussian_prob(
    X: torch.Tensor,
    means: torch.Tensor,
    precisions_chol: torch.Tensor,
    covariance_type: str,
):
    """Estimate the log Gaussian probability."""

    assert covariance_type in ("full",)
    n_samples, n_features = X.size(dim=0), X.size(dim=1)
    n_components = means.size(dim=0)

    # det(precision_chol) = -0.5 * det(precision)
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

    # TODO: understand this
    log_prob = torch.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        y = torch.mm(X, prec_chol) - torch.mm(mu.unsqueeze(0), prec_chol)
        log_prob[:, k] = torch.sum(torch.square(y), dim=1)
    # X = X.unsqueeze(0)
    # means = means.unsqueeze(2)
    # print(f"X shape: {X.size()}")
    # print(f"means shape: {means.size()}")
    # log_prob = torch.sum(
    #     torch.square(
    #         torch.matmul(X, precisions_chol.transpose(1, 2))
    #         - torch.matmul(means, precisions_chol.transpose(1, 2))
    #     ),
    #     dim=1,
    # )

    # Since we are using the precision of the Cholesky decomposition,
    # `-0.5 * log_det` becomes `+ log_det`
    return -0.5 * (n_features * torch.log(torch.tensor(2 * np.pi)) + log_prob) + log_det


class GaussianMixture(torch.nn.Module):
    """Calculate NLL of samples under a Gaussian Mixture Model (GMM).

    Supports fitting the GMM outside of PyTorch using `sklearn`.
    """

    covariance_type: str
    n_components: int
    n_features: int
    seed: int

    def __init__(
        self,
        n_components: Optional[int],
        seed: Optional[int],
        n_features: int,
        covariance_type: str = "full",
    ):
        super(GaussianMixture, self).__init__()
        assert covariance_type in ("full",)
        self.covariance_type = covariance_type
        self.n_components = n_components
        self.n_features = n_features
        self.seed = seed

        self.register_buffer("means", torch.zeros(n_components, n_features))
        self.register_buffer("weights", torch.zeros(n_components))
        self.register_buffer(
            "covariances", torch.zeros(n_components, n_features, n_features)
        )
        self.register_buffer(
            "precisions_cholesky", torch.zeros(n_components, n_features, n_features)
        )
        # TODO: other parameters as buffers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the NLL of samples ``x`` under the GMM."""
        # TODO: implement

        # TODO: check if model has been fitted

        # TODO: testing
        estimated_log_probs = _estimate_log_gaussian_prob(
            x, self.means, self.precisions_cholesky, self.covariance_type
        )

        estimated_weights = np.log(self.weights)

        return torch.logsumexp(estimated_log_probs + estimated_weights, dim=1)

    @torch.jit.unused
    def fit(self, x: torch.Tensor) -> None:
        """Fit the GMM to the samples `x` using sklearn."""
        # TODO: if n_components is None, use the BIC; else just use provided n_components
        #       in the BIC case, make sure to set it to an int of the final n_components
        # TODO: fit with sklearn and set this objects buffers from sklearn values
        # Set number of n_components if given, otherwise use the BIC

        n_components = self.n_components
        if not n_components:
            tmp_heurstic = 30  # TODO: determine how to set max number of components for BIC testing
            components = np.arange(1, tmp_heurstic)
            gmms = [
                mixture.GaussianMixture(
                    n_components=n, covariance_type="full", random_state=27617
                )
                for n in components
            ]
            bics = [model.fit(x).bic(x) for model in gmms]
            n_components = bics.index(min(bics))

        # Fit GMM
        random_state = self.seed if self.seed else 123
        gmm = mixture.GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=random_state,
        )
        gmm.fit(x)

        # Save info from GMM into the register buffers
        self.register_buffer("means", torch.from_numpy(gmm.means_))
        self.register_buffer("weights", torch.from_numpy(gmm.weights_))
        self.register_buffer("covariances", torch.from_numpy(gmm.covariances_))
        self.register_buffer(
            "precisions_cholesky", torch.from_numpy(gmm.precisions_cholesky_)
        )

        # TODO: replace above code with self.means[:] = gmm.means_ ?
