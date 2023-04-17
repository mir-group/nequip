from typing import Optional

import math
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

    # https://github.com/scikit-learn/scikit-learn/blob/d9cfe3f6b1c58dd253dc87cb676ce5171ff1f8a1/sklearn/mixture/_gaussian_mixture.py#L379
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
    n_features = X.size(dim=1)

    # det(precision_chol) = -0.5 * det(precision)
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

    # TODO: understand this
    # log_prob = torch.empty((n_samples, n_components))
    # for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
    #     y = torch.mm(X, prec_chol) - torch.mm(mu.unsqueeze(0), prec_chol)
    #     log_prob[:, k] = torch.sum(torch.square(y), dim=1)

    # ALBY'S CODE
    # X [n_sample, n_feature]
    # prec_chol is [n_component, n_feature, n_feature]
    X_centered = X.unsqueeze(-2) - means.unsqueeze(
        0
    )  # [n_sample, 1, n_feature] - [1, n_component, n_feature] = [n_sample, n_component, n_feature]
    # [n_sample, n_component, n_feature] * [n_component, n_feature, n_feature]
    log_prob = (
        torch.einsum("zci,ij->zcj", X_centered, precisions_chol).square().sum(dim=-1)
    )

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
    return -0.5 * (n_features * math.log(2 * math.pi) + log_prob) + log_det


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
        n_features: int,
        covariance_type: str = "full",
    ):
        super(GaussianMixture, self).__init__()
        assert covariance_type in ("full",)
        self.covariance_type = covariance_type
        self.n_components = n_components
        self.n_features = n_features

        self.register_buffer("means", torch.Tensor())
        self.register_buffer("weights", torch.Tensor())
        self.register_buffer("covariances", torch.Tensor())
        self.register_buffer("precisions_cholesky", torch.Tensor())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the NLL of samples ``X`` under the GMM."""

        # Check if model has been fitted
        assert self.weights.numel() != 0
        assert self.means.numel() != 0
        assert self.covariances.numel() != 0
        assert self.precisions_cholesky.numel() != 0

        # TODO: testing
        estimated_log_probs = _estimate_log_gaussian_prob(
            X, self.means, self.precisions_cholesky, self.covariance_type
        )

        estimated_weights = torch.log(self.weights)

        return torch.logsumexp(estimated_log_probs + estimated_weights, dim=1)

    @torch.jit.unused
    def fit(
        self,
        X: torch.Tensor,
        max_components: int = 50,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        """Fit the GMM to the samples `X` using sklearn."""
        # TODO: if n_components is None, use the BIC; else just use provided n_components
        #       in the BIC case, make sure to set it to an int of the final n_components
        # TODO: fit with sklearn and set this objects buffers from sklearn values
        # Set number of n_components if given, otherwise use the BIC

        n_components = self.n_components
        g = rng if rng else torch.Generator()
        random_state = torch.randint(2**16, (1,), generator=g).item()
        if not n_components:
            components = range(1, max_components)
            gmms = [
                mixture.GaussianMixture(
                    n_components=n, covariance_type="full", random_state=random_state
                )
                for n in components
            ]
            bics = [model.fit(X).bic(X) for model in gmms]
            n_components = np.argmin(bics)

        # Fit GMM
        gmm = mixture.GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=random_state,
        )
        gmm.fit(X)

        # Save info from GMM into the register buffers
        self.register_buffer("means", torch.from_numpy(gmm.means_))
        self.register_buffer("weights", torch.from_numpy(gmm.weights_))
        self.register_buffer("covariances", torch.from_numpy(gmm.covariances_))
        self.register_buffer(
            "precisions_cholesky", torch.from_numpy(gmm.precisions_cholesky_)
        )
