from typing import Optional

import torch
import numpy as np
from sklearn.mixture import GaussianMixture


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
        super(self).__init__()
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
            "precisions", torch.zeros(n_components, n_features, n_features)
        )
        self.register_buffer(
            "precisions_cholesky", torch.zeros(n_components, n_features, n_features)
        )
        # TODO: other parameters as buffers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the NLL of samples ``x`` under the GMM."""
        # TODO: implement
        return x

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
                GaussianMixture(
                    n_components=n, covariance_type="full", random_state=27617
                )
                for n in components
            ]
            bics = [model.fit(x).bic(x) for model in gmms]
            n_components = bics.index(min(bics))

        # Fit GMM
        random_state = self.seed if self.seed else 123
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=random_state,
        )
        gmm.fit(x)

        # Save info from GMM into the register buffers
        self.register_buffer("means", torch.from_numpy(gmm.means_))
        self.register_buffer("weights", torch.from_numpy(gmm.weights_))
        self.register_buffer("covariances", torch.from_numpy(gmm.covariances_))
        self.register_buffer("precisions", torch.from_numpy(gmm.precisions_))
        self.register_buffer(
            "precisions_cholesky", torch.from_numpy(gmm.precisions_cholesky)
        )
