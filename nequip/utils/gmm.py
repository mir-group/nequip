from typing import Optional

import torch


class GaussianMixture(torch.nn.Module):
    """Calculate NLL of samples under a Gaussian Mixture Model (GMM).

    Supports fitting the GMM outside of PyTorch using `sklearn`.
    """

    covariance_type: str
    n_components: int
    n_features: int

    def __init__(
        self,
        n_components: Optional[int],
        n_features: int,
        covariance_type: str = "full",
    ):
        super(self).__init__()
        assert covariance_type in ("full",)
        self.covariance_type = covariance_type
        self.n_components = n_components
        self.n_features = n_features

        self.register_buffer("means", torch.zeros(n_components, n_features))
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
        pass
