from typing import Optional

import torch

from e3nn import o3

from nequip.data import AtomicDataDict


from ._graph_mixin import GraphModuleMixin
from nequip.utils.gmm import GaussianMixture


class GaussianMixtureModelUncertainty(GraphModuleMixin, torch.nn.Module):
    """Compute GMM NLL uncertainties based on some input featurization.

    Args:
        gmm_n_components (int or None): if None, use the BIC to determine the number of components.
    """

    feature_field: str
    out_field: str

    def __init__(
        self,
        feature_field: str,
        out_field: str,
        gmm_n_components: Optional[int] = None,
        gmm_covariance_type: str = "full",
        irreps_in=None,
    ):
        super().__init__()
        self.feature_field = feature_field
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[feature_field],
            irreps_out={out_field: "0e"},
        )
        feature_irreps = self.irreps_in[self.feature_field].simplify()
        if not (len(feature_irreps) == 1 and feature_irreps[0].ir == o3.Irrep("0e")):
            raise ValueError(
                f"GaussianMixtureModelUncertainty feature_field={feature_field} must be only scalars, instead got {feature_irreps}"
            )
        # GaussianMixture already correctly registers things as parameters,
        # so they will get saved & loaded in state dicts
        self.gmm = GaussianMixture(
            n_components=gmm_n_components,
            n_features=feature_irreps.num_irreps,
            covariance_type=gmm_covariance_type,
        )

    @torch.jit.unused
    def fit(self, X, seed=None) -> None:
        self.gmm.fit(X, rng=seed)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.gmm.is_fit():
            nll_scores = self.gmm(data[self.feature_field])
            data[self.out_field] = nll_scores
        return data
