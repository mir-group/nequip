# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import torch
from e3nn.o3._tensor_product._tensor_product import TensorProduct
from .utils import scatter


class TensorProductScatter(torch.nn.Module):

    def __init__(
        self,
        feature_irreps_in,
        irreps_edge_attr,
        irreps_mid,
        instructions,
    ) -> None:
        super().__init__()

        self.tp = TensorProduct(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        edge_features = self.tp(x[edge_src], edge_attr, edge_weight)
        x = scatter(edge_features, edge_dst, dim=0, dim_size=x.size(0))
        return x
