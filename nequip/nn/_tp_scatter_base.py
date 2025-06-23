# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import torch
from e3nn.o3._tensor_product._tensor_product import TensorProduct
from .utils import scatter
from .model_modifier_utils import replace_submodules, model_modifier


class TensorProductScatter(torch.nn.Module):

    def __init__(
        self,
        feature_irreps_in,
        irreps_edge_attr,
        irreps_mid,
        instructions,
    ) -> None:
        super().__init__()

        self.feature_irreps_in = feature_irreps_in
        self.irreps_edge_attr = irreps_edge_attr
        self.irreps_mid = irreps_mid
        self.instructions = instructions

        self.tp = TensorProduct(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.model_dtype = torch.get_default_dtype()

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        edge_features = self.tp(x[edge_src], edge_attr, edge_weight)
        x = scatter(edge_features, edge_dst, dim=0, dim_size=x.size(0))
        return x

    @model_modifier(persistent=False)
    @classmethod
    def enable_OpenEquivariance(cls, model):
        """Enable OpenEquivariance tensor product kernel for accelerated NequIP training and inference."""

        from ._tp_scatter_oeq import OpenEquivarianceTensorProductScatter
        from nequip.utils.dtype import torch_default_dtype
        from nequip.utils.versions.torch_versions import _TORCH_GE_2_4, _TORCH_GE_2_8

        if not _TORCH_GE_2_4:
            raise RuntimeError("OpenEquivariance requires PyTorch >= 2.4.")

        _TRAIN_TIME_COMPILE: bool = model.is_compile_graph_model

        def factory(old):
            with torch_default_dtype(old.model_dtype):
                new = OpenEquivarianceTensorProductScatter(
                    feature_irreps_in=old.feature_irreps_in,
                    irreps_edge_attr=old.irreps_edge_attr,
                    irreps_mid=old.irreps_mid,
                    instructions=old.instructions,
                    use_opaque=_TRAIN_TIME_COMPILE and not _TORCH_GE_2_8,
                )
            return new

        return replace_submodules(model, cls, factory)
