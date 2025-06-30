from ._tp_scatter_base import TensorProductScatter
from openequivariance import (
    TensorProductConv,
    TPProblem,
    torch_to_oeq_dtype,
)


class OpenEquivarianceTensorProductScatter(TensorProductScatter):

    def __init__(
        self,
        feature_irreps_in,
        irreps_edge_attr,
        irreps_mid,
        instructions,
        use_opaque: bool,
    ) -> None:
        super().__init__(
            feature_irreps_in=feature_irreps_in,
            irreps_edge_attr=irreps_edge_attr,
            irreps_mid=irreps_mid,
            instructions=instructions,
        )
        # ^ we ensure that the base class keeps around a `self.tp` that carries its own set of persistent buffers
        # even though `self.tp` is not used, having its (persistent) buffers always around ensures state dict compatibility when adding on or removing this subclass module

        # OEQ
        tpp = TPProblem(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            irrep_dtype=torch_to_oeq_dtype(self.model_dtype),
            weight_dtype=torch_to_oeq_dtype(self.model_dtype),
            shared_weights=False,
            internal_weights=False,
        )
        self.tp_conv = TensorProductConv(
            tpp, torch_op=True, deterministic=False, use_opaque=use_opaque
        )

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        # explicit cast to account for AMP
        return self.tp_conv(
            x.to(self.model_dtype),
            edge_attr.to(self.model_dtype),
            edge_weight.to(self.model_dtype),
            edge_dst,
            edge_src,
        )
