from ._tp_scatter_base import TensorProductScatter
from openequivariance import TensorProductConv, TPProblem


class OpenEquivarianceTensorProductScatter(TensorProductScatter):

    def __init__(
        self,
        feature_irreps_in,
        irreps_edge_attr,
        irreps_mid,
        instructions,
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
        )
        self.tp_conv = TensorProductConv(tpp, torch_op=True, deterministic=False)

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        return self.tp_conv(x, edge_attr, edge_weight, edge_dst, edge_src)
