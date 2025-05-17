# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""Interaction Block"""

from math import sqrt
import torch

from e3nn.o3._irreps import Irreps
from e3nn.o3._linear import Linear
from e3nn.o3._tensor_product._sub import FullyConnectedTensorProduct

from nequip.data import AtomicDataDict

from ._tp_scatter_base import TensorProductScatter
from .mlp import ScalarMLPFunction
from ._graph_mixin import GraphModuleMixin


from typing import Optional


class InteractionBlock(GraphModuleMixin, torch.nn.Module):

    use_sc: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
        radial_mlp_depth: int = 1,
        radial_mlp_width: int = 8,
        avg_num_neighbors: Optional[float] = None,
        use_sc: bool = True,
    ) -> None:
        """InteractionBlock.

        Args:
            irreps_in: input irreps
            irreps_out: output irreps
            radial_mlp_depth (int): number of radial layers
            radial_mlp_width (int): number of hidden neurons in radial function
            avg_num_neighbors (float) : number of neighbors to divide by (default ``None``, i.e. no normalization)
            use_sc (bool): use self-connection or not
        """
        super().__init__()

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_EMBEDDING_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.NODE_ATTRS_KEY,
            ],
            my_irreps_in={
                AtomicDataDict.EDGE_EMBEDDING_KEY: Irreps(
                    [
                        (
                            irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
                            (0, 1),
                        )
                    ]  # (0, 1) is even (invariant) scalars. We are forcing the EDGE_EMBEDDING to be invariant scalars so we can use a dense network
                )
            },
            irreps_out={AtomicDataDict.NODE_FEATURES_KEY: irreps_out},
        )

        # === normalization ===
        self.scatter_norm_factor: Optional[float] = None
        if avg_num_neighbors is not None:
            self.scatter_norm_factor = 1.0 / sqrt(avg_num_neighbors)

        self.use_sc = use_sc

        feature_irreps_in = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        feature_irreps_out = self.irreps_out[AtomicDataDict.NODE_FEATURES_KEY]
        irreps_edge_attr = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]

        # - Build modules -
        self.linear_1 = Linear(
            irreps_in=feature_irreps_in,
            irreps_out=feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid = []
        instructions = []

        for i, (mul, ir_in) in enumerate(feature_irreps_in):
            for j, (_, ir_edge) in enumerate(irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp_scatter = TensorProductScatter(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_mid,
            instructions,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.edge_mlp = ScalarMLPFunction(
            input_dim=self.irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
            output_dim=self.tp_scatter.tp.weight_numel,
            hidden_layers_depth=radial_mlp_depth,
            hidden_layers_width=radial_mlp_width,
            nonlinearity="silu",  # hardcode SiLU
            bias=False,
            forward_weight_init=True,
        )

        self.linear_2 = Linear(
            # irreps_mid has uncoallesed irreps because of the uvu instructions,
            # but there's no reason to treat them seperately for the Linear
            # Note that normalization of o3.Linear changes if irreps are coallesed
            # (likely for the better)
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.sc = None
        if self.use_sc:
            self.sc = FullyConnectedTensorProduct(
                feature_irreps_in,
                self.irreps_in[AtomicDataDict.NODE_ATTRS_KEY],
                feature_irreps_out,
            )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        x = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        if self.sc is not None:
            sc = self.sc(x, data[AtomicDataDict.NODE_ATTRS_KEY])

        x = self.linear_1(x)

        # normalize before TP-scatter
        # necessary to get TorchScript to be able to type infer when its not None
        alpha: Optional[float] = self.scatter_norm_factor
        if alpha is not None:
            x = alpha * x

        edge_weight = self.edge_mlp(data[AtomicDataDict.EDGE_EMBEDDING_KEY])

        x = self.tp_scatter(
            x, data[AtomicDataDict.EDGE_ATTRS_KEY], edge_weight, edge_dst, edge_src
        )

        x = self.linear_2(x)

        if self.sc is not None:
            x = x + sc

        data[AtomicDataDict.NODE_FEATURES_KEY] = x
        return data
