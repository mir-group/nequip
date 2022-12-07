""" Interaction Block """
from typing import Optional, Dict, Callable

import torch

from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct, Linear, FullyConnectedTensorProduct

from nequip.data import AtomicDataDict
from nequip.nn.nonlinearities import ShiftedSoftPlus
from ._graph_mixin import GraphModuleMixin


class InteractionBlock(GraphModuleMixin, torch.nn.Module):
    avg_num_neighbors: Optional[float]
    use_sc: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=None,
        use_sc=True,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
    ) -> None:
        """
        InteractionBlock.

        :param irreps_node_attr: Nodes attribute irreps
        :param irreps_edge_attr: Edge attribute irreps
        :param irreps_out: Output irreps, in our case typically a single scalar
        :param radial_layers: Number of radial layers, default = 1
        :param radial_neurons: Number of hidden neurons in radial function, default = 8
        :param avg_num_neighbors: Number of neighbors to divide by, default None => no normalization.
        :param number_of_basis: Number or Basis function, default = 8
        :param irreps_in: Input Features, default = None
        :param use_sc: bool, use self-connection or not
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
                AtomicDataDict.EDGE_EMBEDDING_KEY: o3.Irreps(
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

        self.avg_num_neighbors = avg_num_neighbors
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
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = FullyConnectedNet(
            [self.irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps]
            + invariant_layers * [invariant_neurons]
            + [tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        self.tp = tp

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
        """
        Evaluate interaction Block with ResNet (self-connection).

        :param node_input:
        :param node_attr:
        :param edge_src:
        :param edge_dst:
        :param edge_attr:
        :param edge_length_embedded:

        :return:
        """
        weight = self.fc(data[AtomicDataDict.EDGE_EMBEDDING_KEY])

        x = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        if self.sc is not None:
            sc = self.sc(x, data[AtomicDataDict.NODE_ATTRS_KEY])

        x = self.linear_1(x)
        edge_features = self.tp(
            x[edge_src], data[AtomicDataDict.EDGE_ATTRS_KEY], weight
        )
        x = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

        # Necessary to get TorchScript to be able to type infer when its not None
        avg_num_neigh: Optional[float] = self.avg_num_neighbors
        if avg_num_neigh is not None:
            x = x.div(avg_num_neigh**0.5)

        x = self.linear_2(x)

        if self.sc is not None:
            x = x + sc

        data[AtomicDataDict.NODE_FEATURES_KEY] = x
        return data
