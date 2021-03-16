from typing import List

import torch
import logging

from e3nn import o3
from e3nn.nn import Gate, NormActivation

from nequip.data import AtomicDataDict
from nequip.nn import (
    GraphModuleMixin,
    InteractionBlock,
)
from nequip.nn.nonlinearities import ShiftedSoftPlus
from nequip.utils.tp_utils import tp_path_exists


act = {
    1: ShiftedSoftPlus,
    -1: torch.tanh,
}

act_gates = {
    1: ShiftedSoftPlus,
    -1: torch.abs,
}


class ConvNet(GraphModuleMixin, torch.nn.Module):
    """
    Args:

    """

    resnet: bool
    num_layers: int
    _do_resnet: List[bool]

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        convolution=InteractionBlock,
        convolution_kwargs: dict = {},
        num_layers: int = 3,
        resnet: bool = True,
        nonlinearity_type: str = "gate",
        nonlinearity_kwargs: dict = {},
    ):
        super().__init__()
        # initialization
        assert nonlinearity_type in ("gate", "norm")
        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.resnet = resnet
        self.num_layers = num_layers

        # We'll set irreps_out later when we know them
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.NODE_ATTRS_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.EDGE_EMBEDDING_KEY,
            ],
        )

        # building the network
        self.convs = torch.nn.ModuleList()
        self.nonlins = torch.nn.ModuleList()

        self._do_resnet = [False] * num_layers
        edge_attr_irreps = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]
        irreps_layer_out_prev = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        conv_irreps_in_dict = self.irreps_in.copy()

        for layer_idx in range(num_layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.feature_irreps_hidden
                    if ir.l == 0
                    and tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, ir)
                ]
            )

            irreps_nonscalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.feature_irreps_hidden
                    if ir.l > 0
                    and tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, ir)
                ]
            )

            irreps_layer_out = irreps_scalars + irreps_nonscalars

            if nonlinearity_type == "gate":
                ir = (
                    "0e"
                    if tp_path_exists(irreps_layer_out_prev, edge_attr_irreps, "0e")
                    else "0o"
                )
                irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_nonscalars])

                # TO DO, it's not that safe to directly use the
                # dictionary
                equivariant_nonlin = Gate(
                    irreps_scalars=irreps_scalars,
                    act_scalars=[act[ir.p] for _, ir in irreps_scalars],
                    irreps_gates=irreps_gates,
                    act_gates=[act_gates[ir.p] for _, ir in irreps_gates],
                    irreps_nonscalars=irreps_nonscalars,
                    **nonlinearity_kwargs,
                )

                conv_irreps_out = equivariant_nonlin.irreps_in

            else:
                conv_irreps_out = irreps_layer_out

                equivariant_nonlin = NormActivation(
                    irreps_in=conv_irreps_out,
                    scalar_nonlinearity=ShiftedSoftPlus,
                    normalize=True,
                    epsilon=1e-8,
                    bias=False,
                    **nonlinearity_kwargs,
                )

            # TODO: partial resnet?
            if irreps_layer_out == irreps_layer_out_prev and resnet:
                # We are doing resnet updates and can for this layer
                self._do_resnet[layer_idx] = True

            conv_irreps_in_dict[
                AtomicDataDict.NODE_FEATURES_KEY
            ] = irreps_layer_out_prev

            # TODO: last convolution should go to explicit irreps out

            # TO DO, it's not that safe to directly use the
            # dictionary
            convolution_kwargs.pop("irreps_in", None)
            convolution_kwargs.pop("irreps_out", None)
            logging.debug(
                f" parameters used to initialize {convolution.__name__}={convolution_kwargs}"
            )

            conv = convolution(
                irreps_in=conv_irreps_in_dict,
                irreps_out=conv_irreps_out,
                **convolution_kwargs,
            )

            self.convs.append(conv)
            self.nonlins.append(equivariant_nonlin)

            irreps_layer_out_prev = irreps_layer_out
        # The output features are whatever the last layer output
        self.irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_layer_out_prev

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        for layer_i, (conv, nonlin) in enumerate(zip(self.convs, self.nonlins)):
            # save old features for resnet
            old_x = data[AtomicDataDict.NODE_FEATURES_KEY]
            # run convolution
            data = conv(data)
            # do nonlinearity
            data[AtomicDataDict.NODE_FEATURES_KEY] = nonlin(
                data[AtomicDataDict.NODE_FEATURES_KEY]
            )
            # do resnet
            if self._do_resnet[layer_i]:
                data[AtomicDataDict.NODE_FEATURES_KEY] = (
                    old_x + data[AtomicDataDict.NODE_FEATURES_KEY]
                )
        return data
