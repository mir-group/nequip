# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import torch.nn.functional

from e3nn.o3._irreps import Irreps

from nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin
from nequip.utils.compile import conditional_torchscript_jit

from typing import List


class NodeTypeEmbed(GraphModuleMixin, torch.nn.Module):
    """Generates node type embeddings.

    Args:
        type_names (List[str]): list of type names
        num_features (int): embedding dimension
        set_features (bool): ``node_features`` will be set in addition to ``node_attrs`` if ``True`` (default)
    """

    num_types: int
    num_features: int
    set_features: bool

    def __init__(
        self,
        type_names: List[str],
        num_features: int,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        # === bookkeeping ===
        self.num_types = len(type_names)
        self.num_features = num_features
        self.set_features = set_features
        # === Embedding module ===
        embed_module = torch.nn.Embedding(
            num_embeddings=self.num_types,
            embedding_dim=self.num_features,
        )
        self.embed_module = conditional_torchscript_jit(embed_module)
        irreps_out = {
            AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_features, (0, 1))])
        }
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        type_numbers: torch.Tensor = data[AtomicDataDict.ATOM_TYPE_KEY].view(-1)
        embedding = self.embed_module(type_numbers)
        data[AtomicDataDict.NODE_ATTRS_KEY] = embedding
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = embedding
        return data
