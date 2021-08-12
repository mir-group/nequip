import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin


@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    num_species: int
    set_features: bool

    # TODO: use torch.unique?
    # TODO: type annotation
    # Docstrings
    def __init__(
        self,
        num_species: int,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_species = num_species
        self.set_features = set_features
        # Output irreps are num_species even (invariant) scalars
        irreps_out = {
            AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_species, (0, 1))])
        }
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type):
        type_numbers = data[AtomicDataDict.SPECIES_INDEX_KEY]
        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_species
        ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        return data
