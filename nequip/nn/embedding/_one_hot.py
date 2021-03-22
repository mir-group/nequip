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
        allowed_species=None,
        num_species: int = None,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        if allowed_species is not None and num_species is not None:
            raise ValueError("allowed_species and num_species cannot both be provided.")

        if allowed_species is not None:
            num_species = len(allowed_species)
            allowed_species = torch.as_tensor(allowed_species)
            self.register_buffer("_min_Z", allowed_species.min())
            self.register_buffer("_max_Z", allowed_species.max())
            Z_to_index = torch.full(
                (1 + self._max_Z - self._min_Z,), -1, dtype=torch.long
            )
            Z_to_index[allowed_species - self._min_Z] = torch.arange(num_species)
            self.register_buffer("_Z_to_index", Z_to_index)
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

    @torch.jit.export
    def index_for_atomic_numbers(self, atomic_nums: torch.Tensor):
        if atomic_nums.min() < self._min_Z or atomic_nums.max() > self._max_Z:
            raise RuntimeError("Invalid atomic numbers for this OneHotEncoding")

        out = self._Z_to_index[atomic_nums - self._min_Z]
        assert out.min() >= 0, "Invalid atomic numbers for this OneHotEncoding"
        return out

    def forward(self, data: AtomicDataDict.Type):
        if AtomicDataDict.SPECIES_INDEX_KEY in data:
            type_numbers = data[AtomicDataDict.SPECIES_INDEX_KEY]
        elif AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
            type_numbers = self.index_for_atomic_numbers(
                data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
            )
            data[AtomicDataDict.SPECIES_INDEX_KEY] = type_numbers
        else:
            raise ValueError(
                "Nothing in this `data` to encode, need either species index or atomic numbers"
            )
        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_species
        ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        return data
