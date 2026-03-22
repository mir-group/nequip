# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Dict, Optional, Sequence

from hydra.utils import instantiate

from nequip.data import AtomicDataDict
from nequip.nn import AtomwiseReduce, SequentialGraphNetwork


def _append_energy_modules(
    model: SequentialGraphNetwork,
    type_names: Sequence[str],
    pair_potential: Optional[Dict] = None,
):
    # === pair potentials ===
    prev_irreps_out = model.irreps_out
    if pair_potential is not None:
        pair_potential = instantiate(
            pair_potential,
            type_names=type_names,
            irreps_in=prev_irreps_out,
        )
        prev_irreps_out = pair_potential.irreps_out
        model.append("pair_potential", pair_potential)

    # === sum to total energy ===
    # perform sum after applying `pair_potential`
    total_energy_sum = AtomwiseReduce(
        irreps_in=prev_irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )
    model.append("total_energy_sum", total_energy_sum)
    return model
