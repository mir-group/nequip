# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from nequip.nn import SequentialGraphNetwork, AtomwiseReduce, ForceStressOutput
from nequip.nn.embedding import (
    EdgeLengthNormalizer,
    AddRadialCutoffToData,
    PolynomialCutoff,
)
from nequip.data import AtomicDataDict
from nequip.nn.pair_potential import ZBL
from .utils import model_builder

from typing import Optional, Dict, Union, Sequence


@model_builder
def ZBLPairPotential(
    r_max: float,
    type_names: Sequence[str],
    chemical_species: Sequence[str],
    units: str,
    polynomial_cutoff_p: int = 6,
    per_edge_type_cutoff: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
):
    """
    Model builder for a force field containing only a ZBL pair potential term, mainly for internal testing purposes.
    """
    edge_norm = EdgeLengthNormalizer(
        r_max=r_max,
        type_names=type_names,
        per_edge_type_cutoff=per_edge_type_cutoff,
    )
    cutoff = AddRadialCutoffToData(
        cutoff=PolynomialCutoff(polynomial_cutoff_p),
        irreps_in=edge_norm.irreps_out,
    )
    zbl_module = ZBL(
        type_names=type_names,
        chemical_species=chemical_species,
        units=units,
        irreps_in=cutoff.irreps_out,
    )
    energy_sum = AtomwiseReduce(
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        irreps_in=zbl_module.irreps_out,
    )
    energy_model = SequentialGraphNetwork(
        {
            "edge_norm": edge_norm,
            "cutoff": cutoff,
            "pair_potential": zbl_module,
            "total_energy_sum": energy_sum,
        }
    )
    model = ForceStressOutput(func=energy_model)

    return model
