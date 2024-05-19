from nequip.nn import SequentialGraphNetwork, AtomwiseReduce
from nequip.nn.embedding import AddRadialCutoffToData
from nequip.data import AtomicDataDict
from nequip.nn.pair_potential import SimpleLennardJones, LennardJones, ZBL

_PAIR_STYLES = {"LJ": SimpleLennardJones, "LJ_fancy": LennardJones, "ZBL": ZBL}


def PairPotentialTerm(
    model: SequentialGraphNetwork,
    config,
) -> SequentialGraphNetwork:
    assert isinstance(model, SequentialGraphNetwork)

    model.insert_from_parameters(
        shared_params=config,
        name="pair_potential",
        builder=_PAIR_STYLES[config.pair_style],
        before="total_energy_sum",
    )
    return model


def PairPotential(config) -> SequentialGraphNetwork:
    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers={
            "cutoff": AddRadialCutoffToData,
            "pair_potential": _PAIR_STYLES[config.pair_style],
            "total_energy_sum": (
                AtomwiseReduce,
                dict(
                    reduce="sum",
                    field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
                ),
            ),
        },
    )
