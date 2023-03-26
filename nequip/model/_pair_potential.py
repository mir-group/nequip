from nequip.nn import SequentialGraphNetwork, AtomwiseReduce
from nequip.data import AtomicDataDict
from nequip.nn.pair_potential import LennardJones, ZBL


def PairPotentialTerm(
    model: SequentialGraphNetwork,
    config,
) -> SequentialGraphNetwork:
    assert isinstance(model, SequentialGraphNetwork)

    model.insert_from_parameters(
        shared_params=config,
        name="pair_potential",
        builder={"LJ": LennardJones, "ZBL": ZBL}[config.pair_style],
        before="total_energy_sum",
    )
    return model


def PairPotential(config) -> SequentialGraphNetwork:
    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers={
            "pair_potential": {"LJ": LennardJones, "ZBL": ZBL}[config.pair_style],
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
