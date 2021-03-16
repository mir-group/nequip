import logging

from nequip.data import AtomicDataDict
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    GradientOutput,
    ConvNet,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)
from nequip.nn.radial_basis import BesselBasis
from nequip.nn.cutoffs import PolynomialCutoff
from nequip.utils import instantiate


# TODO: no allowed_speces?
def EnergyModel(**shared_params):
    """
    The model that predicts total energy.

    Example input for each class
      - OneHotAtomEncoding {'allowed_species': array([1, 6, 8]), 'num_species': None, 'set_features': True}
      - SphericalHarmonicEdgeAttrs {'set_node_features': False, 'l_max': 1}
      - RadialBasisEdgeEncoding {'basis': BesselBasis(), 'cutoff': PolynomialCutoff()}
      - AtomwiseLinear {'field': 'node_features', 'out_field': None, 'irreps_out': '1x0e'}
      - ConvNet {'convolution': <class 'nequip.nn._interaction_block.InteractionBlock'>, 'num_layers': 2,
        'resnet': False, 'nonlinearity_type': 'norm', 'nonlinearity_kwargs': {},
        'feature_irreps_hidden': '16x0o + 16x0e + 16x1o + 16x1e + 16x2o + 16x2e',}
      - AtomwiseLinear
        {'field': 'node_features', 'out_field': None, 'irreps_out': '1x0e'}
    """

    logging.debug("Start building the network model")

    # select the parameters needed for BesselBasis
    # TODO: mark instantiate as ignored for debugger>
    basis, _ = instantiate(
        cls_name=BesselBasis,
        prefix="BesselBasis",
        all_args=shared_params,
    )
    cutoff, _ = instantiate(
        cls_name=PolynomialCutoff,
        prefix="PolynomialCutoff",
        all_args=shared_params,
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=shared_params,
        layers={
            # -- Encode --
            "one_hot": OneHotAtomEncoding,
            "spharm_edges": SphericalHarmonicEdgeAttrs,
            "radial_basis": (
                RadialBasisEdgeEncoding,
                dict(
                    basis=basis,
                    cutoff=cutoff,
                ),
            ),
            # -- Embed features --
            "feature_embedding": AtomwiseLinear,
            # -- ConvNet --
            "convnet": ConvNet,
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # -- output block --
            "conv_to_output_hidden": AtomwiseLinear,
            "output_hidden_to_scalar": (
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field="atomic_energy"),
            ),
            "total_energy_sum": (
                AtomwiseReduce,
                dict(
                    reduce="sum",
                    field="atomic_energy",
                    out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
                ),
            ),
        },
    )


def ForceModel(energy_model):

    return GradientOutput(
        func=energy_model,
        of=AtomicDataDict.TOTAL_ENERGY_KEY,
        wrt=AtomicDataDict.POSITIONS_KEY,
        out_field=AtomicDataDict.FORCE_KEY,
        irreps_in={
            AtomicDataDict.TOTAL_ENERGY_KEY: "1x0e",
            AtomicDataDict.POSITIONS_KEY: "1x1o",
        },
        sign=-1,  # force is the negative gradient
    )
