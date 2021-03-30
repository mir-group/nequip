import inspect
import logging

from nequip.data import AtomicDataDict
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    GradientOutput,
    PerSpeciesShift,
    ConvNetLayer,
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
      - ConvNetLayer {'convolution': <class 'nequip.nn._interaction_block.InteractionBlock'>, 'num_layers': 2,
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

    num_layers = shared_params.pop("num_layers", 3)

    layers = {
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
    }

    # add convnet layers
    # we get any before and after layers:
    before_layer = shared_params.pop("before_layer", {})
    after_layer = shared_params.pop("after_layer", {})
    if len(set(before_layer.keys()).union(after_layer.keys())) > 1:
        raise ValueError(
            "before_layer and after_layer may not have common module names"
        )
    # insertion preserves order
    for layer_i in range(num_layers):
        layers.update({f"layer{layer_i}_{bk}": v for bk, v in before_layer.items()})

        layer_name = f"layer{layer_i}_convnet"

        # find out convolution type and other parameters
        layer_kwargs = instantiate(
            ConvNetLayer,
            prefix=["ConvNetLayer", "convnet", f"layer{layer_i}", layer_name],
            all_args=shared_params,
            remove_kwargs=True,
            return_args_only=True,
        )
        convolution = layer_kwargs["convolution"]

        # find out kwargs for convlution
        convolution_kwargs = instantiate(
            convolution,
            prefix=[
                convolution.__name__,
                "convolution",
                "ConvNetLayer",
                "convnet",
                f"layer{layer_i}",
                f"layer{layer_i}_{convolution.__name__}"
            ],
            all_args=shared_params,
            remove_kwargs=True,
            return_args_only=True,
        )
        layer_kwargs["convolution_kwargs"] = convolution_kwargs

        layers[layer_name] = (ConvNetLayer, layer_kwargs)

        layers.update({f"layer{layer_i}_{ak}": v for ak, v in after_layer.items()})

    # .update also maintains insertion order
    layers.update(
        {
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
                    out_field="raw_total_energy",
                ),
            ),
            "energy_shift": (
                PerSpeciesShift,
                dict(
                    field="raw_total_energy",
                    out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
                ),
            ),
        }
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=shared_params, layers=layers
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
