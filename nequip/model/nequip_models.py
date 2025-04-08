# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import math
from e3nn import o3

from nequip.data import AtomicDataDict

from nequip.nn import (
    GraphModel,
    SequentialGraphNetwork,
    ScalarMLP,
    AtomwiseReduce,
    PerTypeScaleShift,
    ConvNetLayer,
    ForceStressOutput,
    ApplyFactor,
)
from nequip.nn.embedding import (
    NodeTypeEmbed,
    PolynomialCutoff,
    EdgeLengthNormalizer,
    BesselEdgeLengthEncoding,
    SphericalHarmonicEdgeAttrs,
)

from .utils import model_builder
from hydra.utils import instantiate
import warnings
from typing import Sequence, Optional, Dict, Union, Callable


@model_builder
def NequIPGNNEnergyModel(
    num_layers: int = 4,
    l_max: int = 1,
    parity: bool = True,
    num_features: int = 32,
    radial_mlp_depth: int = 2,
    radial_mlp_width: int = 64,
    **kwargs,
) -> GraphModel:
    """NequIP GNN model that predicts energies."""
    # === sanity checks and warnings ===
    assert (
        num_layers > 0
    ), f"at least one convnet layer required, but found `num_layers={num_layers}`"

    # === spherical harmonics ===
    irreps_edge_sh = repr(
        o3.Irreps.spherical_harmonics(lmax=l_max, p=-1 if parity else 1)
    )

    # === convnet ===
    # convert a single set of parameters uniformly for every layer
    feature_irreps_hidden = repr(
        o3.Irreps(
            [
                (num_features, (l, p))
                for p in ((1, -1) if parity else (1,))
                for l in range(l_max + 1)
            ]
        )
    )
    feature_irreps_hidden_list = [feature_irreps_hidden] * (num_layers - 1)
    radial_mlp_depth_list = [radial_mlp_depth] * num_layers
    radial_mlp_width_list = [radial_mlp_width] * num_layers

    # === post convnets ===
    feature_irreps_hidden_list += [repr(o3.Irreps([(num_features, (0, 1))]))]

    # === build model ===
    model = FullNequIPGNNEnergyModel(
        irreps_edge_sh=irreps_edge_sh,
        type_embed_num_features=num_features,
        feature_irreps_hidden=feature_irreps_hidden_list,
        radial_mlp_depth=radial_mlp_depth_list,
        radial_mlp_width=radial_mlp_width_list,
        **kwargs,
    )
    return model


@model_builder
def NequIPGNNModel(**kwargs) -> GraphModel:
    """NequIP GNN model that predicts energies and forces (and stresses if cell is provided).

    Args:
        seed (int): seed for reproducibility
        model_dtype (str): ``float32`` or ``float64``
        r_max (float): cutoff radius
        per_edge_type_cutoff (Dict): one can optionally specify cutoffs for each edge type [must be smaller than ``r_max``] (default ``None``)
        type_names (Sequence[str]): list of atom type names
        num_layers (int): number of interaction blocks, we find 3-5 to work best (default ``4``)
        l_max (int): the maximum rotation order for the network's features, ``1`` is a good default, ``2`` is more accurate but slower (default ``1``)
        parity (bool): whether to include features with odd mirror parity -- often turning parity off gives equally good results but faster networks, so it's worth testing (default ``True``)
        num_features (int): multiplicity of the features, smaller is faster (default ``32``)
        radial_mlp_depth (int): number of radial layers, usually 1-3 works best, smaller is faster (default ``2``)
        radial_mlp_width (int): number of hidden neurons in radial function, smaller is faster (default ``64``)
        num_bessels (int): number of Bessel basis functions (default ``8``)
        bessel_trainable (bool): whether the Bessel roots are trainable (default ``False``)
        polynomial_cutoff_p (int): p-exponent used in polynomial cutoff function, smaller p corresponds to stronger decay with distance (default ``6``)
        avg_num_neighbors (float): used to normalize edge sums for better numerics (default ``None``)
        per_type_energy_scales (float/List[float]): per-atom energy scales, which could be derived from the force RMS of the data (default ``None``)
        per_type_energy_shifts (float/List[float]): per-atom energy shifts, which should generally be isolated atom reference energies or estimated from average per-atom energies of the data (default ``None``)
        per_type_energy_scales_trainable (bool): whether the per-atom energy scales are trainable (default ``False``)
        per_type_energy_shifts_trainable (bool): whether the per-atom energy shifts are trainable (default ``False``)
        pair_potential (torch.nn.Module): additional pair potential term, e.g. ``nequip.nn.pair_potential.ZBL`` (default ``None``)
    """
    return ForceStressOutput(func=NequIPGNNEnergyModel(**kwargs))


@model_builder
def FullNequIPGNNEnergyModel(
    r_max: float,
    type_names: Sequence[str],
    # convnet params
    radial_mlp_depth: Sequence[int],
    radial_mlp_width: Sequence[int],
    feature_irreps_hidden: Sequence[Union[str, o3.Irreps]],
    # irreps and dims
    irreps_edge_sh: Union[int, str, o3.Irreps],
    type_embed_num_features: int,
    # edge length encoding
    per_edge_type_cutoff: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    num_bessels: int = 8,
    bessel_trainable: bool = False,
    polynomial_cutoff_p: int = 6,
    # edge sum normalization
    avg_num_neighbors: Optional[float] = None,
    # per atom energy params
    per_type_energy_scales: Optional[Union[float, Sequence[float]]] = None,
    per_type_energy_shifts: Optional[Union[float, Sequence[float]]] = None,
    per_type_energy_scales_trainable: Optional[bool] = False,
    per_type_energy_shifts_trainable: Optional[bool] = False,
    pair_potential: Optional[Dict] = None,
    # == things that generally shouldn't be changed ==
    # convnet
    convnet_resnet: bool = False,
    convnet_nonlinearity_type: str = "gate",
    convnet_nonlinearity_scalars: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
    convnet_nonlinearity_gates: Dict[int, Callable] = {"e": "silu", "o": "tanh"},
) -> GraphModel:
    """NequIP GNN model that predicts energies based on a more extensive set of arguments."""
    # === sanity checks and warnings ===
    assert all(
        tn.isalnum() for tn in type_names
    ), "`type_names` must contain only alphanumeric characters"

    # require every convnet layer to be specified explicitly in a list
    # infer num_layers from the list size
    assert (
        len(radial_mlp_depth) == len(radial_mlp_width) == len(feature_irreps_hidden)
    ), f"radial_mlp_depth: {radial_mlp_depth}, radial_mlp_width: {radial_mlp_width}, feature_irreps_hidden: {feature_irreps_hidden} should all have the same length"
    num_layers = len(radial_mlp_depth)

    # assert that last convnet produces only scalars
    assert all(
        [l == 0 for l in o3.Irreps(feature_irreps_hidden[-1]).ls]
    ), f"last convnet layer output must only contain scalars but found {feature_irreps_hidden[-1]}"

    if avg_num_neighbors is None:
        warnings.warn(
            "Found `avg_num_neighbors=None` -- it is recommended to set `avg_num_neighbors` for normalization and better numerics during training."
        )
    if per_type_energy_scales is None:
        warnings.warn(
            "Found `per_type_energy_scales=None` -- it is recommended to set `per_type_energy_scales` for better numerics during training."
        )
    if per_type_energy_shifts is None:
        warnings.warn(
            "Found `per_type_energy_shifts=None` -- it is HIGHLY recommended to set `per_type_energy_shifts` as it determines the per-atom energies approaching the isolated atom regime."
        )

    # === encode and embed features ===
    # == edge tensor embedding ==
    spharm = SphericalHarmonicEdgeAttrs(
        irreps_edge_sh=irreps_edge_sh,
    )
    # == edge scalar embedding ==
    edge_norm = EdgeLengthNormalizer(
        r_max=r_max,
        type_names=type_names,
        per_edge_type_cutoff=per_edge_type_cutoff,
        irreps_in=spharm.irreps_out,
    )
    bessel_encode = BesselEdgeLengthEncoding(
        num_bessels=num_bessels,
        trainable=bessel_trainable,
        cutoff=PolynomialCutoff(polynomial_cutoff_p),
        edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=edge_norm.irreps_out,
    )
    # for backwards compatibility of NequIP's bessel encoding
    factor = ApplyFactor(
        in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        factor=(2 * math.pi) / (r_max * r_max),
        irreps_in=bessel_encode.irreps_out,
    )
    # == node scalar embedding ==
    type_embed = NodeTypeEmbed(
        type_names=type_names,
        num_features=type_embed_num_features,
        irreps_in=factor.irreps_out,
    )
    modules = {
        "spharm": spharm,
        "edge_norm": edge_norm,
        "bessel_encode": bessel_encode,
        "factor": factor,
        "type_embed": type_embed,
    }
    prev_irreps_out = type_embed.irreps_out

    # === convnet layers ===
    for layer_i in range(num_layers):
        current_convnet = ConvNetLayer(
            irreps_in=prev_irreps_out,
            feature_irreps_hidden=feature_irreps_hidden[layer_i],
            convolution_kwargs={
                "radial_mlp_depth": radial_mlp_depth[layer_i],
                "radial_mlp_width": radial_mlp_width[layer_i],
                "avg_num_neighbors": avg_num_neighbors,
                # to ensure isolated atom limit
                "use_sc": layer_i != 0,
            },
            resnet=(layer_i != 0) and convnet_resnet,
            nonlinearity_type=convnet_nonlinearity_type,
            nonlinearity_scalars=convnet_nonlinearity_scalars,
            nonlinearity_gates=convnet_nonlinearity_gates,
        )
        prev_irreps_out = current_convnet.irreps_out
        modules.update({f"layer{layer_i}_convnet": current_convnet})

    # === readout ===
    # configure `ScalarMLP` to act as a linear scalar readout
    per_atom_energy_readout = ScalarMLP(
        output_dim=1,
        bias=False,
        forward_weight_init=True,
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        irreps_in=prev_irreps_out,
    )

    per_type_energy_scale_shift = PerTypeScaleShift(
        type_names=type_names,
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        scales=per_type_energy_scales,
        shifts=per_type_energy_shifts,
        scales_trainable=per_type_energy_scales_trainable,
        shifts_trainable=per_type_energy_shifts_trainable,
        irreps_in=per_atom_energy_readout.irreps_out,
    )

    modules.update(
        {
            "per_atom_energy_readout": per_atom_energy_readout,
            "per_type_energy_scale_shift": per_type_energy_scale_shift,
        }
    )

    # === pair potentials ===
    prev_irreps_out = per_type_energy_scale_shift.irreps_out
    if pair_potential is not None:
        pair_potential = instantiate(
            pair_potential, type_names=type_names, irreps_in=prev_irreps_out
        )
        prev_irreps_out = pair_potential.irreps_out
        modules.update({"pair_potential": pair_potential})

    # === sum to total energy ===
    total_energy_sum = AtomwiseReduce(
        irreps_in=prev_irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )
    modules.update({"total_energy_sum": total_energy_sum})

    # === assemble in SequentialGraphNetwork ===
    return SequentialGraphNetwork(modules)


@model_builder
def FullNequIPGNNModel(**kwargs) -> GraphModel:
    """NequIP GNN model that predicts energies and forces (and stresses if cell is provided), based on a more extensive set of arguments."""
    return ForceStressOutput(func=FullNequIPGNNEnergyModel(**kwargs))
