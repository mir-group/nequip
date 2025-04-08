# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from e3nn.o3._irreps import Irreps
from e3nn.o3._spherical_harmonics import SphericalHarmonics
from e3nn.util.jit import compile_mode

from nequip.utils.global_dtype import _GLOBAL_DTYPE
from nequip.utils.compile import conditional_torchscript_jit
from nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin
from ..utils import with_edge_vectors_

from typing import Optional, List, Dict, Union


def _process_per_edge_type_cutoff(type_names, per_edge_type_cutoff, r_max):
    num_types = len(type_names)
    # map dicts from type name to thing into lists
    per_edge_type_cutoff = {
        k: ([e[t] for t in type_names] if not isinstance(e, float) else [e] * num_types)
        for k, e in per_edge_type_cutoff.items()
    }
    per_edge_type_cutoff = [per_edge_type_cutoff[k] for k in type_names]
    per_edge_type_cutoff = torch.as_tensor(
        per_edge_type_cutoff, dtype=_GLOBAL_DTYPE
    ).contiguous()
    assert per_edge_type_cutoff.shape == (num_types, num_types)
    assert torch.all(per_edge_type_cutoff > 0)
    assert torch.all(per_edge_type_cutoff <= r_max)
    return per_edge_type_cutoff


@compile_mode("script")
class EdgeLengthNormalizer(GraphModuleMixin, torch.nn.Module):

    num_types: int
    r_max: float
    _per_edge_type: bool

    def __init__(
        self,
        r_max: float,
        type_names: List[str],
        per_edge_type_cutoff: Optional[
            Dict[str, Union[float, Dict[str, float]]]
        ] = None,
        # bookkeeping
        edge_type_field: str = AtomicDataDict.EDGE_TYPE_KEY,
        norm_length_field: str = AtomicDataDict.NORM_LENGTH_KEY,
        irreps_in=None,
    ):
        super().__init__()

        self.r_max = float(r_max)
        self.num_types = len(type_names)
        self.edge_type_field = edge_type_field
        self.norm_length_field = norm_length_field

        self._per_edge_type = False
        if per_edge_type_cutoff is not None:
            # process per_edge_type_cutoff
            self._per_edge_type = True
            per_edge_type_cutoff = _process_per_edge_type_cutoff(
                type_names, per_edge_type_cutoff, self.r_max
            )
            # compute 1/rmax and flatten for how they're used in forward, i.e. (n_type, n_type) -> (n_type^2,)
            rmax_recip = per_edge_type_cutoff.reciprocal().view(-1)
        else:
            rmax_recip = torch.as_tensor(1.0 / self.r_max, dtype=_GLOBAL_DTYPE)
        self.register_buffer("_rmax_recip", rmax_recip)

        irreps_out = {self.norm_length_field: Irreps([(1, (0, 1))])}
        if self._per_edge_type:
            irreps_out.update({self.edge_type_field: None})

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # == get lengths with shape (num_edges, 1) ==
        data = with_edge_vectors_(data, with_lengths=True)
        r = data[AtomicDataDict.EDGE_LENGTH_KEY].view(-1, 1)
        # == get norm ==
        rmax_recip = self._rmax_recip
        if self._per_edge_type:
            # get edge types with shape (2, num_edges) form first
            edge_type = torch.index_select(
                data[AtomicDataDict.ATOM_TYPE_KEY].reshape(-1),
                0,
                data[AtomicDataDict.EDGE_INDEX_KEY].reshape(-1),
            ).view(2, -1)
            data[self.edge_type_field] = edge_type
            # then convert into row-major NxN matrix index with shape (num_edges,)
            edge_type = edge_type[0] * self.num_types + edge_type[1]
            # (num_type^2,), (num_edges,) -> (num_edges, 1)
            rmax_recip = torch.index_select(rmax_recip, 0, edge_type).unsqueeze(-1)
        data[self.norm_length_field] = r * rmax_recip
        return data


@compile_mode("script")
class BesselEdgeLengthEncoding(GraphModuleMixin, torch.nn.Module):
    r"""Bessel edge length encoding.

    Args:
        num_bessels (int): number of Bessel basis functions
        trainable (bool): whether the :math:`n \pi` coefficients are trainable
        cutoff (torch.nn.Module): ``torch.nn.Module`` to apply a cutoff function that smoothly goes to zero at the cutoff radius
    """

    def __init__(
        self,
        cutoff: torch.nn.Module,
        num_bessels: int = 8,
        trainable: bool = False,
        # bookkeeping
        edge_invariant_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        edge_type_field: str = AtomicDataDict.EDGE_TYPE_KEY,
        norm_length_field: str = AtomicDataDict.NORM_LENGTH_KEY,
        irreps_in=None,
    ):
        super().__init__()
        # === process inputs ===
        self.num_bessels = num_bessels
        self.trainable = trainable
        self.cutoff = conditional_torchscript_jit(cutoff)
        self.edge_invariant_field = edge_invariant_field
        self.edge_type_field = edge_type_field
        self.norm_length_field = norm_length_field

        # === bessel weights ===
        bessel_weights = torch.linspace(
            start=1.0,
            end=self.num_bessels,
            steps=self.num_bessels,
            dtype=_GLOBAL_DTYPE,
        ).unsqueeze(
            0
        )  # (1, num_bessel)
        if self.trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                self.edge_invariant_field: Irreps([(self.num_bessels, (0, 1))]),
                AtomicDataDict.EDGE_CUTOFF_KEY: "0e",
            },
        )
        # i.e. `model_dtype`
        self._output_dtype = torch.get_default_dtype()

    def extra_repr(self) -> str:
        return f"num_bessels={self.num_bessels}"

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # == Bessel basis ==
        x = data[self.norm_length_field]  # (num_edges, 1)
        # (num_edges, 1), (1, num_bessel) -> (num_edges, num_bessel)
        bessel = (torch.sinc(x * self.bessel_weights) * self.bessel_weights).to(
            self._output_dtype
        )

        # == polynomial cutoff ==
        cutoff = self.cutoff(x).to(self._output_dtype)
        data[AtomicDataDict.EDGE_CUTOFF_KEY] = cutoff

        # == save product ==
        data[self.edge_invariant_field] = bessel * cutoff
        return data


@compile_mode("script")
class SphericalHarmonicEdgeAttrs(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        irreps_edge_sh: Union[int, str, Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
    ):
        super().__init__()
        self.out_field = out_field

        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = Irreps(irreps_edge_sh)
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        self.sh = SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )
        # i.e. `model_dtype`
        self._output_dtype = torch.get_default_dtype()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors_(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_sh = self.sh(edge_vec)
        data[self.out_field] = edge_sh.to(self._output_dtype)
        return data


@compile_mode("script")
class AddRadialCutoffToData(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        cutoff: torch.nn.Module,
        norm_length_field: str = AtomicDataDict.NORM_LENGTH_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.cutoff = conditional_torchscript_jit(cutoff)
        self.norm_length_field = norm_length_field
        self._init_irreps(
            irreps_in=irreps_in, irreps_out={AtomicDataDict.EDGE_CUTOFF_KEY: "0e"}
        )
        # i.e. `model_dtype`
        self._output_dtype = torch.get_default_dtype()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.EDGE_CUTOFF_KEY not in data:
            x = data[self.norm_length_field]
            cutoff = self.cutoff(x).to(self._output_dtype)
            data[AtomicDataDict.EDGE_CUTOFF_KEY] = cutoff
        return data
