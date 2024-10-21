import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.utils.global_dtype import _GLOBAL_DTYPE
from nequip.utils.compile import conditional_torchscript_jit
from nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin
from ..radial_basis import BesselBasis
from ..cutoffs import PolynomialCutoff

from typing import Optional, List, Dict, Union


def _process_per_edge_type_cutoff(type_names, per_edge_type_cutoff, r_max):
    num_types = len(type_names)
    # map dicts from type name to thing into lists
    per_edge_type_cutoff = {
        k: ([e[t] for t in type_names] if isinstance(e, dict) else [e] * num_types)
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

        irreps_out = {self.norm_length_field: o3.Irreps([(1, (0, 1))])}
        if self._per_edge_type:
            irreps_out.update({self.edge_type_field: None})

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # == get lengths with shape (num_edges, 1) ==
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
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
        irreps_edge_sh: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
    ):
        super().__init__()
        self.out_field = out_field

        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        model_dtype = data.get(
            AtomicDataDict.MODEL_DTYPE_KEY, data[AtomicDataDict.POSITIONS_KEY]
        ).dtype
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_sh = self.sh(edge_vec)
        data[self.out_field] = edge_sh.to(model_dtype)
        return data


@compile_mode("script")
class RadialBasisEdgeEncoding(GraphModuleMixin, torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
        out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.cutoff = conditional_torchscript_jit(cutoff(**cutoff_kwargs))
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                self.out_field: o3.Irreps([(self.basis.num_basis, (0, 1))]),
                AtomicDataDict.EDGE_CUTOFF_KEY: "0e",
            },
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        model_dtype = data.get(
            AtomicDataDict.MODEL_DTYPE_KEY, data[AtomicDataDict.POSITIONS_KEY]
        ).dtype
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
        cutoff = self.cutoff(edge_length).unsqueeze(-1)
        edge_length_embedded = self.basis(edge_length) * cutoff
        data[self.out_field] = edge_length_embedded.to(model_dtype)
        data[AtomicDataDict.EDGE_CUTOFF_KEY] = cutoff.to(model_dtype)
        return data


@compile_mode("script")
class AddRadialCutoffToData(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        cutoff=PolynomialCutoff,
        cutoff_kwargs={},
        irreps_in=None,
    ):
        super().__init__()
        self.cutoff = conditional_torchscript_jit(cutoff(**cutoff_kwargs))
        self._init_irreps(
            irreps_in=irreps_in, irreps_out={AtomicDataDict.EDGE_CUTOFF_KEY: "0e"}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.EDGE_CUTOFF_KEY not in data:
            model_dtype = data.get(
                AtomicDataDict.MODEL_DTYPE_KEY, data[AtomicDataDict.POSITIONS_KEY]
            ).dtype
            data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
            edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
            cutoff = self.cutoff(edge_length).unsqueeze(-1)
            data[AtomicDataDict.EDGE_CUTOFF_KEY] = cutoff.to(model_dtype)
        return data
