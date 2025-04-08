# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Union, Optional, List

import torch

from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.data.misc import chemical_symbols_to_atomic_numbers_dict
from ._graph_mixin import GraphModuleMixin
from .utils import scatter, with_edge_vectors_
from nequip.utils.compile import conditional_torchscript_jit


class _LJParam(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, param, index1, index2):
        if param.ndim == 2:
            # make it symmetric
            param = param.triu() + param.triu(1).transpose(-1, -2)
            # get for each atom pair
            param = torch.index_select(
                param.view(-1), 0, index1 * param.shape[0] + index2
            )
        # make it positive
        param = param.relu()  # TODO: better way?
        return param


@compile_mode("script")
class LennardJones(GraphModuleMixin, torch.nn.Module):
    """Lennard-Jones and related pair potentials."""

    lj_style: str
    exponent: float

    def __init__(
        self,
        type_names: List[str],
        lj_sigma: Union[torch.Tensor, float],
        lj_delta: Union[torch.Tensor, float] = 0,
        lj_epsilon: Optional[Union[torch.Tensor, float]] = None,
        lj_sigma_trainable: bool = False,
        lj_delta_trainable: bool = False,
        lj_epsilon_trainable: bool = False,
        lj_exponent: Optional[float] = None,
        lj_per_type: bool = True,
        lj_style: str = "lj",
        irreps_in=None,
    ) -> None:
        super().__init__()
        num_types = len(type_names)
        self._init_irreps(
            irreps_in=irreps_in, irreps_out={AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"}
        )
        assert lj_style in ("lj", "lj_repulsive_only", "repulsive")
        self.lj_style = lj_style

        for param, (value, trainable) in {
            "epsilon": (lj_epsilon, lj_epsilon_trainable),
            "sigma": (lj_sigma, lj_sigma_trainable),
            "delta": (lj_delta, lj_delta_trainable),
        }.items():
            if value is None:
                self.register_buffer(param, torch.Tensor())  # torchscript
                continue
            value = torch.as_tensor(value, dtype=torch.get_default_dtype())
            if value.ndim == 0 and lj_per_type:
                # one scalar for all pair types
                value = (
                    torch.ones(
                        num_types, num_types, device=value.device, dtype=value.dtype
                    )
                    * value
                )
            elif value.ndim == 2:
                assert lj_per_type
                # one per pair type, check symmetric
                assert value.shape == (num_types, num_types)
                # per-species square, make sure symmetric
                assert torch.equal(value, value.T)
                value = torch.triu(value)
            else:
                raise ValueError
            setattr(self, param, torch.nn.Parameter(value, requires_grad=trainable))

        if lj_exponent is None:
            lj_exponent = 6.0
        self.exponent = lj_exponent

        self._param = conditional_torchscript_jit(_LJParam())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors_(data, with_lengths=True)
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY]
        edge_len = data[AtomicDataDict.EDGE_LENGTH_KEY].unsqueeze(-1)
        edge_types = torch.index_select(
            atom_types, 0, data[AtomicDataDict.EDGE_INDEX_KEY].reshape(-1)
        ).view(2, -1)
        index1 = edge_types[0]
        index2 = edge_types[1]

        sigma = self._param(self.sigma, index1, index2)
        delta = self._param(self.delta, index1, index2)
        epsilon = self._param(self.epsilon, index1, index2)

        if self.lj_style == "repulsive":
            # 0.5 to assign half and half the energy to each side of the interaction
            lj_eng = 0.5 * epsilon * ((sigma * (edge_len - delta)) ** -self.exponent)
        else:
            lj_eng = (sigma / (edge_len - delta)) ** self.exponent
            lj_eng = torch.neg(lj_eng)
            lj_eng = lj_eng + lj_eng.square()
            # 2.0 because we do the slightly symmetric thing and let
            # ij and ji each contribute half of the LJ energy of the pair
            # this avoids indexing out certain edges in the general case where
            # the edges are not ordered.
            lj_eng = (2.0 * epsilon) * lj_eng

            if self.lj_style == "lj_repulsive_only":
                # if taking only the repulsive part, shift up so the minima is at eng=0
                lj_eng = lj_eng + epsilon
                # this is continuous at the minima, and we mask out everything greater
                # TODO: this is probably broken with NaNs at delta
                lj_eng = lj_eng * (edge_len < (2 ** (1.0 / self.exponent) + delta))

        # apply the cutoff for smoothness
        lj_eng = lj_eng * data[AtomicDataDict.EDGE_CUTOFF_KEY]

        # sum edge LJ energies onto atoms
        atomic_eng = scatter(
            lj_eng,
            edge_center,
            dim=0,
            dim_size=AtomicDataDict.num_nodes(data),
        )
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in data:
            atomic_eng = atomic_eng + data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atomic_eng
        return data

    def __repr__(self) -> str:
        def _f(e):
            e = e.data
            if e.ndim == 0:
                return f"{e:.6f}"
            elif e.ndim == 2:
                return f"{e}"

        return f"PairPotential(lj_style={self.lj_style} | σ={_f(self.sigma)} δ={_f(self.delta)} ε={_f(self.epsilon)} exp={self.exponent:.1f})"


@compile_mode("script")
class SimpleLennardJones(GraphModuleMixin, torch.nn.Module):
    """Simple Lennard-Jones."""

    lj_sigma: float
    lj_epsilon: float
    lj_use_cutoff: bool

    def __init__(
        self,
        lj_sigma: float,
        lj_epsilon: float,
        lj_use_cutoff: bool = False,
        irreps_in=None,
    ) -> None:
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in, irreps_out={AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"}
        )
        self.lj_sigma, self.lj_epsilon, self.lj_use_cutoff = (
            lj_sigma,
            lj_epsilon,
            lj_use_cutoff,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors_(data, with_lengths=True)
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_len = data[AtomicDataDict.EDGE_LENGTH_KEY].unsqueeze(-1)

        lj_eng = (self.lj_sigma / edge_len) ** 6.0
        lj_eng = lj_eng.square() - lj_eng
        lj_eng = 2 * self.lj_epsilon * lj_eng

        if self.lj_use_cutoff:
            # apply the cutoff for smoothness
            lj_eng = lj_eng * data[AtomicDataDict.EDGE_CUTOFF_KEY]

        # sum edge LJ energies onto atoms
        atomic_eng = scatter(
            lj_eng,
            edge_center,
            dim=0,
            dim_size=AtomicDataDict.num_nodes(data),
        )
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in data:
            atomic_eng = atomic_eng + data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atomic_eng
        return data


class _ZBL(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        Z: torch.Tensor,
        r: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        qqr2exesquare: float,
    ) -> torch.Tensor:
        # from LAMMPS pair_zbl_const.h
        pzbl: float = 0.23
        a0: float = 0.46850
        c1: float = 0.02817
        c2: float = 0.28022
        c3: float = 0.50986
        c4: float = 0.18175
        d1: float = -0.20162
        d2: float = -0.40290
        d3: float = -0.94229
        d4: float = -3.19980
        # compute
        edge_types = torch.index_select(atom_types, 0, edge_index.reshape(-1))
        Z = torch.index_select(Z, 0, edge_types.view(-1)).view(
            2, -1
        )  # [center/neigh, n_edge]
        Zi, Zj = Z[0], Z[1]
        del edge_types, Z
        x = ((torch.pow(Zi, pzbl) + torch.pow(Zj, pzbl)) * r) / a0
        psi = (
            c1 * (d1 * x).exp()
            + c2 * (d2 * x).exp()
            + c3 * (d3 * x).exp()
            + c4 * (d4 * x).exp()
        )
        eng = qqr2exesquare * ((Zi * Zj) / r) * psi
        return eng


@compile_mode("script")
class ZBL(GraphModuleMixin, torch.nn.Module):
    """`ZBL <https://docs.lammps.org/pair_zbl.html>`_ pair potential energy term.

    Args:
        type_names (List[str]): list of type names known by the model, ``[atom1, atom2, atom3]``
        chemical_species (List[str]): list of chemical symbols, e.g. ``[C, H, O]``
        units (str): `LAMMPS units <https://docs.lammps.org/units.html>`_ that the data is in; ``metal`` and ``real`` are presently supported -- raise a GitHub issue if more is desired
    """

    def __init__(
        self,
        type_names: List[str],
        chemical_species: List[str],
        units: str,
        irreps_in=None,
    ):
        super().__init__()
        num_types = len(type_names)
        self._init_irreps(
            irreps_in=irreps_in, irreps_out={AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"}
        )
        assert len(chemical_species) == num_types
        atomic_numbers: List[int] = [
            chemical_symbols_to_atomic_numbers_dict[chemical_species[type_i]]
            for type_i in range(num_types)
        ]
        if min(atomic_numbers) < 1:
            raise ValueError(
                f"Your chemical symbols don't seem valid (minimum atomic number is {min(atomic_numbers)} < 1); did you try to use fake chemical symbols for arbitrary atom types?"
            )

        # LAMMPS note on units:
        # > The numerical values of the exponential decay constants in the
        # > screening function depend on the unit of distance. In the above
        # > equation they are given for units of Angstroms. LAMMPS will
        # > automatically convert these values to the distance unit of the
        # > specified LAMMPS units setting. The values of Z should always be
        # > given as multiples of a proton’s charge, e.g. 29.0 for copper.
        # So, we store the atomic numbers directly.
        self.register_buffer(
            "atomic_numbers",
            torch.as_tensor(atomic_numbers, dtype=torch.get_default_dtype()),
        )
        # And we have to convert our value of prefector into the model's physical units
        # Here, prefactor is (electron charge)^2 / (4 * pi * electrical permisivity of vacuum)
        # we have a value for that in eV and Angstrom
        # See https://github.com/lammps/lammps/blob/c415385ab4b0983fa1c72f9e92a09a8ed7eebe4a/src/update.cpp#L187 for values from LAMMPS
        # LAMMPS uses `force->qqr2e * force->qelectron * force->qelectron`
        # Make it a buffer so rescalings are persistent, it still acts as a scalar Tensor
        self.register_buffer(
            "_qqr2exesquare",
            torch.as_tensor(
                {"metal": 14.399645 * (1.0) ** 2, "real": 332.06371 * (1.0) ** 2}[
                    units
                ],
                dtype=torch.float64,
            )
            * 0.5,  # Put half the energy on each of ij, ji
        )
        self._zbl = conditional_torchscript_jit(_ZBL())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """"""
        data = with_edge_vectors_(data, with_lengths=True)
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        zbl_edge_eng = self._zbl(
            Z=self.atomic_numbers,
            r=data[AtomicDataDict.EDGE_LENGTH_KEY],
            atom_types=data[AtomicDataDict.ATOM_TYPE_KEY],
            edge_index=data[AtomicDataDict.EDGE_INDEX_KEY],
            qqr2exesquare=self._qqr2exesquare,
        ).unsqueeze(-1)
        # apply cutoff
        zbl_edge_eng = zbl_edge_eng * data[AtomicDataDict.EDGE_CUTOFF_KEY]
        atomic_eng = scatter(
            zbl_edge_eng,
            edge_center,
            dim=0,
            dim_size=AtomicDataDict.num_nodes(data),
        )
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in data:
            atomic_eng = atomic_eng + data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atomic_eng
        return data


__all__ = [LennardJones, ZBL]
