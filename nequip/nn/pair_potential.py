# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import List

import torch

from e3nn.o3._irreps import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.data.misc import chemical_symbols_to_atomic_numbers_dict
from ._graph_mixin import GraphModuleMixin
from .utils import scatter, with_edge_vectors_
from nequip.utils.compile import conditional_torchscript_jit
from .embedding.cutoffs import PolynomialCutoff


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
        # (num_atoms,) -> (num_atoms, 1)
        node_Zs = torch.nn.functional.embedding(atom_types.view(-1), Z.view(-1, 1))
        # (num_atoms,) -> (2 * num_edges,)
        edge_Zs = torch.nn.functional.embedding(edge_index.view(-1), node_Zs).view(
            2, -1
        )
        Zi = torch.select(edge_Zs, 0, 0)
        Zj = torch.select(edge_Zs, 0, 1)
        del node_Zs, edge_Zs
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

    Useful as a prior for core repulsion to mitigate molecular dynamics failure modes associated with atoms getting too close.

    Args:
        type_names (List[str]): list of type names known by the model, ``[atom1, atom2, atom3]``
        chemical_species (List[str]): list of chemical symbols, e.g. ``[C, H, O]``
        units (str): `LAMMPS units <https://docs.lammps.org/units.html>`_ that the data is in; ``metal`` and ``real`` are presently supported -- raise a GitHub issue if more is desired
        polynomial_cutoff_p (float): exponent used for the polynomial cutoff (default ``6``)
    """

    def __init__(
        self,
        type_names: List[str],
        chemical_species: List[str],
        units: str,
        polynomial_cutoff_p: float = 6.0,
        per_atom_energy_field: str = AtomicDataDict.PER_ATOM_ENERGY_KEY,
        irreps_in=None,
    ):
        super().__init__()
        num_types = len(type_names)
        self.per_atom_energy_field = per_atom_energy_field

        # === irreps registration ===
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NORM_LENGTH_KEY],
            irreps_out={self.per_atom_energy_field: "0e"},
        )
        if self.per_atom_energy_field in self.irreps_in:
            energy_irreps = Irreps(self.irreps_in[self.per_atom_energy_field])
            assert all(ir.l == 0 for _, ir in energy_irreps), (
                f"{self.per_atom_energy_field} must be scalar irreps, found {energy_irreps}"
            )
            self.irreps_out[self.per_atom_energy_field] = energy_irreps

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
        self.cutoff = conditional_torchscript_jit(PolynomialCutoff(polynomial_cutoff_p))
        self.model_dtype = torch.get_default_dtype()
        self._zbl = conditional_torchscript_jit(_ZBL())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """"""
        data = with_edge_vectors_(data, with_lengths=True)
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        # account for possibility of reduced num nodes in atomic energy in a local-ghost atom context
        if self.per_atom_energy_field in data:
            num_nodes = data[self.per_atom_energy_field].size(0)
        else:
            num_nodes = AtomicDataDict.num_nodes(data)

        zbl_edge_eng = self._zbl(
            Z=self.atomic_numbers,
            r=data[AtomicDataDict.EDGE_LENGTH_KEY].view(-1),
            atom_types=data[AtomicDataDict.ATOM_TYPE_KEY],
            edge_index=data[AtomicDataDict.EDGE_INDEX_KEY],
            qqr2exesquare=self._qqr2exesquare,
        ).unsqueeze(-1)

        # apply cutoff
        zbl_edge_cutoff = self.cutoff(data[AtomicDataDict.NORM_LENGTH_KEY]).to(
            self.model_dtype
        )
        zbl_edge_eng = zbl_edge_eng * zbl_edge_cutoff
        atomic_eng = scatter(
            zbl_edge_eng,
            edge_center,
            dim=0,
            dim_size=num_nodes,
        )
        if self.per_atom_energy_field in data:
            atomic_eng = atomic_eng + data[self.per_atom_energy_field]
        data[self.per_atom_energy_field] = atomic_eng
        return data
