from typing import Union, Optional, Dict, List

import torch
from torch_runstats.scatter import scatter

from e3nn.util.jit import compile_mode

import ase.data

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, RescaleOutput
from nequip.nn.cutoffs import PolynomialCutoff


@torch.jit.script
def _param(param, index1, index2):
    if param.ndim == 2:
        # make it symmetric
        param = param.triu() + param.triu(1).transpose(-1, -2)
        # get for each atom pair
        param = torch.index_select(param.view(-1), 0, index1 * param.shape[0] + index2)
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
        num_types: int,
        lj_sigma: Union[torch.Tensor, float],
        lj_delta: Union[torch.Tensor, float] = 0,
        lj_epsilon: Optional[Union[torch.Tensor, float]] = None,
        lj_sigma_trainable: bool = False,
        lj_delta_trainable: bool = False,
        lj_epsilon_trainable: bool = False,
        lj_exponent: Optional[float] = None,
        lj_per_type: bool = True,
        lj_style: str = "lj",
        cutoff=PolynomialCutoff,
        cutoff_kwargs={},
        irreps_in=None,
    ) -> None:
        super().__init__()
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

        self._has_cutoff = cutoff is not None
        if self._has_cutoff:
            self.cutoff = cutoff(**cutoff_kwargs)
        else:
            self.cutoff = torch.nn.Identity()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY]
        edge_len = data[AtomicDataDict.EDGE_LENGTH_KEY].unsqueeze(-1)
        edge_types = torch.index_select(
            atom_types, 0, data[AtomicDataDict.EDGE_INDEX_KEY].reshape(-1)
        ).view(2, -1)
        index1 = edge_types[0]
        index2 = edge_types[1]

        sigma = _param(self.sigma, index1, index2)
        delta = _param(self.delta, index1, index2)
        epsilon = _param(self.epsilon, index1, index2)

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

        if self._has_cutoff:
            # apply the cutoff for smoothness
            lj_eng = lj_eng * self.cutoff(edge_len)

        # sum edge LJ energies onto atoms
        atomic_eng = scatter(
            lj_eng,
            edge_center,
            dim=0,
            dim_size=len(data[AtomicDataDict.POSITIONS_KEY]),
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

    def update_for_rescale(self, rescale_module: RescaleOutput):
        if AtomicDataDict.PER_ATOM_ENERGY_KEY not in rescale_module.scale_keys:
            return
        if not rescale_module.has_scale:
            return
        with torch.no_grad():
            # Our energy will be scaled by scale_by later, so we have to divide here to cancel out:
            self.epsilon.copy_(self.epsilon / rescale_module.scale_by.item())


@torch.jit.script
def _zbl(
    Z: torch.Tensor,
    r: torch.Tensor,
    atom_types: torch.Tensor,
    edge_index: torch.Tensor,
    r_max: float,
    p: float,
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

    # compute cutoff envelope
    r = r / r_max
    cutoff = 1.0 - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r, p))
    cutoff = cutoff + (p * (p + 2.0) * torch.pow(r, p + 1.0))
    cutoff = cutoff - ((p * (p + 1.0) / 2) * torch.pow(r, p + 2.0))
    cutoff = cutoff * (r < 1.0)

    return cutoff * eng


@compile_mode("script")
class ZBL(GraphModuleMixin, torch.nn.Module):
    """Add a ZBL pair potential to the edge energy.

    Args:
        units (str): what units the model/data are in using LAMMPS names.
    """

    num_types: int
    r_max: float
    PolynomialCutoff_p: float

    def __init__(
        self,
        num_types: int,
        r_max: float,
        units: str,
        type_to_chemical_symbol: Optional[Dict[int, str]] = None,
        PolynomialCutoff_p: float = 6.0,
        irreps_in=None,
    ):
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in, irreps_out={AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"}
        )
        if type_to_chemical_symbol is not None:
            assert set(type_to_chemical_symbol.keys()) == set(range(num_types))
            atomic_numbers: List[int] = [
                ase.data.atomic_numbers[type_to_chemical_symbol[type_i]]
                for type_i in range(num_types)
            ]
            if min(atomic_numbers) < 1:
                raise ValueError(
                    f"Your chemical symbols don't seem valid (minimum atomic number is {min(atomic_numbers)} < 1); did you try to use fake chemical symbols for arbitrary atom types?  If so, instead provide atom_types directly in your dataset and specify `type_names` and `type_to_chemical_symbol` in your config. `type_to_chemical_symbol` then tells ZBL what atomic numbers to use for the various atom types in your system."
                )
        else:
            raise RuntimeError(
                "Either chemical_symbol_to_type or type_to_chemical_symbol is required."
            )
        assert len(atomic_numbers) == num_types
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
        self.r_max = float(r_max)
        self.PolynomialCutoff_p = float(PolynomialCutoff_p)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        zbl_edge_eng = _zbl(
            Z=self.atomic_numbers,
            r=data[AtomicDataDict.EDGE_LENGTH_KEY],
            atom_types=data[AtomicDataDict.ATOM_TYPE_KEY],
            edge_index=data[AtomicDataDict.EDGE_INDEX_KEY],
            r_max=self.r_max,
            p=self.PolynomialCutoff_p,
            qqr2exesquare=self._qqr2exesquare,
        ).unsqueeze(-1)
        atomic_eng = scatter(
            zbl_edge_eng,
            edge_center,
            dim=0,
            dim_size=len(data[AtomicDataDict.POSITIONS_KEY]),
        )
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in data:
            atomic_eng = atomic_eng + data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atomic_eng
        return data

    def update_for_rescale(self, rescale_module: RescaleOutput):
        if AtomicDataDict.PER_ATOM_ENERGY_KEY not in rescale_module.scale_keys:
            return
        if not rescale_module.has_scale:
            return
        # Our energy will be scaled by scale_by later, so we have to divide here to cancel out:
        self._qqr2exesquare /= rescale_module.scale_by.item()


__all__ = [LennardJones, ZBL]
