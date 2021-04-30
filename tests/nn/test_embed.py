import torch

from e3nn.util.test import assert_auto_jitable

from nequip.data import AtomicDataDict
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.nn.radial_basis import BesselBasis
from nequip.nn.cutoffs import PolynomialCutoff
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs,
    RadialBasisEdgeEncoding,
)


def test_onehot(CH3CHO):
    _, data = CH3CHO
    oh = OneHotAtomEncoding(
        allowed_species=torch.unique(data[AtomicDataDict.ATOMIC_NUMBERS_KEY]),
        irreps_in=data.irreps,
    )
    assert_auto_jitable(oh)
    assert_AtomicData_equivariant(oh, data)


def test_spharm(CH3CHO):
    _, data = CH3CHO
    sph = SphericalHarmonicEdgeAttrs(
        irreps_edge_sh="0e + 1o + 2e", irreps_in=data.irreps
    )
    assert_auto_jitable(sph)
    assert_AtomicData_equivariant(sph, data)


def test_radial_basis(CH3CHO):
    _, data = CH3CHO
    basis = BesselBasis
    cutoff = PolynomialCutoff
    rad = RadialBasisEdgeEncoding(
        basis,
        cutoff,
        basis_kwargs={"r_max": 5.0},
        cutoff_kwargs={"r_max": 5.0},
        irreps_in=data.irreps,
    )
    assert_auto_jitable(rad)
    assert_AtomicData_equivariant(rad, data)
