from e3nn.util.test import assert_auto_jitable

from nequip.nn.radial_basis import BesselBasis
from nequip.nn.cutoffs import PolynomialCutoff
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs,
    RadialBasisEdgeEncoding,
)


def test_onehot_jit():
    oh = OneHotAtomEncoding(allowed_species=[1, 4, 5])
    assert_auto_jitable(oh)


def test_spharm_jit():
    sph = SphericalHarmonicEdgeAttrs(
        irreps_edge_sh=3, irreps_in={"another_field": "2x0e"}
    )
    assert_auto_jitable(sph)


def test_radial_basis_jit():
    basis = BesselBasis
    cutoff = PolynomialCutoff
    rad = RadialBasisEdgeEncoding(basis, cutoff)
    assert_auto_jitable(rad)
