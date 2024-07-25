from e3nn.util.test import assert_auto_jitable

from nequip.utils.test import assert_AtomicData_equivariant
from nequip.nn.radial_basis import BesselBasis
from nequip.nn.cutoffs import PolynomialCutoff
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs,
    RadialBasisEdgeEncoding,
)
from nequip.utils import dtype_from_name, torch_default_dtype


def test_onehot(model_dtype, CH3CHO):
    _, data = CH3CHO
    with torch_default_dtype(dtype_from_name(model_dtype)):
        oh = OneHotAtomEncoding(
            type_names=["A", "B", "C"],
        )
    assert_auto_jitable(oh)
    assert_AtomicData_equivariant(oh, data)


def test_spharm(model_dtype, CH3CHO):
    _, data = CH3CHO
    with torch_default_dtype(dtype_from_name(model_dtype)):
        sph = SphericalHarmonicEdgeAttrs(irreps_edge_sh="0e + 1o + 2e")
    assert_auto_jitable(sph)
    assert_AtomicData_equivariant(sph, data)


def test_radial_basis(model_dtype, CH3CHO):
    _, data = CH3CHO

    with torch_default_dtype(dtype_from_name(model_dtype)):
        basis = BesselBasis
        cutoff = PolynomialCutoff
        rad = RadialBasisEdgeEncoding(
            basis,
            cutoff,
            basis_kwargs={"r_max": 5.0},
            cutoff_kwargs={"r_max": 5.0},
        )
    assert_auto_jitable(rad)
    assert_AtomicData_equivariant(rad, data)
