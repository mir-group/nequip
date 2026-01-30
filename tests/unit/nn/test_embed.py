from e3nn.util.test import assert_auto_jitable

from nequip.utils.test import assert_AtomicData_equivariant
from nequip.nn import SequentialGraphNetwork, GraphModel
from nequip.nn.embedding import (
    PolynomialCutoff,
    SphericalHarmonicEdgeAttrs,
    EdgeLengthNormalizer,
    BesselEdgeLengthEncoding,
)
from nequip.utils import dtype_from_name, torch_default_dtype
from nequip.utils.versions import _TORCH_GE_2_10


def _conditional_assert_auto_jitable(func, **kwargs):
    """torch.jit.script is deprecated in PT 2.10+, so only test JIT compilation for PT < 2.10"""
    if not _TORCH_GE_2_10:
        return assert_auto_jitable(func, **kwargs)
    # for PT >= 2.10, skip the test and return None
    return None


def test_spharm(model_dtype, CH3CHO):
    _, data = CH3CHO
    mdtype = dtype_from_name(model_dtype)
    with torch_default_dtype(mdtype):
        sph = SphericalHarmonicEdgeAttrs(irreps_edge_sh="0e + 1o + 2e")
        gm = GraphModel(sph)
    _conditional_assert_auto_jitable(sph)
    tol = {"float32": 1e-5, "float64": 1e-9}[model_dtype]
    assert_AtomicData_equivariant(gm, data, e3_tolerance=tol)


def test_radial_basis(model_dtype, CH3CHO):
    _, data = CH3CHO

    mdtype = dtype_from_name(model_dtype)
    with torch_default_dtype(mdtype):
        rad = SequentialGraphNetwork(
            {
                "edge_norm": EdgeLengthNormalizer(r_max=5.0, type_names=[0, 1, 2]),
                "bessel": BesselEdgeLengthEncoding(cutoff=PolynomialCutoff(6)),
            }
        )
        gm = GraphModel(rad)
    _conditional_assert_auto_jitable(rad.edge_norm)
    _conditional_assert_auto_jitable(rad.bessel)
    tol = {"float32": 1e-5, "float64": 1e-9}[model_dtype]
    assert_AtomicData_equivariant(gm, data, e3_tolerance=tol)
