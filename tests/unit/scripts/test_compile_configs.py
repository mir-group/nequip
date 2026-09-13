# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import pytest

from nequip.scripts.compile import _build_inductor_configs


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("cpp.simdlen=256", {"cpp.simdlen": 256}),
        ("cpp.simdlen=0", {"cpp.simdlen": 0}),   # a bare `false` left as a string would be non-empty, i.e. truthy
        ("cpp.cpp_wrapper=false", {"cpp.cpp_wrapper": False}),
        ("cpp.cpp_wrapper=true", {"cpp.cpp_wrapper": True}),
        # bare strings must survive YAML parsing unchanged:
        ("max_autotune_gemm_backends=TRITON", {"max_autotune_gemm_backends": "TRITON"}),
        # a value may itself contain "=" (previously split on the first "=" and the rest treated as a
        # string, now safely YAML-parsed):
        ("cpp.cxx_flags=-DFOO=1", {"cpp.cxx_flags": "-DFOO=1"}),
    ],
)
def test_inductor_config_values_are_yaml_parsed(arg, expected):
    assert _build_inductor_configs([arg]) == expected


def test_build_inductor_configs_multiple():
    assert _build_inductor_configs(["cpp.simdlen=256", "max_autotune=true"]) == {
        "cpp.simdlen": 256,
        "max_autotune": True,
    }
    assert _build_inductor_configs([]) == {}


def test_coerced_simdlen_selects_a_vec_isa():
    """An int `cpp.simdlen` must reach `pick_vec_isa` in a form it can act on."""
    # `torch._inductor.cpu_vec_isa` only exists on newer torch:
    cpu_vec_isa = pytest.importorskip("torch._inductor.cpu_vec_isa")
    from torch._inductor import config

    pick_vec_isa = cpu_vec_isa.pick_vec_isa
    invalid_vec_isa = cpu_vec_isa.invalid_vec_isa
    valid_vec_isa_list = cpu_vec_isa.valid_vec_isa_list

    original = config.cpp.simdlen
    try:
        # `simdlen=0` matches no ISA's bit width, which is how vectorization is disabled
        config.cpp.simdlen = _build_inductor_configs(["cpp.simdlen=0"])["cpp.simdlen"]
        assert pick_vec_isa() is invalid_vec_isa

        # and a width the host actually supports must select that ISA, not fall through
        supported = valid_vec_isa_list()
        if supported:
            width = supported[0].bit_width()
            config.cpp.simdlen = _build_inductor_configs([f"cpp.simdlen={width}"])[
                "cpp.simdlen"
            ]
            assert pick_vec_isa() is not invalid_vec_isa
    finally:
        config.cpp.simdlen = original
