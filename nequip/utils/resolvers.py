# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

"""Custom OmegaConf resolvers for nequip."""

from omegaconf import OmegaConf


def _sanitize_int(x, client: str):
    err_msg = f"`{client} resolver accepts nonnegative integer inputs, but found {x}"
    if isinstance(x, str):
        assert x.isdigit(), err_msg
        x = int(x)
    assert isinstance(x, int), err_msg
    return x


def int_div(a, b):
    """Integer division resolver for OmegaConf."""
    a = _sanitize_int(a, "int_div")
    b = _sanitize_int(b, "int_div")

    if a % b != 0:
        raise ValueError(
            f"`int_div` requires exact division, but {a} is not divisible by {b}"
        )

    return a // b


def int_mul(a, b):
    """Integer multiplication resolver for OmegaConf."""
    a = _sanitize_int(a, "int_mul")
    b = _sanitize_int(b, "int_mul")
    return a * b


def _register_resolvers():
    """Register all nequip OmegaConf resolvers."""
    OmegaConf.register_new_resolver("int_div", int_div)
    OmegaConf.register_new_resolver("int_mul", int_mul)
