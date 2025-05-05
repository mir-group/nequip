# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

"""Custom OmegaConf resolvers for nequip."""

from omegaconf import OmegaConf


def int_div(a, b):
    """Integer division resolver for OmegaConf."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError(f"`int_div` requires integer inputs, got {a} and {b}")

    if a % b != 0:
        raise ValueError(
            f"`int_div` requires exact division, but {a} is not divisible by {b}"
        )

    return a // b


def int_mul(a, b):
    """Integer multiplication resolver for OmegaConf."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError(f"`int_mul` requires integer inputs, got {a} and {b}")
    return a * b


def _register_resolvers():
    """Register all nequip OmegaConf resolvers."""
    OmegaConf.register_new_resolver("int_div", int_div)
    OmegaConf.register_new_resolver("int_mul", int_mul)
