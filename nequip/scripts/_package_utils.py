# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

"""
Utilities for `nequip-package`, mainly to handle externing and mocking modules.
"""

from nequip.__init__ import _DISCOVERED_NEQUIP_EXTENSION
from typing import Final, Set, Iterable, Optional


_INTERNAL_MODULES: Final[Set[str]] = set(
    ["e3nn", "nequip"] + [ep.value for ep in _DISCOVERED_NEQUIP_EXTENSION]
)

_DEFAULT_EXTERNAL_MODULES: Final[Set[str]] = {
    # included by all custom Triton kernels
    "triton",
    # included by e3nn.util.jit
    "io",
    # included by e3nn TPs
    "opt_einsum_fx",
    "numpy",
    # for NequIP GNN OpenEquivariance OpenEquivarianceTensorProductScatter
    "openequivariance",
    # for version parsing in torch_versions.py
    "packaging",
}
_EXTERNAL_MODULES: Set[str] = set(_DEFAULT_EXTERNAL_MODULES)
_DEFAULT_MOCK_MODULES = {
    # included by e3nn TPs, but not required to run anything with packaged models
    "matplotlib",
}
_MOCK_MODULES: Set[str] = set(_DEFAULT_MOCK_MODULES)


def register_libraries_as_external_for_packaging(
    extern_modules: Optional[Iterable[str]] = None,
    mock_modules: Optional[Iterable[str]] = None,
) -> None:
    """Register a library as "external" or mocked for packaging.

    Registers an entire top-level library as "extern" for packaging. This prevents any code
    from that library from being included in the package file.

    Two primary types of libraries should be registered as external:
    1. Libraries that provide custom C++ or CUDA ops in PyTorch, for example OpenEquivariance.
    2. Large and **stable** third-party, non-PyTorch libraries like NumPy.

    NequIP extension packages should never be registered as extern, and issues that seem to
    suggest that doing so is necessary should almost certainly be solved through refactoring
    the code to make it compatible with being interned.

    .. warning::
        Registering a library as extern means that a *compatible* version of that library must be
        installed in the environment where the package is run or used.  This significantly complicates
        dependency management for packaged models and should be avoided as much as possible.

    Mocking libraries is useful for libraries that are not required to run the model, but are
    used in the code that is packaged. This allows code that imports the mocked module to be
    packaged, but if any code actually tries to use the mocked module, it will raise an error.
    For example, we mock ``matplotlib`` by default.

    .. tip::
        Refactoring code to avoid unnecessary imports in packaged code is **always** preferred over
        registering libraries as external or mock modules.

    See ``_DEFAULT_EXTERNAL_MODULES`` and ``_DEFAULT_MOCK_MODULES`` for the defaults.

    Args:
        extern_modules (Optional[Iterable[str]]): libraries to register as external modules
        mock_modules (Optional[Iterable[str]]): libraries to register as mock modules
    """
    extern_modules = set(extern_modules) if extern_modules is not None else set()
    mock_modules = set(mock_modules) if mock_modules is not None else set()
    assert extern_modules.isdisjoint(
        mock_modules
    ), "Cannot register the same library as both external and mock modules."

    # TODO: should there be a way to extern only submodules of a library, which is supported by the underlying PyTorch package system?
    global _EXTERNAL_MODULES
    if extern_modules is not None:
        _EXTERNAL_MODULES.update(extern_modules)
    global _MOCK_MODULES
    if mock_modules is not None:
        _MOCK_MODULES.update(mock_modules)
