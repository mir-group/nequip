# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from typing import Final, Callable, Type, Set, Union, List


_CUSTOM_OP_ATTR_NAME: Final[str] = "_nequip_module_uses_custom_op_libraries"
_GLOBAL_CUSTOM_OP_LIBRARIES: Set[str] = set()


def uses_custom_op(
    library_name: Union[str, List[str]],
) -> Callable[[Type[torch.nn.Module]], Type[torch.nn.Module]]:
    """Decorator to mark nn.Module classes that use custom ops from external libraries.

    This decorator should be applied to nn.Module classes that use custom PyTorch
    operations provided by external libraries. The library information is stored
    as metadata on the class and can be retrieved later to ensure the necessary
    libraries are loaded before using compiled/packaged models.

    External libraries specified this way will also be automatically marked as
    extern during packaging.

    Note:
        Modules that use custom Trition operations should not be marked with this
        decorator and are supported out-of-the-box.

    Args:
        library_name: Name of the library/libraries that provides the custom op(s).
            Can be a single string or a list of strings for multiple libraries.

    Returns:
        A decorator function that adds custom op metadata to the module class.

    Example:
        >>> @uses_custom_op("my_custom_ops")
        >>> class MyModule(torch.nn.Module):
        ...     def forward(self, x):
        ...         return my_custom_ops.special_operation(x)

        >>> @uses_custom_op(["library1", "library2"])
        >>> class MultiOpModule(torch.nn.Module):
        ...     def forward(self, x):
        ...         x = library1.op1(x)
        ...         return library2.op2(x)
    """

    def decorator(cls: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
        # Normalize to list
        libraries = (
            [library_name] if isinstance(library_name, str) else list(library_name)
        )

        # Add to global registry
        _GLOBAL_CUSTOM_OP_LIBRARIES.update(libraries)

        # Check if the class already has custom op libraries (from parent class)
        existing_libraries = getattr(cls, _CUSTOM_OP_ATTR_NAME, None)
        if existing_libraries is not None:
            # Merge with existing libraries
            libraries = list(set(existing_libraries) | set(libraries))

        # Set the attribute on the class
        setattr(cls, _CUSTOM_OP_ATTR_NAME, libraries)

        return cls

    return decorator


def _get_module_custom_op_libraries(module_class: Type[torch.nn.Module]) -> Set[str]:
    libraries = getattr(module_class, _CUSTOM_OP_ATTR_NAME, None)
    if libraries is None:
        return set()
    return set(libraries)


def get_custom_op_libraries(model: torch.nn.Module) -> Set[str]:
    """Walk the module tree and collect all required custom op libraries.

    This function recursively traverses all submodules of the given model
    and collects the names of all libraries that provide custom operations
    used by any module in the tree.

    Args:
        model: The root module to analyze.

    Returns:
        A set of library names that must be imported before the model
        can be used (especially in compiled/packaged form).

    Example:
        >>> model = MyComplexModel()
        >>> required_libraries = get_custom_op_libraries(model)
        >>> for lib in required_libraries:
        ...     importlib.import_module(lib)
    """
    all_libraries = set()

    # Walk through all modules (including the root)
    for module in model.modules():
        # Get libraries from this module's class
        module_libraries = _get_module_custom_op_libraries(type(module))
        all_libraries.update(module_libraries)

    return all_libraries


def get_all_registered_custom_op_libraries() -> Set[str]:
    """Get all custom op libraries that have been registered.

    This function returns the set of all libraries that have ever been
    registered using the @uses_custom_op decorator up until now.

    Returns:
        A set of all library names that have been registered via @uses_custom_op.
    """
    return _GLOBAL_CUSTOM_OP_LIBRARIES.copy()
