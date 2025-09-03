# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from typing import Final, Callable, Optional


# NOTE: persistent modifiers are modifiers that fundamentally change the behavior of the model (same input will lead to different outputs)
# non-persistent modifiers generally refer to accelerations that should preserve similar model behavior, with the only difference being speed
_MODEL_MODIFIER_PERSISTENT_ATTR_NAME: Final[str] = (
    "_nequip_model_modifier_is_persistent"
)
_MODEL_MODIFIER_PRIVATE_ATTR_NAME: Final[str] = "_nequip_model_modifier_is_private"


def model_modifier(persistent: bool, private: Optional[bool] = None):
    """
    Mark a ``@classmethod`` of an ``nn.Module`` as a "model modifier" that can be applied by the user to modify a packaged or other loaded model on-the-fly. Model modifiers must be a ``@classmethod`` of one of the ``nn.Module`` objects in the model.

    Args:
        persistent (bool): Whether the modifier should be applied when building the model for packaging.
        private (bool, optional): Whether the modifier is private and should not be exposed in public interfaces. Defaults to None.
    """

    def decorator(func):
        assert isinstance(func, classmethod), (
            "@model_modifier must be applied after @classmethod"
        )
        assert not hasattr(func.__func__, _MODEL_MODIFIER_PERSISTENT_ATTR_NAME)

        setattr(func.__func__, _MODEL_MODIFIER_PERSISTENT_ATTR_NAME, persistent)

        if private is not None:
            setattr(func.__func__, _MODEL_MODIFIER_PRIVATE_ATTR_NAME, private)

        return func

    return decorator


def is_model_modifier(func: callable) -> bool:
    # for backwards compatibility, we use the "persistent" flag as a marker for whether the method is a model modifier
    return hasattr(func, _MODEL_MODIFIER_PERSISTENT_ATTR_NAME)


def is_persistent_model_modifier(func: callable) -> bool:
    return getattr(func, _MODEL_MODIFIER_PERSISTENT_ATTR_NAME)


def is_private_model_modifier(func: callable) -> Optional[bool]:
    # for backwards compatibility of packaged models whose modifier would not have this metadata entry,
    # we just default to making it public for convenience of clients
    # should be ok since this mechanism is not safety critical and more just a convenience for documenting modifiers
    return getattr(func, _MODEL_MODIFIER_PRIVATE_ATTR_NAME, False)


def replace_submodules(
    model: torch.nn.Module,
    target_cls: type,
    factory: Callable[[torch.nn.Module], torch.nn.Module],
) -> torch.nn.Module:
    """
    Recursively walk the children of ``model``, and whenever we see an instance of ``target_cls``, replace it (in-place) with ``factory(old_module)`` by mutating ``model._modules[name]``.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, target_cls):
            # build a brand-new one based on `factory`
            model._modules[name] = factory(child)
        else:
            # recurse down
            replace_submodules(child, target_cls, factory)
    return model
