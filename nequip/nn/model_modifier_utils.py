# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from typing import Final, Callable


_MODEL_MODIFIER_ATTR_NAME: Final[str] = "_nequip_model_modifier_is_persistent"


def model_modifier(persistent: bool):
    """
    Mark a ``@classmethod`` of an ``nn.Module`` as a "model modifier" that can be applied by the user to modify a packaged or other loaded model on-the-fly. Model modifiers must be a ``@classmethod`` of one of the ``nn.Module`` objects in the model.

    Args:
        persistent (bool): Whether the modifier should be applied when building the model for packaging.
    """

    def decorator(func):
        assert isinstance(
            func, classmethod
        ), "@model_modifier must be applied after @classmethod"
        assert not hasattr(func.__func__, _MODEL_MODIFIER_ATTR_NAME)
        setattr(func.__func__, _MODEL_MODIFIER_ATTR_NAME, persistent)
        return func

    return decorator


def is_model_modifier(func: callable) -> bool:
    return hasattr(func, _MODEL_MODIFIER_ATTR_NAME)


def is_persistent_model_modifier(func: callable) -> bool:
    return getattr(func, _MODEL_MODIFIER_ATTR_NAME)


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
