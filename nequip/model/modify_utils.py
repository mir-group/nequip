# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.nn.model_modifier_utils import (
    is_model_modifier,
    is_persistent_model_modifier,
)

import inspect
import contextvars
import contextlib
from hydra.utils import get_method
from typing import Dict, List, Union, Any, Optional

_ONLY_APPLY_PERSISTENT = contextvars.ContextVar("_ONLY_APPLY_PERSISTENT", default=False)


@contextlib.contextmanager
def only_apply_persistent_modifiers(persistent_only: bool):
    """
    Used during `nequip-package` to only apply persistent modifiers.
    """
    global _ONLY_APPLY_PERSISTENT
    init_state = _ONLY_APPLY_PERSISTENT.get()
    assert (
        not init_state
    ), "this error implies that the `only_apply_persistent_modifiers` context manager is being nested, which is unexpected behavior"
    _ONLY_APPLY_PERSISTENT.set(persistent_only)
    try:
        yield
    finally:
        _ONLY_APPLY_PERSISTENT.set(init_state)


def get_all_modifiers(
    module: torch.nn.Module, _all_modifiers: Optional[Dict[str, callable]] = None
) -> Dict[str, callable]:
    """
    Find all model modifiers available in a model.

    Args:
        module (torch.nn.Module): The model to collect modifiers from.

    Returns:
        Dict[str, callable]: A dictionary mapping modifier names to their functions.
    """
    if _all_modifiers is None:
        _all_modifiers = {}

    for name, member in inspect.getmembers(module, predicate=inspect.ismethod):
        if is_model_modifier(member):
            if name in _all_modifiers:
                # confirm (indirectly) that these are @classmethods (bound instance methods will not be equal)
                # this ensures that having a globally unique name for each modifier does not hide differences between different copies of the same modifier hiding in a single module tree
                assert (
                    _all_modifiers[name] == member
                ), f"Found at least two non-unique modifiers with same name `{name}`: {_all_modifiers[name]:r} and {member:r}"
            _all_modifiers[name] = member

    for _, child in module.named_children():
        get_all_modifiers(child, _all_modifiers=_all_modifiers)

    return _all_modifiers


def modify(
    model: Union[Dict[str, torch.nn.Module], torch.nn.Module],
    modifiers: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]],
) -> Union[Dict[str, torch.nn.Module], torch.nn.Module]:
    """Applies a sequence of model modifier functions to a model.

    The modifiers will be applied in the specified order. Whether the order of modifiers matters depends on the specific modifiers used.

    Args:
        model (Union[Dict[str, torch.nn.Module], torch.nn.Module]): The model(s) to modify.
        modifiers (Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]): A list of modifier configurations (if `model` is a single model) or a dictionary mapping model names to lists of modifier configurations (if `model` is a dictionary).
            Each modifier configuration is a dictionary. The dictionary must contain a key "modifier" that specifies the name of the modifier function to apply as a string. All other keys in the dictionary are passed as keyword arguments to the modifier function.

    Returns:
        Union[Dict[str, torch.nn.Module], torch.nn.Module]: The modified model(s).
    """
    # check persistence
    global _ONLY_APPLY_PERSISTENT
    persistent_only: bool = _ONLY_APPLY_PERSISTENT.get()

    # build inner model if not already built
    if not isinstance(model, torch.nn.Module):
        # don't use `hydra.utils.instantiate` because it may lead to a hydra dependency during packaging
        model = model.copy()
        model_fn = get_method(model.pop("_target_"))
        model = model_fn(**model)

    def _apply_modifier(
        avail_modifiers: Dict[str, callable],
        modifier_cfg: Dict[str, Any],
        this_model: torch.nn.Module,
    ) -> None:
        modifier_cfg = modifier_cfg.copy()
        modifier_name = modifier_cfg.pop("modifier")
        if modifier_name not in avail_modifiers.keys():
            avail_names = list(avail_modifiers.keys())
            raise RuntimeError(
                f"`{modifier_name}` is not a registered model modifier. The following are registered model modifiers: {avail_names}"
            )
        modifier_fn = avail_modifiers[modifier_name]
        is_persistent = is_persistent_model_modifier(modifier_fn)
        # only skip if doing `persistent_only` and modifier is non-persistent, otherwise always apply
        if not (persistent_only and not is_persistent):
            this_model = modifier_fn(this_model, **modifier_cfg)

    if isinstance(model, torch.nn.ModuleDict):
        # because `model` is actually a `ModuleDict`, we make the modifiers flexible while keeping a simple default for the more common single-model use case
        # a single list of modifiers is given, we assume it'll be uniformly applied to everything
        if isinstance(modifiers, list):
            modifiers = {model_name: modifiers.copy() for model_name in model.keys()}
        # ^ the above allows us to use a common loop over individual sub-models and apply the relevant model-specific modifiers

        for model_name, submodel in model.items():
            avail_modifiers: Dict[str, callable] = get_all_modifiers(submodel)
            for modifier in modifiers[model_name]:
                _apply_modifier(avail_modifiers, modifier, submodel)

    elif isinstance(model, torch.nn.Module):
        assert isinstance(modifiers, list)
        avail_modifiers: Dict[str, callable] = get_all_modifiers(model)
        for modifier in modifiers:
            _apply_modifier(avail_modifiers, modifier, model)
    else:
        raise RuntimeError("Unrecognized model object found.")

    return model
