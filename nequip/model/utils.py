# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from lightning.pytorch.utilities.seed import isolate_rng

from nequip.nn.graph_model import GraphModel
from nequip.nn.compile import CompileGraphModel
from nequip.utils import (
    dtype_from_name,
    torch_default_dtype,
    conditional_torchscript_mode,
)
from nequip.utils.global_state import (
    global_state_initialized,
    get_latest_global_state,
    TF32_KEY,
)

import functools
import contextvars
import contextlib

from typing import Optional, Final

_IS_BUILDING_MODEL = contextvars.ContextVar("_IS_BUILDING_MODEL", default=False)

# the following is the set of model build types for specific purposes
_EAGER_MODEL_KEY = "eager"
_TRAIN_TIME_SCRIPT_KEY: Final[str] = "script"
_TRAIN_TIME_COMPILE_KEY: Final[str] = "compile"
_COMPILE_TIME_AOTINDUCTOR_KEY: Final[str] = "aotinductor"

_COMPILE_MODE_OPTIONS = {
    _EAGER_MODEL_KEY,
    _TRAIN_TIME_SCRIPT_KEY,
    _TRAIN_TIME_COMPILE_KEY,
    _COMPILE_TIME_AOTINDUCTOR_KEY,
}


_OVERRIDE_COMPILE_MODE = contextvars.ContextVar("_OVERRIDE_COMPILE_MODE", default=False)
_CURRENT_COMPILE_MODE = contextvars.ContextVar(
    "_CURRENT_COMPILE_MODE", default=_TRAIN_TIME_SCRIPT_KEY
)


@contextlib.contextmanager
def override_model_compile_mode(compile_mode: Optional[str]):
    """
    Overrides the ``compile_mode`` for model building.
    If several of these context managers are nested, the outermost one will be prioritized while the inner ones are ignored.
    The intended client is `ModelFromCheckpoint`.
    Anybody using this function should be warned that the behavior is designed for loading models from checkpoints and packages correctly.
    """
    assert compile_mode in _COMPILE_MODE_OPTIONS
    global _OVERRIDE_COMPILE_MODE
    global _CURRENT_COMPILE_MODE
    init_state = _OVERRIDE_COMPILE_MODE.get()
    # in the case of nested overrides, we prioritize the outermost context manager
    if init_state:
        yield
    else:
        init_mode = _CURRENT_COMPILE_MODE.get()
        _OVERRIDE_COMPILE_MODE.set(True)
        _CURRENT_COMPILE_MODE.set(compile_mode)
        try:
            yield
        finally:
            _OVERRIDE_COMPILE_MODE.set(init_state)
            _CURRENT_COMPILE_MODE.set(init_mode)


def get_current_compile_mode(return_override: bool = False):
    # returns tuple of (whether compile mode is overriden, compile mode)
    global _CURRENT_COMPILE_MODE
    if return_override:
        global _OVERRIDE_COMPILE_MODE
        return _CURRENT_COMPILE_MODE.get(), _OVERRIDE_COMPILE_MODE.get()
    else:
        return _CURRENT_COMPILE_MODE.get()


def model_builder(func):
    """Decorator for model builder functions in the ``nequip`` ecosystem.

    Builds the model based on ``seed`` and ``model_dtype``, wraps it with ``GraphModel``, and imposes the presence of the ``type_names`` argument. Implicitly, this decorator also imposes that inner models do not possess ``seed`` and ``model_dtype`` as arguments.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # to handle nested model building
        global _IS_BUILDING_MODEL

        # to handle compile modes
        global _OVERRIDE_COMPILE_MODE
        global _CURRENT_COMPILE_MODE

        # this means we're in an inner model, so we shouldn't apply the model builder operations, and just pass the function
        if _IS_BUILDING_MODEL.get():
            return func(*args, **kwargs)

        # this means we're in the outer model, and have to apply the model builder operations
        _IS_BUILDING_MODEL.set(True)
        try:
            model_cfg = kwargs.copy()
            # === sanity checks ===
            assert (
                global_state_initialized()
            ), "global state must be initialized before building models"
            assert all(
                key in kwargs for key in ["seed", "model_dtype", "type_names"]
            ), "`seed`, `model_dtype`, and `type_names` are mandatory model arguments."

            if get_latest_global_state().get(TF32_KEY, False):
                assert (
                    kwargs["model_dtype"] == "float32"
                ), "`allow_tf32=True` only works with `model_dtype=float32`"

            # seed and model_dtype are removed from kwargs, so they will NOT get passed to inner models
            seed = kwargs.pop("seed")
            model_dtype = kwargs.pop("model_dtype")
            dtype = dtype_from_name(model_dtype)

            # === compilation options ===
            # `compile_mode` dictates the optimization path chosen
            # users can set this with the `compile_mode` arg to the model builder
            # devs can override it with `override_model_compile_mode`

            # always pop because inner models won't need `compile_mode` arg
            compile_mode = kwargs.pop("compile_mode", _CURRENT_COMPILE_MODE.get())
            # compile mode overriding logic
            if _OVERRIDE_COMPILE_MODE.get():
                compile_mode = _CURRENT_COMPILE_MODE.get()
            assert (
                compile_mode in _COMPILE_MODE_OPTIONS
            ), f"`compile_mode` can only be any of {_COMPILE_MODE_OPTIONS}, but `{compile_mode}` found"

            # use `CompileGraphModel` if doing train-time compile build
            if compile_mode == _TRAIN_TIME_COMPILE_KEY:
                # === torch version check ===
                from nequip.utils.versions import check_pt2_compile_compatibility

                check_pt2_compile_compatibility()
                graph_model_module = CompileGraphModel
            else:
                graph_model_module = GraphModel

            # set torchscript mode -- True if "jit" mode
            with conditional_torchscript_mode(compile_mode == _TRAIN_TIME_SCRIPT_KEY):
                # set dtype and seed
                with torch_default_dtype(dtype):
                    with isolate_rng():
                        torch.manual_seed(seed)
                        model = func(*args, **kwargs)
                        # wrap with GraphModel
                        graph_model = graph_model_module(
                            model=model,
                            model_config=model_cfg,
                            model_input_fields=model.irreps_in,
                        )
            return graph_model
        finally:
            # reset to default in case of failure
            _IS_BUILDING_MODEL.set(False)

    return wrapper
