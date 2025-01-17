import torch
from lightning.pytorch.utilities.seed import isolate_rng

from nequip.nn.graph_model import GraphModel
from nequip.nn.compile import CompileGraphModel
from nequip.utils import (
    dtype_from_name,
    torch_default_dtype,
    conditional_torchscript_mode,
)
from nequip.utils._global_options import _get_latest_global_options

import functools
import contextvars
import contextlib

from typing import Optional

_IS_BUILDING_MODEL = contextvars.ContextVar("_IS_BUILDING_MODEL", default=False)

_OVERRIDE_COMPILE_MODE = contextvars.ContextVar("_OVERRIDE_COMPILE_MODE", default=False)
_DEFAULT_COMPILE_MODE = contextvars.ContextVar(
    "_DEFAULT_COMPILE_MODE", default="script"
)

_COMPILE_MODE_OPTIONS = {"compile", "script", None}


@contextlib.contextmanager
def override_model_compile_mode(compile_mode: Optional[str] = None):
    assert compile_mode in _COMPILE_MODE_OPTIONS
    global _OVERRIDE_COMPILE_MODE
    global _DEFAULT_COMPILE_MODE
    init_state = _OVERRIDE_COMPILE_MODE.get()
    init_mode = _DEFAULT_COMPILE_MODE.get()
    _OVERRIDE_COMPILE_MODE.set(True)
    _DEFAULT_COMPILE_MODE.set(compile_mode)
    yield
    _OVERRIDE_COMPILE_MODE.set(init_state)
    _DEFAULT_COMPILE_MODE.set(init_mode)


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
        global _DEFAULT_COMPILE_MODE

        # this means we're in an inner model, so we shouldn't apply the model builder operations, and just pass the function
        if _IS_BUILDING_MODEL.get():
            return func(*args, **kwargs)

        # this means we're in the outer model, and have to apply the model builder operations
        _IS_BUILDING_MODEL.set(True)
        try:
            model_cfg = kwargs.copy()
            # === sanity checks ===
            assert all(
                key in kwargs for key in ["seed", "model_dtype", "type_names"]
            ), "`seed`, `model_dtype`, and `type_names` are mandatory model arguments."

            if _get_latest_global_options().get("allow_tf32", False):
                assert (
                    kwargs["model_dtype"] == "float32"
                ), "`allow_tf32=True` only works with `model_dtype=float32`"

            # seed and model_dtype are removed from kwargs, so they will NOT get passed to inner models
            seed = kwargs.pop("seed")
            model_dtype = kwargs.pop("model_dtype")
            dtype = dtype_from_name(model_dtype)

            # === compilation options ===
            # `compile_mode` dictates the optimization path chosen, either `None`, `script`, or `compile`
            # users can set this with the `compile_mode` arg to the model builder
            # devs can override it with `override_model_compile_mode`

            # always pop because inner models won't need `compile_mode` arg
            compile_mode = kwargs.pop("compile_mode", _DEFAULT_COMPILE_MODE.get())
            # compile mode overriding logic
            if _OVERRIDE_COMPILE_MODE.get():
                compile_mode = _DEFAULT_COMPILE_MODE.get()
            assert (
                compile_mode in _COMPILE_MODE_OPTIONS
            ), f"`compile_mode` can only be any of {_COMPILE_MODE_OPTIONS}, but `{compile_mode}` found"
            graph_model_module = (
                CompileGraphModel if compile_mode == "compile" else GraphModel
            )

            # set torchscript mode -- True if "jit" mode
            with conditional_torchscript_mode(compile_mode == "script"):
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
