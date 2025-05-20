# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from lightning.pytorch.utilities.seed import isolate_rng

from nequip.nn.graph_model import GraphModel
from nequip.nn.compile import CompileGraphModel
from nequip.data import AtomicDataDict
from nequip.utils import (
    dtype_from_name,
    torch_default_dtype,
    floating_point_tolerance,
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

from typing import Optional, Final, Callable, Union, List

_IS_BUILDING_MODEL = contextvars.ContextVar("_IS_BUILDING_MODEL", default=False)

# the following is the set of model build types for specific purposes
_EAGER_MODEL_KEY = "eager"
_TRAIN_TIME_COMPILE_KEY: Final[str] = "compile"

_COMPILE_MODE_OPTIONS = {
    _EAGER_MODEL_KEY,
    _TRAIN_TIME_COMPILE_KEY,
}


_OVERRIDE_COMPILE_MODE = contextvars.ContextVar("_OVERRIDE_COMPILE_MODE", default=False)
_CURRENT_COMPILE_MODE = contextvars.ContextVar(
    "_CURRENT_COMPILE_MODE", default=_EAGER_MODEL_KEY
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

            # never script
            with conditional_torchscript_mode(False):
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


def _pt2_compile_error_message(key, tol, err, absval, model_dtype):
    return f"Compilation check MaxAbsError: {err:.6g} (tol: {tol}) for field `{key}`. This assert was triggered because the outputs of an eager model and a compiled model numerically differ above the specified tolerance. This may indicate a compilation error (bug) specific to certain machines and installation environments, or may be an artefact of poor initialization if the error is close to the tolerance. Note that the largest absolute (MaxAbs) entry of the model prediction is {absval:.6g} -- you can use this detail to discern if it is a numerical artefact (the errors could be large because the MaxAbs value is very large) or a more fundamental compilation error. Raise a GitHub issue if you believe it is a compilation error or are unsure. If you are confident that it is purely numerical, and want to bypass the tolerance check, you may set the following environment variables: `NEQUIP_FLOAT64_MODEL_TOL`, `NEQUIP_FLOAT32_MODEL_TOL`, or `NEQUIP_TF32_MODEL_TOL`, depending on the model dtype you are using (which is currently {model_dtype}) and whether TF32 is on."


def _default_error_message(key, tol, err, absval, model_dtype):
    return f"MaxAbsError: {err:.6g} (tol: {tol} for {model_dtype} model) for field `{key}`. MaxAbs value: {absval:.6g}."

# for `test_model_output_similarity`, we perform evaluation `_NUM_EVAL_TRIALS` times to account for numerical randomness in the model
_NUM_EVAL_TRIALS = 5

def test_model_output_similarity_by_dtype(
    model1: Callable,
    model2: Callable,
    data: AtomicDataDict.Type,
    model_dtype: Union[str, torch.dtype],
    fields: Optional[List[str]] = None,
    error_message: Callable = _default_error_message,
):
    """
    Assumptions and behavior:
    - `model1` and `model2` have signature `AtomicDataDict -> AtomicDataDict`
    - if `fields` are not provided, the function will loop over `model1`'s output keys
    """
    tol = floating_point_tolerance(model_dtype)

    # do one evaluation to figure out the fields if not provided
    if fields is None:
        fields = model1(data.copy()).keys()

    # perform `_NUM_EVAL_TRIALS` evaluations with each model and average to account for numerical randomness
    out1_list, out2_list = {k: [] for k in fields}, {k: [] for k in fields}
    for _ in range(_NUM_EVAL_TRIALS):
        out1 = model1(data.copy())
        out2 = model2(data.copy())
        for k in fields:
            out1_list[k].append(out1[k].detach().double())
            out2_list[k].append(out2[k].detach().double())
        del out1, out2

    for k in fields:
        t1, t2 = (
            torch.mean(torch.stack(out1_list[k], -1), -1),
            torch.mean(torch.stack(out2_list[k], -1), -1),
        )
        err = torch.max(torch.abs(t1 - t2)).item()
        absval = t1.abs().max().item()

        assert torch.allclose(t1, t2, atol=tol, rtol=tol), error_message(
            k, tol, err, absval, model_dtype
        )

        del t1, t2, err, absval