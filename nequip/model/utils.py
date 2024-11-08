import torch
from lightning.pytorch.utilities.seed import isolate_rng

from nequip.nn import GraphModel
from nequip.utils import dtype_from_name, torch_default_dtype
from nequip.utils._global_options import _get_latest_global_options

import functools
import contextvars

_IS_BUILDING_MODEL = contextvars.ContextVar("_IS_BUILDING_MODEL", default=False)


def model_builder(func):
    """Decorator for model builder functions in the ``nequip`` ecosystem.

    Builds the model based on ``seed`` and ``model_dtype``, wraps it with ``GraphModel``, and imposes the presence of the ``type_names`` argument. Implicitly, this decorator also imposes that inner models do not possess ``seed`` and ``model_dtype`` as arguments.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        global _IS_BUILDING_MODEL

        # this means we're in an inner model, so we shouldn't apply the model builder operations, and just pass the function
        if _IS_BUILDING_MODEL.get():
            return func(*args, **kwargs)

        # this means we're in the outer model, and have to apply the model builder operations
        _IS_BUILDING_MODEL.set(True)
        try:
            # === sanity checks ===
            assert all(
                key in kwargs for key in ["seed", "model_dtype", "type_names"]
            ), "`seed`, `model_dtype`, and `type_names` are mandatory arguments."

            if _get_latest_global_options().get("allow_tf32", False):
                assert (
                    kwargs["model_dtype"] == "float32"
                ), "`allow_tf32=True` only works with `model_dtype=float32`"

            # seed and model_dtype are removed from kwargs, so they will NOT get passed to inner models
            seed = kwargs.pop("seed")
            model_dtype = kwargs.pop("model_dtype")
            dtype = dtype_from_name(model_dtype)
            # set dtype and seed
            with torch_default_dtype(dtype):
                with isolate_rng():
                    torch.manual_seed(seed)
                    model = func(*args, **kwargs)
                    # wrap with GraphModel
                    graph_model = GraphModel(
                        model=model,
                        type_names=kwargs["type_names"],
                        model_dtype=dtype,
                        model_input_fields=model.irreps_in,
                    )
            return graph_model
        finally:
            # reset to default in case of failure
            _IS_BUILDING_MODEL.set(False)

    return wrapper
