import inspect
from typing import Optional

import torch

from nequip.data import AtomicDataset
from nequip.data.transforms import TypeMapper
from nequip.nn import GraphModuleMixin, GraphModel
from nequip.utils import (
    load_callable,
    instantiate,
    dtype_from_name,
    torch_default_dtype,
    Config,
)
from nequip.utils.config import _GLOBAL_ALL_ASKED_FOR_KEYS


def model_from_config(
    config: Config,
    initialize: bool = False,
    dataset: Optional[AtomicDataset] = None,
    deploy: bool = False,
) -> GraphModuleMixin:
    """Build a model based on `config`.

    Model builders (`model_builders`) can have arguments:
     - ``config``: the config. Always present.
     - ``model``: the model produced by the previous builder. Cannot be requested by the first builder, must be requested by subsequent ones.
     - ``initialize``: whether to initialize the model
     - ``dataset``: if ``initialize`` is True, the dataset
     - ``deploy``: whether the model object is for deployment / inference

    Note that this function temporarily sets ``torch.set_default_dtype()`` and as such is not thread safe.

    Args:
        config
        initialize (bool): whether ``model_builders`` should be instructed to initialize the model
        dataset: dataset for initializers if ``initialize`` is True.
        deploy (bool): whether ``model_builders`` should be told the model is for deployment / inference

    Returns:
        The build model.
    """
    if isinstance(config, dict):
        config = Config.from_dict(config)
    # Pre-process config
    type_mapper = None
    if dataset is not None:
        type_mapper = dataset.type_mapper
    else:
        try:
            type_mapper, _ = instantiate(TypeMapper, all_args=config)
        except RuntimeError:
            pass

    if type_mapper is not None:
        if "num_types" in config:
            assert (
                config["num_types"] == type_mapper.num_types
            ), "inconsistant config & dataset"
        if "type_names" in config:
            assert (
                config["type_names"] == type_mapper.type_names
            ), "inconsistant config & dataset"
        config["num_types"] = type_mapper.num_types
        config["type_names"] = type_mapper.type_names
        config["type_to_chemical_symbol"] = type_mapper.type_to_chemical_symbol
        # We added them, so they are by definition valid:
        _GLOBAL_ALL_ASKED_FOR_KEYS.update(
            {"num_types", "type_names", "type_to_chemical_symbol"}
        )

    default_dtype = torch.get_default_dtype()
    model_dtype: torch.dtype = dtype_from_name(config.get("model_dtype", default_dtype))
    config["model_dtype"] = str(model_dtype).lstrip("torch.")
    # confirm sanity
    assert default_dtype in (torch.float32, torch.float64)
    if default_dtype == torch.float32 and model_dtype == torch.float64:
        raise ValueError(
            "Overall default_dtype=float32, but model_dtype=float64 is a higher precision- change default_dtype to float64"
        )
    # temporarily set the default dtype
    start_graph_model_builders = None
    with torch_default_dtype(model_dtype):

        # Build
        builders = [
            load_callable(b, prefix="nequip.model")
            for b in config.get("model_builders", [])
        ]

        model = None

        for builder_i, builder in enumerate(builders):
            pnames = inspect.signature(builder).parameters
            params = {}
            if "graph_model" in pnames:
                # start graph_model builders, which happen later
                start_graph_model_builders = builder_i
                break
            if "initialize" in pnames:
                params["initialize"] = initialize
            if "deploy" in pnames:
                params["deploy"] = deploy
            if "config" in pnames:
                params["config"] = config
            if "dataset" in pnames:
                if "initialize" not in pnames:
                    raise ValueError(
                        "Cannot request dataset without requesting initialize"
                    )
                if (
                    initialize
                    and pnames["dataset"].default == inspect.Parameter.empty
                    and dataset is None
                ):
                    raise RuntimeError(
                        f"Builder {builder.__name__} requires the dataset, initialize is true, but no dataset was provided to `model_from_config`."
                    )
                params["dataset"] = dataset
            if "model" in pnames:
                if model is None:
                    raise RuntimeError(
                        f"Builder {builder.__name__} asked for the model as an input, but no previous builder has returned a model"
                    )
                params["model"] = model
            else:
                if model is not None:
                    raise RuntimeError(
                        f"All model_builders after the first one that returns a model must take the model as an argument; {builder.__name__} doesn't"
                    )
            model = builder(**params)
            if model is not None and not isinstance(model, GraphModuleMixin):
                raise TypeError(
                    f"Builder {builder.__name__} didn't return a GraphModuleMixin, got {type(model)} instead"
                )
    # reset to default dtype by context manager

    # Wrap the model up
    model = GraphModel(
        model,
        model_dtype=model_dtype,
        model_input_fields=config.get("model_input_fields", {}),
    )

    # Run GraphModel builders
    if start_graph_model_builders is not None:
        for builder in builders[start_graph_model_builders:]:
            pnames = inspect.signature(builder).parameters
            params = {}
            assert "graph_model" in pnames
            params["graph_model"] = model
            if "model" in pnames:
                raise ValueError(
                    f"Once any builder requests `graph_model` (first requested by {builders[start_graph_model_builders].__name__}), no builder can request `model`, but {builder.__name__} did"
                )
            if "initialize" in pnames:
                params["initialize"] = initialize
            if "deploy" in pnames:
                params["deploy"] = deploy
            if "config" in pnames:
                params["config"] = config
            if "dataset" in pnames:
                if "initialize" not in pnames:
                    raise ValueError(
                        "Cannot request dataset without requesting initialize"
                    )
                if (
                    initialize
                    and pnames["dataset"].default == inspect.Parameter.empty
                    and dataset is None
                ):
                    raise RuntimeError(
                        f"Builder {builder.__name__} requires the dataset, initialize is true, but no dataset was provided to `model_from_config`."
                    )
                params["dataset"] = dataset

            model = builder(**params)
            if not isinstance(model, GraphModel):
                raise TypeError(
                    f"Builder {builder.__name__} didn't return a GraphModel, got {type(model)} instead"
                )

    return model
