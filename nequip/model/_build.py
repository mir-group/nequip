import inspect

import torch
from lightning.pytorch.utilities.seed import isolate_rng

from nequip.nn import GraphModuleMixin, GraphModel
from nequip.utils import (
    load_callable,
    dtype_from_name,
    torch_default_dtype,
    Config,
)
from nequip.utils.config import _GLOBAL_ALL_ASKED_FOR_KEYS

import logging

logger = logging.getLogger(__name__)


default_config = dict(
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "StressForceOutput",
        "RescaleEnergyEtc",
    ],
    default_dtype="float64",
    model_dtype="float32",
)
# All default_config keys are valid / requested
_GLOBAL_ALL_ASKED_FOR_KEYS.update(default_config.keys())


def model_from_config(
    config: Config,
    initialize: bool = False,
    deploy: bool = False,
) -> GraphModuleMixin:
    """Build a model based on `config`.

    Model builders (`model_builders`) can have arguments:
     - ``config``: the config. Always present.
     - ``model``: the model produced by the previous builder. Cannot be requested by the first builder, must be requested by subsequent ones.
     - ``initialize``: whether to initialize the model
     - ``deploy``: whether the model object is for deployment / inference

    Note that this function temporarily sets ``torch.set_default_dtype()`` and as such is not thread safe.

    Args:
        config
        initialize (bool): whether ``model_builders`` should be instructed to initialize the model
        deploy (bool): whether ``model_builders`` should be told the model is for deployment / inference

    Returns:
        The build model.
    """
    if isinstance(config, dict):
        config = Config.from_dict(config)
    tmp = default_config.copy()
    tmp.update(config)
    config = tmp
    del tmp
    # Pre-process config
    assert config.get("seed", None) is not None, "Users must provide a model seed."

    # validate type names
    assert config.get("type_names", None) is not None
    assert all(
        n.isalnum() for n in config["type_names"]
    ), "Model type names must contain only alphanumeric characters"

    # average number of neighbors normalization check
    avg_num_neighbors = config.get("avg_num_neighbors", None)
    if avg_num_neighbors is None:
        logger.info(
            "Found `avg_num_neighbors`=None -- it is recommended to set `avg_num_neighbors` for normalization and better numerics during training."
        )
    else:
        logger.info(
            f"Normalization of `avg_num_neighbors`={avg_num_neighbors} will be used to build the network."
        )

    model_dtype: torch.dtype = dtype_from_name(config["model_dtype"])

    # temporarily set the default dtype and isolate rng state
    start_graph_model_builders = None
    with torch_default_dtype(model_dtype):
        with isolate_rng():
            torch.manual_seed(config["seed"])

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
    # reset to default dtype and original global random state by context manager

    # Wrap the model up
    model = GraphModel(
        model,
        type_names=config.get("type_names"),
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

            model = builder(**params)
            if not isinstance(model, GraphModel):
                raise TypeError(
                    f"Builder {builder.__name__} didn't return a GraphModel, got {type(model)} instead"
                )

    return model
