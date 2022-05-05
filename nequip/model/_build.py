import inspect
from typing import Optional

from nequip.data import AtomicDataset
from nequip.data.transforms import TypeMapper
from nequip.nn import GraphModuleMixin
from nequip.utils import load_callable, instantiate


def model_from_config(
    config, initialize: bool = False, dataset: Optional[AtomicDataset] = None
) -> GraphModuleMixin:
    """Build a model based on `config`.

    Model builders (`model_builders`) can have arguments:
     - ``config``: the config. Always present.
     - ``model``: the model produced by the previous builder. Cannot be requested by the first builder, must be requested by subsequent ones.
     - ``initialize``: whether to initialize the model
     - ``dataset``: if ``initialize`` is True, the dataset

    Args:
        config
        initialize (bool): if True (default False), ``model_initializers`` will also be run.
        dataset: dataset for initializers if ``initialize`` is True.

    Returns:
        The build model.
    """
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

    # Build
    builders = [
        load_callable(b, prefix="nequip.model")
        for b in config.get("model_builders", [])
    ]

    model = None

    for builder in builders:
        pnames = inspect.signature(builder).parameters
        params = {}
        if "initialize" in pnames:
            params["initialize"] = initialize
        if "config" in pnames:
            params["config"] = config
        if "dataset" in pnames:
            if "initialize" not in pnames:
                raise ValueError("Cannot request dataset without requesting initialize")
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

    return model
