from typing import Optional, Union, Callable
import inspect
import yaml

from nequip.data import AtomicDataset
from nequip.nn import GraphModuleMixin


def _load_callable(obj: Union[str, Callable], prefix: Optional[str] = None) -> Callable:
    """Load a callable from a name, or pass through a callable."""
    if callable(obj):
        pass
    elif isinstance(obj, str):
        if "." not in obj:
            # It's an unqualified name
            if prefix is not None:
                obj = prefix + "." + obj
            else:
                # You can't have an unqualified name without a prefix
                raise ValueError(f"Cannot load unqualified name {obj}.")
        obj = yaml.load(f"!!python/name:{obj}", Loader=yaml.Loader)
    else:
        raise TypeError
    assert callable(obj), f"{obj} isn't callable"
    return obj


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
    builders = [
        _load_callable(b, prefix="nequip.model") for b in config["model_builders"]
    ]

    model = None

    for builder_i, builder in enumerate(builders):
        pnames = inspect.signature(builder).parameters
        params = {}
        if "initialize" in pnames:
            params["initialize"] = initialize
        if "config" in pnames:
            params["config"] = config
        if "dataset" in pnames:
            if "initialize" not in pnames:
                raise ValueError("Cannot request dataset without requesting initialize")
            if initialize and dataset is None:
                raise RuntimeError(
                    f"Builder {builder.__name__} asked for the dataset, initialize is true, but no dataset was provided to `model_from_config`."
                )
            params["dataset"] = dataset
        if "model" in pnames:
            if builder_i == 0:
                raise RuntimeError(
                    f"Builder {builder.__name__} asked for the model as an input, but it's the first builder so there is no model to provide"
                )
            params["model"] = model
        else:
            if builder_i > 0:
                raise RuntimeError(
                    f"All model_builders but the first one must take the model as an argument; {builder.__name__} doesn't"
                )
        model = builder(**params)
        if not isinstance(model, GraphModuleMixin):
            raise TypeError(
                f"Builder {builder.__name__} didn't return a GraphModuleMixin, got {type(model)} instead"
            )

    return model
