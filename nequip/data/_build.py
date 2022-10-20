import inspect
from importlib import import_module

from nequip import data
from nequip.data.transforms import TypeMapper
from nequip.data import AtomicDataset, register_fields
from nequip.utils import instantiate, get_w_prefix, load_callable


def dataset_from_config(config, prefix: str = "dataset") -> AtomicDataset:
    """initialize database based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Examples see tests/data/test_dataset.py TestFromConfig
    and tests/datasets/test_simplest.py

    Note: this function can cause arbitrary extension packages, including those listed
          in ``model_builders``, to be imported / loaded.

    Args:
        config (dict, nequip.utils.Config): dict/object that store all the parameters
        prefix (str): Optional. The prefix of all dataset parameters

    Return:
        dataset (nequip.data.AtomicDataset)
    """

    config_dataset = config.get(prefix, None)
    if config_dataset is None:
        raise KeyError(f"Dataset with prefix `{prefix}` isn't present in this config!")

    if inspect.isclass(config_dataset):
        # user define class
        class_name = config_dataset
    else:
        try:
            module_name = ".".join(config_dataset.split(".")[:-1])
            class_name = ".".join(config_dataset.split(".")[-1:])
            class_name = getattr(import_module(module_name), class_name)
        except Exception:
            # ^ TODO: don't catch all Exception
            # default class defined in nequip.data or nequip.dataset
            dataset_name = config_dataset.lower()

            class_name = None
            for k, v in inspect.getmembers(data, inspect.isclass):
                if k.endswith("Dataset"):
                    if k.lower() == dataset_name:
                        class_name = v
                    if k[:-7].lower() == dataset_name:
                        class_name = v
                elif k.lower() == dataset_name:
                    class_name = v

    if class_name is None:
        raise NameError(f"dataset type {dataset_name} does not exists")

    # if dataset r_max is not found, use the universal r_max
    eff_key = "extra_fixed_fields"
    prefixed_eff_key = f"{prefix}_{eff_key}"
    config[prefixed_eff_key] = get_w_prefix(
        eff_key, {}, prefix=prefix, arg_dicts=config
    )
    config[prefixed_eff_key]["r_max"] = get_w_prefix(
        "r_max",
        prefix=prefix,
        arg_dicts=[config[prefixed_eff_key], config],
    )

    # Build a TypeMapper from the config
    type_mapper, _ = instantiate(TypeMapper, prefix=prefix, optional_args=config)

    # Register fields:
    # This might reregister fields, but that's OK:
    instantiate(register_fields, all_args=config)

    # We also want to make sure fields get registered for any models that are getting built
    # with any custom model builders from extension packages...
    # This will load them, which should trigger imports and relevant calls to `register_fields`
    # this is a weird hack with side effects but it shouldn't be too bad and is better than
    # writing everything up with entry points
    # taken from model/_build.py
    for b in config.get("model_builders", []):
        load_callable(b, prefix="nequip.model")

    instance, _ = instantiate(
        class_name,
        prefix=prefix,
        positional_args={"type_mapper": type_mapper},
        optional_args=config,
    )

    return instance
