import inspect
from importlib import import_module

from nequip import data, datasets
from nequip.data.transforms import TypeMapper
from nequip.data import AtomicDataset
from nequip.utils import instantiate


def dataset_from_config(config, prefix: str = "dataset") -> AtomicDataset:
    """initialize database based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Examples see tests/data/test_dataset.py TestFromConfig
    and tests/datasets/test_simplest.py
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
        except Exception as e:
            # ^ TODO: don't catch all Exception
            # default class defined in nequip.data or nequip.dataset
            dataset_name = config_dataset.lower()

            class_name = None
            for k, v in inspect.getmembers(data, inspect.isclass) + inspect.getmembers(
                datasets, inspect.isclass
            ):
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
    extra_fixed_fields_key = prefix + "_extra_fixed_fields"
    if extra_fixed_fields_key not in config:
        config[extra_fixed_fields_key] = {}
        if "extra_fixed_fields" in config:
            config[extra_fixed_fields_key].update(config.extra_fixed_fields)

    if "r_max" in config and "r_max" not in config[extra_fixed_fields_key]:
        config[extra_fixed_fields_key]["r_max"] = config.r_max

    # Build a TypeMapper from the config
    type_mapper, _ = instantiate(TypeMapper, prefix=prefix, optional_args=config)

    instance, _ = instantiate(
        class_name,
        prefix=prefix,
        positional_args={"type_mapper": type_mapper},
        optional_args=config,
    )

    return instance
