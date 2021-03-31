import inspect
import logging

from importlib import import_module
from typing import Optional, Union, List

from nequip import data, datasets
from .config import Config


def dataset_from_config(config):
    """initialize database based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Examples see tests/data/test_dataset.py TestFromConfig
    and tests/datasets/test_simplest.py
    """

    if inspect.isclass(config.dataset):
        # user define class
        class_name = config.dataset
    else:
        try:
            module_name = ".".join(config.dataset.split(".")[:-1])
            class_name = ".".join(config.dataset.split(".")[-1:])
            class_name = getattr(import_module(module_name), class_name)
        except Exception as e:
            # default class defined in nequip.data or nequip.dataset
            dataset_name = config.dataset.lower()

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
        raise NameError(f"dataset {dataset_name} does not exists")

    # if dataset r_max is not found, use the universal r_max
    if "dataset_extra_fixed_fields" not in config:
        config.dataset_extra_fixed_fields = {}
        if "extra_fixed_fields" in config:
            config.dataset_extra_fixed_fields.update(config.extra_fixed_fields)

    if "r_max" in config and "r_max" not in config.dataset_extra_fixed_fields:
        config.dataset_extra_fixed_fields["r_max"] = config.r_max

    instance, _ = instantiate(class_name, prefix="dataset", optional_args=dict(config))

    return instance


def instantiate_from_cls_name(
    module,
    class_name: str,
    prefix: Union[str, List[str]] = "",
    positional_args: dict = {},
    optional_args: Optional[dict] = None,
    all_args: Optional[dict] = None,
    remove_kwargs: bool = True,
    return_args_only: bool = False,
):
    """Initialize a class based on a string class name

    Args:
    module: the module to import the class, i.e. torch.optim
    class_name: the string name of the class, i.e. "CosineAnnealingWarmRestarts"
    positional_args (dict): positional arguments
    optional_args (optional, dict): optional arguments
    all_args (dict): list of all candidate parameters tha could potentially match the argument list
    remove_kwargs: if True, ignore the kwargs argument in the init funciton
        same definition as the one in Config.from_function
    return_args_only (bool): if True, do not instantiate, only return the arguments

    Returns:

    instance: the instance
    optional_args (dict):
    """

    if class_name is None:
        raise NameError(f"class_name type is not defined ")

    # first obtain a list of all classes in this module
    class_list = inspect.getmembers(module, inspect.isclass)
    class_dict = {}
    for k, v in class_list:
        class_dict[k] = v

    # find the matching class
    the_class = class_dict.get(class_name, None)
    if the_class is None:
        raise NameError(f"{class_name} type is not found in {module.__name__} module")

    return instantiate(
        builder=the_class,
        prefix=prefix,
        positional_args=positional_args,
        optional_args=optional_args,
        all_args=all_args,
        remove_kwargs=remove_kwargs,
        return_args_only=return_args_only,
    )


def instantiate(
    builder,
    prefix: Union[str, List[str]],
    positional_args: dict = {},
    optional_args: dict = None,
    all_args: dict = None,
    remove_kwargs: bool = True,
    return_args_only: bool = False,
    parent_builders: list = [],
):
    """Automatic initializing class instance by matching keys in the parameter dictionary to the constructor function.

    Keys that are exactly the same, or with a 'prefix_' in all_args, optional_args will be used.
    Priority:

        all_args[key] < all_args[prefix_key] < optional_args[key] < optional_args[prefix_key] < positional_args

    Args:
        builder: the type of the instance
        prefix: the prefix used to address the parameter keys
        positional_args: the arguments used for input. These arguments have the top priority.
        optional_args: the second priority group to search for keys.
        all_args: the third priority group to search for keys.
        remove_kwargs: if True, ignore the kwargs argument in the init funciton
            same definition as the one in Config.from_function
        return_args_only (bool): if True, do not instantiate, only return the arguments
    """

    if isinstance(prefix, str):
        prefix_list = [prefix]
    else:
        prefix_list = prefix

    # detect the input parameters needed from params
    config = Config.from_class(builder, remove_kwargs=remove_kwargs)

    keys = {}
    if all_args is not None:
        # fetch paratemeters that directly match the name
        _keys = config.update(all_args)
        keys["all"] = {k: k for k in _keys}
        for idx, prefix_str in enumerate(prefix_list):
            # fetch paratemeters that match prefix + "_" + name
            _keys = config.update_w_prefix(
                all_args,
                prefix=prefix_str,
            )
            keys["all"].update(_keys)

    if optional_args is not None:
        _keys = config.update(optional_args)
        keys["optional"] = {k: k for k in _keys}
        for idx, prefix_str in enumerate(prefix_list):
            _keys = config.update_w_prefix(
                optional_args,
                prefix=prefix_str,
            )
            keys["optional"].update(_keys)

    # for logging only, remove the overlapped keys
    if "all" in keys and "optional" in keys:
        keys["all"] = {
            k: v for k, v in keys["all"].items() if k not in keys["optional"]
        }

    # remove duplicates
    for key in positional_args:
        config.pop(key, None)
        for t in keys:
            keys[t].pop(key, None)

    # update
    optional_args = dict(config)
    init_args = dict(**positional_args, **dict(optional_args))

    # find out argument for the nested keyword
    search_keys = [key for key in init_args if key + "_kwargs" in config.allow_list()]
    for key in search_keys:
        sub_builder = init_args[key]
        # add double check to avoid cycle
        # only overwrite the optional argument, not the positional ones
        if (
            sub_builder not in parent_builders
            and key + "_kwargs" not in positional_args
        ):
            nested_kwargs = instantiate(
                sub_builder,
                prefix=[
                    sub_builder.__name__,
                    key,
                    prefix + "_" + key,
                    prefix,
                ],
                optional_args=optional_args,
                all_args=all_args,
                remove_kwargs=True,
                return_args_only=True,
                parent_builders=[builder] + parent_builders,
            )
            # the values in kwargs get higher priority
            nested_kwargs.update(optional_args.get(key + "_kwargs", {}))
            optional_args[key + "_kwargs"] = nested_kwargs

    # debug info
    logging.debug(f"instantiate {builder.__name__}")
    for t in keys:
        for k, v in keys[t].items():
            string = f" {t:>10s}_args :  {k:>30s}"
            if k != v:
                string += f" <- {v:>30s}"
            logging.debug(string)
    logging.debug(f"...{builder.__name__}_param = dict(")
    logging.debug(f"...   optional_args = {optional_args},")
    logging.debug(f"...   positional_args = {positional_args})")

    if return_args_only:
        return dict(**positional_args, **dict(optional_args))

    instance = builder(**positional_args, **optional_args)

    return instance, optional_args
