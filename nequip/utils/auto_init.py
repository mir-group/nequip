import inspect
import logging

from importlib import import_module
from typing import Optional

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
    prefix: str = "",
    positional_args: dict = {},
    optional_args: Optional[dict] = None,
    all_args: Optional[dict] = None,
    remove_kwargs: bool = True,
):
    """Initialize a class based on a string class name

    Args:
    module: the module to import the class, i.e. torch.optim
    class_name: the string name of the class, i.e. "CosineAnnealingWarmRestarts"
    positional_args (dict): positional arguments
    optional_args (optional, dict): optional arguments
    all_args (dict): list of all candidate parameters tha could potentially match the argument list

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
        cls_name=the_class,
        prefix=prefix,
        positional_args=positional_args,
        optional_args=optional_args,
        all_args=all_args,
        remove_kwargs=remove_kwargs,
    )


def instantiate(
    cls_name,
    prefix: str,
    positional_args: dict = {},
    optional_args: dict = None,
    all_args: dict = None,
    remove_kwargs: bool = True,
):
    """Automatic initializing class instance by matching keys in the parameter dictionary to the constructor function.

    Keys that are exactly the same, or with a 'prefix_' in all_args, optional_args will be used.
    Priority:

        all_args[key] < all_args[prefix_key] < optional_args[key] < optional_args[prefix_key] < positional_args

    Args:
        cls_name: the type of the instance
        prefix: the prefix used to address the parameter keys
        positional_args: the arguments used for input. These arguments have the top priority.
        optional_args: the second priority group to search for keys.
        all_args: the third priority group to search for keys.
        remove_kwargs: if True, ignore the kwargs argument in the init funciton
            same definition as the one in Config.from_function
    """

    # debug info
    logging.debug(f"..{cls_name.__name__} init: ")

    # detect the input parameters needed from params
    config = Config.from_class(cls_name, remove_kwargs=remove_kwargs)
    if all_args is not None:
        key1, key2, key3 = config.update_w_prefix(all_args, prefix=prefix)
        if len(key1) > 0:
            logging.debug(f"....found keys {key1} from all_args")
        if len(key2) > 0:
            logging.debug(f"....found keys {key2} from all_args with prefix {prefix}_")
        if len(key3) > 0:
            logging.debug(f"....found keys {key3} from all_args in {prefix}_params")
    if optional_args is not None:
        key1, key2, key3 = config.update_w_prefix(optional_args, prefix=prefix)
        if len(key1) > 0:
            logging.debug(f"....found keys {key1} from optional_args")
        if len(key2) > 0:
            logging.debug(
                f"....found keys {key2} from optional_args with prefix {prefix}_"
            )
        if len(key3) > 0:
            logging.debug(
                f"....found keys {key3} from optional_args in {prefix}_params"
            )

    optional_params = dict(config)

    # remove duplicates
    for key in positional_args:
        optional_params.pop(key, None)

    logging.debug(f"....{cls_name.__name__}_param = dict(")
    logging.debug(f"....   optional_params = {optional_params},")
    logging.debug(f"....   positional_params = {positional_args})")

    instance = cls_name(**positional_args, **optional_params)

    return instance, optional_params
