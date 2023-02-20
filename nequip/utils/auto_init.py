from typing import Optional, Union, List
import inspect
import logging

from .config import Config, _GLOBAL_ALL_ASKED_FOR_KEYS


def instantiate_from_cls_name(
    module,
    class_name: str,
    prefix: Optional[Union[str, List[str]]] = [],
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
        raise NameError("class_name type is not defined ")

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
    prefix: Optional[Union[str, List[str]]] = [],
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

    prefix_list = [builder.__name__] if inspect.isclass(builder) else []
    if isinstance(prefix, str):
        prefix_list += [prefix]
    elif isinstance(prefix, list):
        prefix_list += prefix
    else:
        raise ValueError(f"prefix has the wrong type {type(prefix)}")

    # detect the input parameters needed from params
    config = Config.from_class(builder, remove_kwargs=remove_kwargs)

    # be strict about _kwargs keys:
    allow = config.allow_list()
    for key in allow:
        bname = key[:-7]
        if key.endswith("_kwargs") and bname not in allow:
            raise KeyError(
                f"Instantiating {builder.__name__}: found kwargs argument `{key}`, but no parameter `{bname}` for the corresponding builder. (Did you rename `{bname}` but forget to change `{bname}_kwargs`?) Either add a parameter for `{bname}` if you are trying to allow construction of a submodule, or, if `{bname}_kwargs` is just supposed to be a dictionary, rename it without `_kwargs`."
            )
    del allow

    key_mapping = {}
    if all_args is not None:
        # fetch paratemeters that directly match the name
        _keys = config.update(all_args)
        key_mapping["all"] = {k: k for k in _keys}
        # fetch paratemeters that match prefix + "_" + name
        for idx, prefix_str in enumerate(prefix_list):
            _keys = config.update_w_prefix(
                all_args,
                prefix=prefix_str,
            )
            key_mapping["all"].update(_keys)

    if optional_args is not None:
        # fetch paratemeters that directly match the name
        _keys = config.update(optional_args)
        key_mapping["optional"] = {k: k for k in _keys}
        # fetch paratemeters that match prefix + "_" + name
        for idx, prefix_str in enumerate(prefix_list):
            _keys = config.update_w_prefix(
                optional_args,
                prefix=prefix_str,
            )
            key_mapping["optional"].update(_keys)

    # for logging only, remove the overlapped keys
    if "all" in key_mapping and "optional" in key_mapping:
        key_mapping["all"] = {
            k: v
            for k, v in key_mapping["all"].items()
            if k not in key_mapping["optional"]
        }

    final_optional_args = Config.as_dict(config)

    # for nested argument, it is possible that the positional args contain unnecesary keys
    if len(parent_builders) > 0:
        _positional_args = {
            k: v for k, v in positional_args.items() if k in config.allow_list()
        }
        positional_args = _positional_args

    init_args = final_optional_args.copy()
    init_args.update(positional_args)

    # find out argument for the nested keyword
    search_keys = [key for key in init_args if key + "_kwargs" in config.allow_list()]
    for key in search_keys:
        sub_builder = init_args[key]
        if sub_builder is None:
            # if the builder is None, skip it
            continue

        if not (callable(sub_builder) or inspect.isclass(sub_builder)):
            raise ValueError(
                f"Builder for submodule `{key}` must be a callable or a class, got `{sub_builder!r}` instead."
            )

        # add double check to avoid cycle
        # only overwrite the optional argument, not the positional ones
        if (
            sub_builder not in parent_builders
            and key + "_kwargs" not in positional_args
        ):
            sub_prefix_list = [sub_builder.__name__, key]
            for prefix in prefix_list:
                sub_prefix_list = sub_prefix_list + [
                    prefix,
                    prefix + "_" + key,
                ]

            nested_km, nested_kwargs = instantiate(
                sub_builder,
                prefix=sub_prefix_list,
                positional_args=positional_args,
                optional_args=optional_args,
                all_args=all_args,
                remove_kwargs=remove_kwargs,
                return_args_only=True,
                parent_builders=[builder] + parent_builders,
            )
            # the values in kwargs get higher priority
            nested_kwargs.update(final_optional_args.get(key + "_kwargs", {}))
            final_optional_args[key + "_kwargs"] = nested_kwargs

            for t in key_mapping:
                key_mapping[t].update(
                    {key + "_kwargs." + k: v for k, v in nested_km[t].items()}
                )
        elif sub_builder in parent_builders:
            raise RuntimeError(
                f"cyclic recursion in builder {parent_builders} {sub_builder}"
            )
        elif not callable(sub_builder) and not inspect.isclass(sub_builder):
            logging.warning(f"subbuilder is not callable {sub_builder}")
        elif key + "_kwargs" in positional_args:
            logging.warning(
                f"skip searching for nested argument because {key}_kwargs are defined in positional arguments"
            )

    # remove duplicates
    for key in positional_args:
        final_optional_args.pop(key, None)
        for t in key_mapping:
            key_mapping[t].pop(key, None)

    # debug info
    if len(parent_builders) == 0:
        # ^ we only want to log or consume arguments for the "unused keys" check
        #   if this is a root-level build. For subbuilders, we don't want to log
        #   or, worse, mark keys without prefixes as consumed.
        logging.debug(
            f"{'get args for' if return_args_only else 'instantiate'} {builder.__name__}"
        )
        for t in key_mapping:
            for k, v in key_mapping[t].items():
                string = f" {t:>10s}_args :  {k:>50s}"
                # key mapping tells us how values got from the
                # users config (v) to the object being built (k)
                # thus v is by definition a valid key
                _GLOBAL_ALL_ASKED_FOR_KEYS.add(v)
                if k != v:
                    string += f" <- {v:>50s}"
                logging.debug(string)
        logging.debug(f"...{builder.__name__}_param = dict(")
        logging.debug(f"...   optional_args = {final_optional_args},")
        logging.debug(f"...   positional_args = {positional_args})")

    # Short circuit for return_args_only
    if return_args_only:
        return key_mapping, final_optional_args
    # Otherwise, actually build the thing:
    try:
        instance = builder(**positional_args, **final_optional_args)
    except Exception as e:
        raise RuntimeError(
            f"Failed to build object with prefix `{prefix}` using builder `{builder.__name__}`"
        ) from e

    return instance, final_optional_args


def get_w_prefix(
    key: List[str],
    *kwargs,
    arg_dicts: List[dict] = [],
    prefix: Optional[Union[str, List[str]]] = [],
):
    """
    act as the get function and try to search for the value key from arg_dicts
    """

    # detect the input parameters needed from params
    config = Config(config={}, allow_list=[key])

    # sort out all possible prefixes
    if isinstance(prefix, str):
        prefix_list = [prefix]
    elif isinstance(prefix, list):
        prefix_list = prefix
    else:
        raise ValueError(f"prefix is with a wrong type {type(prefix)}")

    if not isinstance(arg_dicts, list):
        arg_dicts = [arg_dicts]

    # extract all the parameters that has the pattern prefix_variable
    # debug container to record all the variable name transformation
    key_mapping = {}
    for idx, arg_dict in enumerate(arg_dicts[::-1]):
        # fetch paratemeters that directly match the name
        _keys = config.update(arg_dict)
        key_mapping[idx] = {k: k for k in _keys}
        # fetch paratemeters that match prefix + "_" + name
        for idx, prefix_str in enumerate(prefix_list):
            _keys = config.update_w_prefix(
                arg_dict,
                prefix=prefix_str,
            )
            key_mapping[idx].update(_keys)

    # for logging only, remove the overlapped keys
    num_dicts = len(arg_dicts)
    if num_dicts > 1:
        for id_dict in range(num_dicts - 1):
            higher_priority_keys = []
            for id_higher in range(id_dict + 1, num_dicts):
                higher_priority_keys += list(key_mapping[id_higher].keys())
            key_mapping[id_dict] = {
                k: v
                for k, v in key_mapping[id_dict].items()
                if k not in higher_priority_keys
            }

    # debug info
    logging.debug(f"search for {key} with prefix {prefix}")
    for t in key_mapping:
        for k, v in key_mapping[t].items():
            string = f" {str(t):>10.10}_args :  {k:>50s}"
            if k != v:
                string += f" <- {v:>50s}"
            logging.debug(string)

    return config.get(key, *kwargs)
