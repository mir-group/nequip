"""
utilities that involve file searching and operations (i.e. save/load)
"""
import logging

from os import makedirs
from os.path import isfile, isdir, dirname, realpath


def save_file(
    item, supported_formats: dict, filename: str, enforced_format: str = None
):
    """
    Save file. It can take yaml, json, pickle, json, npz and torch save
    """

    # check whether folder exist
    path = dirname(realpath(filename))
    if not isdir(path):
        logging.debug(f"save_file make dirs {path}")
        makedirs(path, exist_ok=True)

    format, filename = adjust_format_name(
        supported_formats=supported_formats,
        filename=filename,
        enforced_format=enforced_format,
    )

    if format == "json":
        import json

        with open(filename, "w+") as fout:
            json.dump(item, fout)

    elif format == "yaml":
        import yaml

        with open(filename, "w+") as fout:
            yaml.dump(item, fout)

    elif format == "torch":
        import torch

        torch.save(item, filename)

    elif format == "pickle":
        import pickle

        with open(filename, "wb") as fout:
            pickle.save(item, fout)

    elif format == "npz":
        import numpy as np

        np.savez(filename, item)

    else:
        raise NotImplementedError(
            f"Output format {format} not supported:"
            f" try from {supported_formats.keys()}"
        )

    return filename


def load_file(supported_formats: dict, filename: str, enforced_format: str = None):
    """
    Load file. Current support form
    """

    if not isfile(filename):
        raise OSError(f"file {filename} is not found")

    if enforced_format is None:
        format = match_suffix(supported_formats=supported_formats, filename=filename)

    else:
        format = enforced_format

    if format == "json":
        import json

        with open(filename) as fin:
            return json.load(fin)

    elif format == "yaml":
        import yaml

        with open(filename) as fin:
            return yaml.load(fin, Loader=yaml.Loader)

    elif format == "torch":
        import torch

        return torch.load(filename)

    elif format == "pickle":
        import pickle

        with open(filename, "rb") as fin:
            return pickle.load(item, fin)

    elif format == "npz":
        import numpy as np

        return np.load(filename, allow_pickle=True)

    else:
        raise NotImplementedError(
            f"Input format not supported:" f" try from {supported_formats.keys()}"
        )


def adjust_format_name(
    supported_formats: dict, filename: str, enforced_format: str = None
):
    """
    Recognize whether proper suffix is added to the filename.
    If not, add it and return the formatted file name

    Args:

        supported_formats (dict): list of supported formats and corresponding suffix
        filename (str): initial filename
        enforced_format (str): default format

    Returns:

        newformat (str): the chosen format
        newname (str): the adjusted filename

    """

    if enforced_format is None:
        newformat = match_suffix(supported_formats=supported_formats, filename=filename)

    else:
        newformat = enforced_format

    newname = f"{filename}"

    add_suffix = True
    suffix = supported_formats[newformat]

    if not isinstance(suffix, (set, list, tuple)):
        suffix = [suffix]

    if len(suffix) > 0:
        for suf in suffix:
            if filename.endswith(f".{suf}"):
                add_suffix = False

        if add_suffix:
            suffix = suffix[0]
            newname += f".{suffix}"

    return newformat, newname


def match_suffix(supported_formats: str, filename: str):
    """
    Recognize format based on suffix

    Args:

        supported_formats (dict): list of supported formats and corresponding suffix
        filename (str): initial filename

    Returns:

        format (str): the recognized format

    """
    for form, suffs in supported_formats.items():
        if isinstance(suffs, (set, list, tuple)):
            for suff in suffs:
                if filename.lower().endswith(f".{suff}"):
                    return form
        else:
            if filename.lower().endswith(f".{suffs}"):
                return form

    return list(supported_formats.keys())[0]
