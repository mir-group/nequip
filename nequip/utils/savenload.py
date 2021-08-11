"""
utilities that involve file searching and operations (i.e. save/load)
"""
from typing import Union
import sys
import logging
import contextlib
from pathlib import Path
from os import makedirs
from os.path import isfile, isdir, dirname, realpath


@contextlib.contextmanager
def atomic_write(filename: Union[Path, str]):
    filename = Path(filename)
    tmp_path = filename.parent / (f".tmp-{filename.name}~")
    # Create the temp file
    open(tmp_path, "w").close()
    try:
        # do the IO
        yield tmp_path
        # move the temp file to the final output path, which also removes the temp file
        tmp_path.rename(filename)
    finally:
        # clean up
        # better for python 3.8 >
        if sys.version_info[1] >= 8:
            tmp_path.unlink(missing_ok=True)
        else:
            # race condition?
            if tmp_path.exists():
                tmp_path.unlink()


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

    with atomic_write(filename) as write_to:
        if format == "json":
            import json

            with open(write_to, "w+") as fout:
                json.dump(item, fout)
        elif format == "yaml":
            import yaml

            with open(write_to, "w+") as fout:
                yaml.dump(item, fout)
        elif format == "torch":
            import torch

            torch.save(item, write_to)
        elif format == "pickle":
            import pickle

            with open(write_to, "wb") as fout:
                pickle.save(item, fout)
        elif format == "npz":
            import numpy as np

            np.savez(write_to, item)
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
    if enforced_format is None:
        format = match_suffix(supported_formats=supported_formats, filename=filename)
    else:
        format = enforced_format

    if not isfile(filename):
        raise OSError(f"file {filename} is not found")

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
            return pickle.load(fin)
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
