"""
utilities that involve file searching and operations (i.e. save/load)
"""
from typing import Union, List
import sys
import logging
import contextlib
from pathlib import Path
from os import makedirs
from os.path import isfile, isdir, dirname, realpath


@contextlib.contextmanager
def _atomic_write(
    filename: Union[Path, str, List[Union[Path, str]]],
    blocking: bool = True,
    binary: bool = False,
):
    """Blockingly write a file in an atomic way.

    Ignores `blocking`.
    """
    aslist: bool = True
    if not isinstance(filename, list):
        aslist = False
        filename = [filename]
    filename = [Path(f) for f in filename]
    tmp_path = [f.parent / (f".tmp-{f.name}~") for f in filename]
    try:
        # do the IO
        with contextlib.ExitStack() as stack:
            files = [
                stack.enter_context(open(tp, "w" + ("b" if binary else "")))
                for tp in tmp_path
            ]
            if not aslist:
                yield files[0]
            else:
                yield files

        for tp, fname in zip(tmp_path, filename):
            # move the temp file to the final output path, which also removes the temp file
            tp.rename(fname)
    finally:
        # clean up
        # better for python 3.8 >
        if sys.version_info[1] >= 8:
            for tp in tmp_path:
                tp.unlink(missing_ok=True)
        else:
            # race condition?
            for tp in tmp_path:
                if tp.exists():
                    tp.unlink()


if True:  # change this to disable async IO
    import threading
    from queue import Queue
    import io

    _WRITING_THREAD = None
    _WRITING_QUEUE = Queue()

    # Because we use a queue, later writes will always (correctly)
    # overwrite earlier writes
    def _writing_thread(queue):
        while True:
            fname, binary, data = queue.get()
            with _atomic_write(fname, binary=binary) as f:
                f.write(data)
            # logging is thread safe: https://stackoverflow.com/questions/2973900/is-pythons-logging-module-thread-safe
            logging.debug(f"Finished writing {fname}")

    @contextlib.contextmanager
    def atomic_write(
        filename: Union[Path, str, List[Union[Path, str]]],
        blocking: bool = True,
        binary: bool = False,
    ):
        global _WRITING_QUEUE
        global _WRITING_THREAD
        if blocking:
            with _atomic_write(filename, binary=binary) as f:
                yield f
        else:
            aslist: bool = True
            if not isinstance(filename, list):
                aslist = False
                filename = [filename]
            # First, do the IO to a memory buffer:
            buffer = [io.BytesIO() if binary else io.StringIO() for _ in filename]
            if not aslist:
                yield buffer[0]
            else:
                yield buffer
            # Now, we have a copy of the data--
            # the main thread can keep going and do
            # whatever without affecting it
            # So we can submit it to the writing queue

            # if we don't have a writing thread, make one
            if _WRITING_THREAD is None:
                _WRITING_THREAD = threading.Thread(
                    target=_writing_thread, args=(_WRITING_QUEUE,), daemon=True
                )
                _WRITING_THREAD.start()

            if not _WRITING_THREAD.is_alive():
                _WRITING_THREAD.join()  # will raise exception
                raise RuntimeError("Writer thread failed.")

            for fname, buf in zip(filename, buffer):
                _WRITING_QUEUE.put((fname, binary, buf.getvalue()))


else:
    # Just use the blocking fallback for everything
    atomic_write = _atomic_write


def save_file(
    item,
    supported_formats: dict,
    filename: str,
    enforced_format: str = None,
    blocking: bool = True,
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

    with atomic_write(
        filename,
        blocking=blocking,
        binary={
            "json": False,
            "yaml": False,
            "pickle": True,
            "torch": True,
            "npz": True,
        }[format],
    ) as write_to:
        if format == "json":
            import json

            json.dump(item, write_to)
        elif format == "yaml":
            import yaml

            yaml.dump(item, write_to)
        elif format == "torch":
            import torch

            torch.save(item, write_to)
        elif format == "pickle":
            import pickle

            pickle.dump(item, write_to)
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
        abs_path = str(Path(filename).resolve())
        raise OSError(f"file {filename} at {abs_path} is not found")

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
