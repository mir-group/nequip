"""
utilities that involve file searching and operations (i.e. save/load)
"""
from typing import Union, List, Tuple, Optional, Callable
import sys
import logging
import contextlib
import contextvars
import tempfile
from pathlib import Path
import shutil
import os
import yaml


# accumulate writes to group for renaming
_MOVE_SET = contextvars.ContextVar("_move_set", default=None)


def _delete_files_if_exist(paths):
    # clean up
    # better for python 3.8 >
    if sys.version_info[1] >= 8:
        for f in paths:
            f.unlink(missing_ok=True)
    else:
        # race condition?
        for f in paths:
            if f.exists():
                f.unlink()


def _process_moves(moves: List[Tuple[bool, Path, Path]]):
    """blocking to copy (possibly across filesystems) to temp name; then atomic rename to final name"""
    try:
        for _, from_name, to_name in moves:
            # blocking copy to temp file in same filesystem
            tmp_path = to_name.parent / (f".tmp-{to_name.name}~")
            shutil.move(from_name, tmp_path)
            # then atomic rename to overwrite
            tmp_path.rename(to_name)
    finally:
        _delete_files_if_exist([m[1] for m in moves])


# allow user to enable/disable depending on their filesystem
_ASYNC_ENABLED = os.environ.get("NEQUIP_ASYNC_IO", "false").lower()
assert _ASYNC_ENABLED in ("true", "false")
_ASYNC_ENABLED = _ASYNC_ENABLED == "true"

if _ASYNC_ENABLED:
    import threading
    from queue import Queue

    _MOVE_QUEUE = Queue()
    _MOVE_THREAD = None

    # Because we use a queue, later writes will always (correctly)
    # overwrite earlier writes
    def _moving_thread(queue):
        while True:
            moves = queue.get()
            _process_moves(moves)
            # logging is thread safe: https://stackoverflow.com/questions/2973900/is-pythons-logging-module-thread-safe
            logging.debug(f"Finished writing {', '.join(m[2].name for m in moves)}")
            queue.task_done()

    def _submit_move(from_name, to_name, blocking: bool):
        global _MOVE_QUEUE
        global _MOVE_THREAD
        global _MOVE_SET

        # launch thread if its not running
        if _MOVE_THREAD is None:
            _MOVE_THREAD = threading.Thread(
                target=_moving_thread, args=(_MOVE_QUEUE,), daemon=True
            )
            _MOVE_THREAD.start()

        # check on health of copier thread
        if not _MOVE_THREAD.is_alive():
            _MOVE_THREAD.join()  # will raise exception
            raise RuntimeError("Writer thread failed.")

        # submit this move
        obj = (blocking, from_name, to_name)
        if _MOVE_SET.get() is None:
            # no current group
            _MOVE_QUEUE.put([obj])
            # if it should be blocking, wait for it to be processed
            if blocking:
                _MOVE_QUEUE.join()
        else:
            # add and let the group submit and block (or not)
            _MOVE_SET.get().append(obj)

    @contextlib.contextmanager
    def atomic_write_group():
        global _MOVE_SET
        if _MOVE_SET.get() is not None:
            # nesting is a no-op
            # submit along with outermost context manager
            yield
            return
        token = _MOVE_SET.set(list())
        # run the saves
        yield
        _MOVE_QUEUE.put(_MOVE_SET.get())  # send it off
        # if anyone is blocking, block the whole group:
        if any(m[0] for m in _MOVE_SET.get()):
            # someone is blocking
            _MOVE_QUEUE.join()
        # exit context
        _MOVE_SET.reset(token)

    def finish_all_writes():
        global _MOVE_QUEUE
        _MOVE_QUEUE.join()
        # ^ wait for all remaining moves to be processed

else:

    def _submit_move(from_name, to_name, blocking: bool):
        global _MOVE_SET
        obj = (blocking, from_name, to_name)
        if _MOVE_SET.get() is None:
            # no current group just do it
            _process_moves([obj])
        else:
            # add and let the group do it
            _MOVE_SET.get().append(obj)

    @contextlib.contextmanager
    def atomic_write_group():
        global _MOVE_SET
        if _MOVE_SET.get() is not None:
            # don't nest them
            yield
            return
        token = _MOVE_SET.set(list())
        yield
        _process_moves(_MOVE_SET.get())  # do it
        _MOVE_SET.reset(token)

    def finish_all_writes():
        pass  # nothing to do since all writes blocked


@contextlib.contextmanager
def atomic_write(
    filename: Union[Path, str, List[Union[Path, str]]],
    blocking: bool = True,
    binary: bool = False,
):
    aslist: bool = True
    if not isinstance(filename, list):
        aslist = False
        filename = [filename]
    filename = [Path(f) for f in filename]

    with contextlib.ExitStack() as stack:
        files = [
            stack.enter_context(
                tempfile.NamedTemporaryFile(
                    mode="w" + ("b" if binary else ""), delete=False
                )
            )
            for _ in filename
        ]
        try:
            if not aslist:
                yield files[0]
            else:
                yield files
        except:  # noqa
            # ^ noqa cause we want to delete them no matter what if there was a failure
            # only remove them if there was an error
            _delete_files_if_exist([Path(f.name) for f in files])
            raise

        for tp, fname in zip(files, filename):
            _submit_move(Path(tp.name), Path(fname), blocking=blocking)


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
    path = os.path.dirname(os.path.realpath(filename))
    if not os.path.isdir(path):
        logging.debug(f"save_file make dirs {path}")
        os.makedirs(path, exist_ok=True)

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

    if not os.path.isfile(filename):
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


def load_callable(obj: Union[str, Callable], prefix: Optional[str] = None) -> Callable:
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
