import datetime
import inspect
import logging
import sys

from logging import FileHandler, StreamHandler
from os import makedirs
from os.path import abspath, relpath, isfile, isdir
from time import time
from typing import Optional

from .config import Config


class Output:
    """Class to manage file and folders

    Args:
        run_name: unique name of the simulation
        root: the base folder where the processed data will be stored
        workdir: the path where all log files will be stored. will be updated to root/{run_name}_{timestr} if the folder already exists.
        timestr (optional): unique id to generate work folder and store the output instance. default is time stamp if not defined.
        logfile (optional): if define, an additional logger (from the root one) will be defined and write to the file
        restart (optional): if True, the append flag will be used.
        append (optional): if True, the workdir and files can be append
        screen (optional): if True, root logger print to screen
        verbose (optional): same as Logging verbose level
        force_append (optional): flag for requeue. Forcing to append everything
    """

    def __init__(
        self,
        run_name: Optional[str] = None,
        root: Optional[str] = None,
        timestr: Optional[str] = None,
        workdir: Optional[str] = None,
        logfile: Optional[str] = None,
        restart: bool = False,
        append: bool = False,
        screen: bool = False,
        verbose: str = "info",
        force_append: bool = False,
    ):

        # add screen output to the universal logger
        logger = logging.getLogger("")
        logger.setLevel(getattr(logging, verbose.upper()))

        if len(logger.handlers) == 0 and (screen or verbose.lower() == "debug"):
            logger.addHandler(logging.StreamHandler(sys.stdout))

        logging.debug("* Initialize Output")

        FORMAT = "%(message)s"
        formatter = logging.Formatter(FORMAT)
        for handler in logger.handlers:
            handler.setFormatter(fmt=formatter)

        self.restart = restart
        self.append = append
        self.force_append = force_append
        self.screen = screen
        self.verbose = verbose

        print(f"initialize, {timestr}")

        # open root folder for storing
        # if folder exists and not append, the folder name and filename will be updated
        if timestr is None:
            if force_append:
                logging.debug(f"Using default timestr for requeue")
            elif restart and not append:
                timestr = datetime.datetime.fromtimestamp(time()).strftime(
                    "%Y-%m-%d_%H:%M:%S:%f"
                )
            else:
                logging.info("Timestr is left as None")

        print(f"changed, {timestr}")

        if not force_append:
            root = set_if_none(root, f".")
            run_name = set_if_none(run_name, f"NequIP")
            workdir = set_if_none(workdir, f"{root}/{run_name}")

        print(f"workdir, {workdir}")

        assert "/" not in run_name

        # if folder exists in a non-append-mode or a fresh run
        # rename the work folder based on run name
        if (
            isdir(workdir)
            and (((restart and not append) or (not restart)))
            and not force_append
        ):
            logging.debug(f"  ...renaming workdir from {workdir} to")

            workdir = f"{root}/{run_name}_{timestr}"
            logging.debug(f"  ...{workdir}")

        makedirs(workdir, exist_ok=True)

        self.timestr = timestr
        self.run_name = run_name
        self.root = root
        self.workdir = workdir

        self.logfile = logfile
        if logfile is not None:
            self.logfile = self.open_logfile(
                file_name=logfile, screen=screen, propagate=True
            )
            logging.debug(f"  ...logfile {self.logfile} to")

    def updated_dict(self):
        return dict(
            timestr=self.timestr,
            run_name=self.run_name,
            root=self.root,
            workdir=self.workdir,
            logfile=self.logfile,
        )

    def generate_file(self, file_name: str):
        """
        only works with relative path. open a file
        """

        if file_name.startswith("/"):
            raise ValueError("filename should be a relative path file name")
        file_name = f"{self.workdir}/{file_name}"

        if isfile(file_name) and not (self.restart and self.append):
            raise RuntimeError(
                f"Tried to create file `{file_name}` but it already exists and either (1) append is disabled or (2) this run is not a restart"
            )

        logging.debug(f"  ...generate file name {file_name}")
        return file_name

    def open_logfile(
        self,
        file_name: str,
        screen: bool = False,
        propagate: bool = False,
    ):
        """open a logger with a file and screen print

        If the log file already exist and not in append mode, a new logfile with
        time string suffix will be used instead.

        Args:

        logfile (str): file name for logging
        screen (bool): if True, log to stdout as well

        Returns:
        """

        file_name = self.generate_file(file_name)

        logger = logging.getLogger(file_name)
        logger.propagate = propagate
        verbose = getattr(logging, self.verbose.upper())
        logger.setLevel(verbose)

        if len(logger.handlers) == 0:

            formatter = logging.Formatter("%(message)s")
            fh = FileHandler(file_name, mode="a" if self.append else "w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            if screen:
                ch = StreamHandler(sys.stdout)
                ch.setLevel(logging.DEBUG)
                ch.setFormatter(formatter)
                logger.addHandler(ch)

        logging.debug(f"  ...open log file {file_name}")

        return file_name

    def as_dict(self):
        d = inspect.signature(Output.__init__)
        return {
            key: getattr(self, key)
            for key in list(d.parameters.keys())
            if key not in ["self", "kwargs"]
        }

    @classmethod
    def get_output(cls, kwargs: dict = {}):

        d = inspect.signature(cls.__init__)
        _kwargs = {
            key: kwargs.get(key, None)
            for key in list(d.parameters.keys())
            if key not in ["self", "kwargs"]
        }
        return cls(**_kwargs)

    @classmethod
    def from_config(cls, config):
        c = Config.from_class(cls)
        c.update(config)
        return cls(**dict(c))


def set_if_none(x, y):
    return y if x is None else x


def path_or_None(path, relative=False):
    """return the absolute/relative path of a path

    Args:

    path (str): path of the file/folder
    relative (bool): if True, return relative path
    """

    if relative:
        return None if path is None else relpath(path)
    else:
        return None if path is None else abspath(path)
