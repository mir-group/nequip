import logging
import subprocess
from pathlib import Path
from importlib import import_module


def get_commit(module: str):

    module = import_module(module)
    path = str(Path(module.__file__).parents[0] / "..")

    retcode = subprocess.run(
        "git show --oneline -s".split(),
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if retcode.returncode == 0:
        return retcode.stdout.decode().splitlines()[0].split()[0]
    else:
        err_info = retcode.stderr.decode()
        logging.info(err_info)
        logging.info(
            f"Fail to retrieve {module} commit version."
            "It is installe dat {path} but no git is found;"
            "Try `pip install -e ./` for installation."
        )
        return "NaN"
