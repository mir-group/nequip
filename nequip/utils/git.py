from typing import Optional

import subprocess
from pathlib import Path
from importlib import import_module


def get_commit(module: str) -> Optional[str]:

    module = import_module(module)
    path = str(Path(module.__file__).parents[0] / "..")

    retcode = subprocess.run(
        "git show --oneline --abbrev=40 -s".split(),
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if retcode.returncode == 0:
        return retcode.stdout.decode().splitlines()[0].split()[0]
    else:
        return None
