from typing import Optional

import subprocess
from pathlib import Path
from importlib import import_module


def get_commit(module: str) -> Optional[str]:

    module = import_module(module)
    package = Path(module.__file__).parents[0]
    if package.is_file():
        # We're installed as a ZIP .egg file,
        # which means there's no git information
        # and looking for the parent would fail anyway
        # https://github.com/mir-group/nequip/issues/264
        return None
    path = str(package / "..")

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
