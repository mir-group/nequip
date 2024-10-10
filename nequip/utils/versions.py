import packaging.version

import torch
import e3nn
import nequip

from .git import get_commit
from .logger import RankedLogger
from typing import Tuple, Final

logger = RankedLogger(__name__, rank_zero_only=True)

_TORCH_IS_GE_1_13: Final[bool] = packaging.version.parse(
    torch.__version__
) >= packaging.version.parse("1.13.0")

_DEFAULT_VERSION_CODES = [torch, e3nn, nequip]
_DEFAULT_COMMIT_CODES = ["e3nn", "nequip"]

CODE_COMMITS_KEY = "code_commits"


def get_config_code_versions(config) -> Tuple[dict, dict]:
    code_versions = {}
    for code in _DEFAULT_VERSION_CODES:
        version = config.get(f"{code.__name__}_version", None)
        if version is not None:
            code_versions[code.__name__] = version
    code_commits = config.get(CODE_COMMITS_KEY, {})
    if len(code_commits) == 0:
        # look for the old style
        code_commits = config.get("code_versions", {})
    return code_versions, code_commits


def get_current_code_versions(config) -> Tuple[dict, dict]:
    code_versions = {}
    for code in _DEFAULT_VERSION_CODES:
        code_versions[code.__name__] = code.__version__
    code_commits = set(_DEFAULT_COMMIT_CODES)
    for builder in config.model.model_builders:
        if not isinstance(builder, str):
            continue
        builder = builder.split(".")
        if len(builder) > 1:
            # it's not a single name which is from nequip
            code_commits.add(builder[0])
    code_commits = {code: get_commit(code) for code in code_commits}
    code_commits = {k: v for k, v in code_commits.items() if v is not None}

    logger.info("{:^29}".format("Version Information"))
    for k, v in code_versions.items():
        logger.info(f"{k:^14}:{v:^14}")

    return code_versions, code_commits


def check_code_version(config):
    current_code_versions, current_code_commits = get_current_code_versions(config)
    code_versions, code_commits = get_config_code_versions(config)

    for code, version in code_versions.items():
        # we use .get just in case we recorded something in an old version we don't in a new one
        if version != current_code_versions.get(code, version):
            logger.error(
                "Loading a saved model created with different library version(s) may cause issues."
                f" Current {code} version: {current_code_versions[code]} "
                f"vs  original version: {version}"
            )

    for code, commit in code_commits.items():
        # see why .get above
        if commit != current_code_commits.get(code, commit):
            logger.error(
                "Loading a saved model created with different library git commit(s) may cause issues."
                f" Currently {code}'s git commit: {current_code_commits[code]} "
                f"vs  original commit: {commit}"
            )

    return current_code_versions, code_commits
