from typing import Tuple

import logging

import torch
import e3nn
import nequip

from .git import get_commit

_DEFAULT_VERSION_CODES = [torch, e3nn, nequip]
_DEFAULT_COMMIT_CODES = ["e3nn", "nequip"]

_CODE_VERSIONS_KEY = "code_versions"
_CODE_COMMITS_KEY = "code_commits"


def _get_code_versions(config) -> Tuple[dict, dict]:
    code_versions = {}
    for code in _DEFAULT_VERSION_CODES:
        code_versions[code.__name__] = code.__version__
    code_commits = set(_DEFAULT_COMMIT_CODES)
    for builder in config["model_builders"]:
        if not isinstance(builder, str):
            continue
        builder = builder.split(".")
        if len(builder) > 1:
            # it's not a single name which is from nequip
            code_commits.add(builder[0])
    code_commits = {code: get_commit(code) for code in code_commits}
    code_commits = {k: v for k, v in code_commits.items() if v is not None}
    return code_versions, code_commits


def check_code_version(config, add_to_config: bool = False):
    current_code_versions, current_code_commits = _get_code_versions(config)

    code_versions = config.get(_CODE_VERSIONS_KEY, {})
    for code, version in code_versions.items():
        # we use .get just in case we recorded something in an old version we don't in a new one
        if version != current_code_versions.get(code, version):
            logging.warning(
                "Loading a saved model created with different library version(s) may cause issues."
                f"current {code} version: {current_code_versions[code]} "
                f"vs  original version: {version}"
            )

    code_commits = config.get(_CODE_COMMITS_KEY, {})
    for code, commit in code_commits.items():
        # see why .get above
        if commit != current_code_commits.get(code, commit):
            logging.warning(
                "Loading a saved model created with different library git commit(s) may cause issues."
                f"currently {code}'s git commit: {current_code_commits[code]} "
                f"vs  original commit: {commit}"
            )

    if add_to_config:
        config[_CODE_VERSIONS_KEY] = current_code_versions
        config[_CODE_COMMITS_KEY] = current_code_commits
