# Contributing to NequIP

Issues and pull requests are welcome!

**!! If you want to make a major change, or one whose correct location/implementation is not obvious, please reach out to discuss with us first. !!**

In general:
 - Optional additions/alternatives to how the model is built or initialized should be implemented as model builders (see `nequip.model`)
 - New model features should be implemented as new modules when possible
 - Added options should be documented in the docs and changes in the CHANGELOG.md file

Unless they fix a significant bug with immediate impact, **all PRs should be onto the `develop` branch!**

## Code style

We use the [`black`](https://black.readthedocs.io/en/stable/index.html) code formatter with default settings and the flake8 linter with settings:
```
--ignore=E226,E501,E741,E743,C901,W503,E203 --max-line-length=127
```

Please run the formatter before you commit and certainly before you make a PR. The formatter can be easily set up to run automatically on file save in various editors.
You can also use ``pre-commit install`` to install a [pre-commit](https://pre-commit.com/) hook.

You may need to install `black`, `flake8` and `Flake8-pyproject` (to read the `flake8` settings from `pyproject.toml`) to run the linter and formatter locally.

## CUDA support

All additions should support CUDA/GPU.

If possible, please test your changes on a GPU— the CI tests on GitHub actions do not have GPU resources available.