# Contributing to NequIP

```{note}
If you want to make a major change, add a new feature, or have any uncertainty about where the change should go or how it should be designed, please reach out to discuss with us first. Details on contacting the development team can be found in `README.md`.
```

## Design Philosophy

- Prioritize **extensibility** and **configurability** in core abstractions, we can also make simpler user-interfaces when needed.
- It is preferable to add features with more code (e.g. subclassing core abstractions) than with more options (e.g. by adding new arguments to core abstractions) wherever possible.
- Correct `nequip-train` restart-ability must be preserved with any code changes or additions

## Code Standards

### Unit tests

For new features, write a **unit test** that covers it wherever possible. If it is a significant change to the training workflow, updating the integrations tests might be required.

All additions should support CUDA/GPU. If possible, please test your changes on a GPU -- the CI tests on GitHub actions do not have GPU resources available.

### Documentation

- Add **comments** for code whose purpose is not immediately obvious.

- For new classes or functions that will be exposed to users, comprehensive user-facing docstrings are a must. We follow **Google-style Python docstrings**.

- Please carefully follow correct reStructuredText markup in user-facing docstrings or reStructuredText docs and ensure that they render correctly.  Pay particular attention to whitespace, which can easily end up rendering things in blockquotes by accident in reStructuredText.

- Use correct Sphinx (and InterSphinx) references to other classes, functions, etc. that are mentioned.

- Use `.. code-block: language` directives in reStructuredText or `` ```language `` in Markdown to make sure code blocks are rendered with syntax highlighting.

- For new classes or functions that are not user-facing, docstrings and explanatory comments are strongly encouraged and will likely be asked for during code reviews.

- Added options should be documented in the docs and changes in the `CHANGELOG.md` file

### Style Enforcement

We use [`ruff`](https://docs.astral.sh/ruff/) for code formatting and linting, [`yamllint`](https://yamllint.readthedocs.io/) for YAML files, and additional hooks for file quality (trailing whitespace, end-of-file newlines, symlink validity, etc.). All tools are configured in `pyproject.toml` and `.pre-commit-config.yaml`.

#### Pre-commit hooks

All contributors should use [pre-commit](https://pre-commit.com/) to run the same checks locally that CI runs. Pre-commit automatically manages all tool installations and configurations:

```bash
pip install pre-commit
pre-commit install
```

After installation, hooks run automatically on every commit. To manually run all hooks:

```bash
pre-commit run --all-files
```

All checks are configured in `.pre-commit-config.yaml` and match what runs in CI

```{tip}
VSCode and PyCharm both have ruff extensions available for automatic formatting and linting on file save.
```

### Git Practices

PRs should generally come from feature branches.

All PRs should be onto the `develop` branch or other feature branches, and **NOT** `main` (unless they fix a significant bug with immediate impact).

On feature branches, it is preferable to [rebase](https://docs.github.com/en/get-started/using-git/about-git-rebase) wherever possible. We strive towards a clean and easily-readable commit history. Note that this does not apply to `develop` or `main` -- the commit history on these core branches are sacred.
