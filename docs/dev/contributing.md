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

We use [`ruff`](https://docs.astral.sh/ruff/) for both code formatting and linting. Ruff is configured in `pyproject.toml` with the following settings:

- Line length: 88 characters
- Selected lint rules: E, F, W, C90 (pycodestyle errors, pyflakes, warnings, and complexity)
- Ignored rules: E226, E501, E741, E743, C901, E203
- Quote style: double quotes
- Indentation: spaces

For YAML files, we use [`yamllint`](https://yamllint.readthedocs.io/) to ensure consistent formatting and catch syntax errors.

Please run the formatter and linter before you commit:

```bash
ruff check .          # Run linting
ruff format .         # Run formatting
```

You can also check formatting without making changes:

```bash
ruff format --check .
```

The formatter can be easily set up to run automatically on file save in various editors.
  
You can also use ``pre-commit install`` to install a [pre-commit](https://pre-commit.com/) hook.

```{tip}
Install `ruff` and `yamllint` to run the linter and formatter locally. VSCode and PyCharm both have ruff extensions available for automatic formatting and linting.
```

### Git Practices

PRs should generally come from feature branches.

All PRs should be onto the `develop` branch or other feature branches, and **NOT** `main` (unless they fix a significant bug with immediate impact).

On feature branches, it is preferable to [rebase](https://docs.github.com/en/get-started/using-git/about-git-rebase) wherever possible. We strive towards a clean and easily-readable commit history. Note that this does not apply to `develop` or `main` -- the commit history on these core branches are sacred.