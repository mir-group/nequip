Getting Started
===============

Extension packages allow you to build new functionality on top of the NequIP framework, such as custom model architectures, custom data handling, or training procedures.

Entry Points Registration
--------------------------

The first step for any extension package is to register with the NequIP framework's extension discovery system. Add the following to your ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."nequip.extension"]
   init_always = "your_package_name"

When the NequIP framework is imported, it automatically discovers and loads all registered extensions (see ``nequip/__init__.py``). This allows your extension to:

- Register custom data fields with :func:`~nequip.data.register_fields`
- Register custom OmegaConf resolvers
- Perform any other initialization needed for your package

Your package's ``__init__.py`` will be automatically imported whenever the NequIP framework is used, ensuring your extensions are always available.