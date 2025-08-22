Extension Packages
==================

Extension packages allow you to build new functionality on top of the NequIP framework, such as custom model architectures, custom data handling, or training procedures.

Getting Started
---------------

**Quick Start**: Use the `NequIP Extension Template <https://github.com/mir-group/nequip-extension-template>`__ to bootstrap your extension package with a pre-configured project structure, then follow the detailed guide below.

.. toctree::
   :maxdepth: 2

   extension_packages/getting_started
   extension_packages/data
   extension_packages/training

Example Extension Packages
---------------------------

Feel free to use the following extension packages for inspiration and reference:

- **Allegro** (`GitHub <https://github.com/mir-group/allegro>`__, `Docs <https://nequip.readthedocs.io/projects/allegro/en/latest/?badge=latest>`__, `Paper <https://www.nature.com/articles/s41467-023-36329-y>`__): Strictly local equivariant models with excellent scalability for multirank molecular dynamics simulations.
- **NequIP-LES** (`Github <https://github.com/ChengUCB/NequIP-LES>`__, `Paper <https://arxiv.org/abs/2507.14302>`__): An extension of NequIP and Allegro that adds long-range electrostatics via the Latent Ewald Summation (LES) algorithm.