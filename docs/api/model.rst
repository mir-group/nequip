nequip.model
============

NequIP Model
------------

.. autofunction:: nequip.model.NequIPGNNModel


Saved Models
------------

There are two main forms of saved models that can be loaded for use in training, validation, and/or testing with ``nequip-train`` or custom Python scripts. There are **checkpoint files** (saved during ``nequip-train`` training runs) and **package files** (constructed with ``nequip-package`` to save the model and the code that constructs in a ``.nequip.zip`` file). These files can be loaded using the following ``ModelFromCheckpoint`` and ``ModelFromPackage`` model builders.

.. autofunction:: nequip.model.ModelFromCheckpoint

.. autofunction:: nequip.model.ModelFromPackage


