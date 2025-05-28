Saved Models
============

There are two main forms of saved models that can be loaded for use in training, validation, and/or testing with ``nequip-train`` or custom Python scripts.
There are **checkpoint files** (saved during ``nequip-train`` training runs) and **package files** (constructed with ``nequip-package`` and has the ``.nequip.zip`` extension).
These files can be loaded using the following ``ModelFromCheckpoint`` and ``ModelFromPackage`` model loaders.

.. important::
  The following model loaders save the paths you provide to them into the checkpoint file created for the _new_ training run. Loading that new checkpoint file, for example to restart training or package a model, will also require loading the file at the _original_ path you provided during the initial invocation of ``ModelFromCheckpoint`` or ``ModelFromPackage``.  Moving or modifying those original checkpoint or model files will cause loading the new checkpoint to fail. As a result, the following is recommended when using ``ModelFromCheckpoint`` or ``ModelFromPackage``:

    - Use absolute paths instead of relative paths.
    - Do not change the directory structure or move your files when using the model loaders.
    - Ideally, store the original checkpoint/package files somewhere that makes their association with the new training run clear to you.
    
  Be aware that iterated nested use of ``ModelFromCheckpoint`` will result in a checkpoint chaining phenomenon where loading the checkpoint at the end of the chain requires successfully loading every intermediate checkpoint file in the chain. One can break this chain if necessary by using ``nequip-package`` to convert the checkpoint file into a packaged model, and then using ``ModelFromPackage``. 

.. autofunction:: nequip.model.ModelFromCheckpoint

.. autofunction:: nequip.model.ModelFromPackage
