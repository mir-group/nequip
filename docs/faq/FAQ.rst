FAQ
===

How do I...
-----------

... continue to train a model that reached a stopping condition?
    There will be an answer here.

1. Reload the model trained with version 0.3.3 to the code in 0.4.
   check out the migration note at :ref:`migration_note`.
2. Specify my dataset for `nequip-train` and `nequip-eval`, see :ref:`_dataset_note`.

Common errors
-------------

Various shape errors
    Check the sanity of the shapes in your dataset.

Out-of-memory errors with `nequip-evaluate`
    Choose a lower ``--batch-size``; while the highest value that fits in your GPU memory is good for performance,
    lowering this does *not* affect the final results (beyond numerics).
