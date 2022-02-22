Errors
======

Common errors
-------------

Various shape errors
    Check the sanity of the shapes in your dataset.

Out-of-memory errors with `nequip-evaluate`
    Choose a lower ``--batch-size``; while the highest value that fits in your GPU memory is good for performance,
    lowering this does *not* affect the final results (beyond numerics).
