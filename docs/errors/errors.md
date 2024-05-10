Common errors and warnings
==========================

#### Unused keys

  - ```txt
    KeyError: 'The following keys in the config file were not used, did you make a typo?: optimizer_params.
    ```
    Since >=0.6.0, using `prefix_params` style subdictionaries of options is no longer supported.  Only `_kwargs` is supported, i.e. `optimizer_kwargs`. Please update your YAML configs.

#### Out-of-memory errors

  - ...with `nequip-evaluate`

    Choose a lower ``--batch-size``; while the highest value that fits in your GPU memory is good for performance,
    lowering this does *not* affect the final results (beyond numerics).

#### Other

  - Various shape errors

    Check the sanity of the shapes in your dataset.