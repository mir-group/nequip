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

  - Setting global configuration settings
    
    Warnings of the form
    ```txt
    Setting the GLOBAL value for ...
    ```
    `nequip` manages a number of global configuration settings of PyTorch and e3nn and correctly restores those values when a deployed model is loaded. These settings, however, must be set at a global level and thus may affect a host application;  for this reason `nequip` by default will warn whenever overriding global configuration options.  If you need to, these warnings can be silenced with `set_global_options=True`.  (Setting `set_global_options=False` is **strongly** discouraged and might lead to strange issues or incorrect numerical results.)