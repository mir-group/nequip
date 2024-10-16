# FAQs and Common Errors/Warnings

## FAQs

Examples of frequently asked questions

## Commons Errors

  - Problem: Trying to run `nequip-train` as follows fails.
    ```bash
	nequip-train config.yaml
	```
	Solution: Read the [workflow docs](workflow.md) and follow hydra's command line options, e.g.
	```bash
	nequip-train -cn config.yaml
	```

  - Various shape errors

    Check the sanity of the shapes in your dataset.


## Common Warnings

### Unused keys

  - ```txt
    KeyError: 'The following keys in the config file were not used, did you make a typo?: optimizer_params.
    ```
    Since >=0.6.0, using `prefix_params` style subdictionaries of options is no longer supported.  Only `_kwargs` is supported, i.e. `optimizer_kwargs`. Please update your YAML configs.


### Other

  - Setting global configuration settings
    
    Warnings of the form
    ```txt
    Setting the GLOBAL value for ...
    ```
    `nequip` manages a number of global configuration settings of PyTorch and e3nn and correctly restores those values when a deployed model is loaded. These settings, however, must be set at a global level and thus may affect a host application;  for this reason `nequip` by default will warn whenever overriding global configuration options.  If you need to, these warnings can be silenced with `set_global_options=True`.  (Setting `set_global_options=False` is **strongly** discouraged and might lead to strange issues or incorrect numerical results.)