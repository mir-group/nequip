# Weights and Biases Sweeps

## Introduction
Weights and Biases (W&B) provides automated hyperparameter search utilities with interactive visualizations. For more details, visit the [wandb Sweeps documentation](https://docs.wandb.ai/guides/sweeps).

## Usage
W&B Sweeps require two primary files:
- `config.yaml`: a standard configuration file for the model.
- `sweep.yaml`: a file that defines the sweep configuration.

When W&B Sweeps call the `config.yaml`, they override parameters in the `config.yaml` in a manner consistent with [Hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). For clarity, it may be convenient to make sweep variables top-level using variable interpolation in the `config.yaml`.

An example of a sweep configuration file for the tutorial config ([configs/tutorial.yaml](https://github.com/mir-group/nequip/blob/main/configs/tutorial.yaml)) is provided in [misc/wandb_sweep/sweep.yaml](https://github.com/mir-group/nequip/blob/main/misc/wandb_sweep/sweep.yaml), where paths are relative to the root of the `nequip` repository. The `tutorial.yaml` is an example `config.yaml` that uses variable interpolation for W&B Sweep parameters used in the associated `sweep.yaml`.

Once a `config.yaml` and `sweep.yaml` have been written, a sweep agent can be created:

```bash
wandb sweep sweep.yaml
```

and then the sweep agent can be started:

```bash
wandb agent <sweep-ID>
