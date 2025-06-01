# Config File

The config file has five main sections: `run`, `data`, `trainer`, `training_module`, `global_options`. These top level config entries must always be present.

## Variable interpolation

NequIP uses the [Hydra library](https://hydra.cc/) for configurations, which is built on top of the {mod}`omegaconf` YAML configuration library. OmegaConf offers a powerful [variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) feature, which includes special "functions" that can be called in variable interpolation expressions.  These "functions" are called ["resolvers"](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#resolvers).
Hydra provides [built-in resolvers](https://hydra.cc/docs/1.3/configure_hydra/intro/#resolvers-provided-by-hydra) that allow you to interpolate the run name or output directory into the config.

NequIP also registers a number of custom resolvers to allow users to do basic integer arithmetic directly in the config file:
- Integer multiplication: `area: ${int_mul:${width},${height}}`
- Integer division: `half_width: ${int_div:${width},2}`
These resolvers will throw errors if the inputs are not integers or if division is not exact.


## `run`

`run` allows users to specify an ordered agenda of tasks that `nequip-train` will run, of which there are three types: `train` (which requires a `train` and at least one `val` dataset), `val` (which requires one or more `val` datasets), and `test` (which requires one or more `test` datasets).

Users can specify one or more of these run types in the config. A common pattern is to perform training followed immediately by testing:
```yaml
run: [train, test]
```

```{important}
Any `val` or `test` tasks that come after `train` will use the **best** model checkpoint.
```

If you want to check how the untrained model performs on the validation and test datasets at initialization before training, train, and then assess the trained model's performance:
```yaml
run: [val, test, train, val, test]
```

```{note}
[Continuing training from a checkpoint file](./workflow.md#saving-and-restarting) will continue from the last `run` task the checkpoint file was at before stopping. For example, if one uses `run: [test, train, val, test]` and a `nequip-train` run crashed at the `train` step, a run restarted from that checkpoint will continue in the `train` stage (skipping the initial `test` stage that had already been completed in the previously crashed run).
```


## `data`

`data` defines the {class}`~nequip.data.datamodule.NequIPDataModule` object, which manages the train, validation, and test datasets.
For guidance on data configuration, see the [Data Configuration guide](data.md).
For technical API details, users are directed to the {mod}`nequip.data.datamodule` [API page](../api/datamodule.rst).


## `trainer`

The `trainer` specifies arguments to instantiate a {class}`~lightning.pytorch.trainer.trainer.Trainer` object. To understand how to configure it, users are directed to {class}`~lightning.pytorch.trainer.trainer.Trainer`. The sections on trainer [flags](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags) and its [API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) are especially important.

```{tip}
It is in the {class}`~lightning.pytorch.trainer.trainer.Trainer` that users can specify [callbacks](https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks) used to influence the course of training. This includes the very important {class}`~lightning.pytorch.callbacks.ModelCheckpoint` callback that should be configured to save checkpoint files in the way the user so pleases. `nequip`'s own [callbacks](../api/callbacks.rst) can also be used here.
```

### Logging

`nequip` supports various loggers through PyTorch Lightning, including its [built-in loggers](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers), e.g. Tensorboard, Weights & Biases, etc.

#### Tensorboard

Tensorboard can be configured, for example, as follows:
```yaml
logger:
  _target_: lightning.pytorch.loggers.TensorBoardLogger
  # The run name in tensorboard can be, for example, inherited from Hydra.
  version: ${hydra:job.name}
  # By default (not if overridden) Hydra will make `./outputs` and put various runs at `./outputs/{name}`.
  # Here we add an additional `./outputs/tensorboard_logs` within which logs will be stored _across_ runs.
  save_dir: outputs/tensorboard_logs
```
The full set of options are found in the documentation of the [underlying object from PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.tensorboard.html#module-lightning.pytorch.loggers.tensorboard).

## `training_module`

`training_module` defines the {class}`~nequip.train.NequIPLightningModule` (or its subclasses). Users are directed to its [API page](../api/lightning_module.rst) to learn how to configure it. Usually the {class}`~nequip.train.EMALightningModule` is the right choice.

The following important objects are configured as part of the `training_module`:
 
 ### `model`
  This section configures the model itself, including hyperparameters and the choice of architecture (for example, the NequIP message-passing E(3)-equivariant GNN, or the Allegro architecture). Refer to the [model documentation page](../api/model) to learn how to configure this section.

 ### `loss` and `metrics`
  Loss functions and metrics to monitor training progress are configured here in the `training_module`. See the [Loss and Metrics guide](metrics.md) for configuration details, including simplified wrappers, coefficient mechanics, and monitoring setup.

 ### `optimizer` and `lr_scheduler`

  The `optimizer` can be any PyTorch-compatible optimizer. Options from PyTorch can be found in {mod}`torch.optim`. The {class}`~torch.optim.Adam` optimizer, for example, can be configured as follows: 
```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.01
```
  The `lr_scheduler` is configured according to PyTorch Lightning's `lr_scheduler_config` (see {meth}`~lightning.pytorch.core.LightningModule.configure_optimizers` for the full range of options). Consider the following use of {class}`~torch.optim.lr_scheduler.ReduceLROnPlateau` as an example.
```yaml
lr_scheduler:
  scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    factor: 0.6
    patience: 5
    threshold: 0.2
    min_lr: 1e-6
  monitor: val0_epoch/weighted_sum
  interval: epoch
  frequency: 1
```
  The `scheduler` is a PyTorch-compatible learning rate scheduler. Options from PyTorch can be found in {mod}`torch.optim.lr_scheduler`.

## `global_options`

`global_options` is used to specify options that affect the global state of the entire `nequip-train` process. Currently, the only option is `allow_tf32` (which is `false` by default). See the [TF32 page](./tf32.md) for more details about TF32 settings.
```yaml
global_options:
  allow_tf32: false
```
