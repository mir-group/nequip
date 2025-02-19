# Config File

The config file has five main sections -- `run`, `data`, `trainer`, `training_module`, `global_options`. These top level config entries must always be present.
Before going into what each section entails, users are advised to take note of OmegaConf's [variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) utilities, which may be a useful tool for managing runs.
Interpolation can be particularly useful when multiple locations in the config require the same values to be repeated.
It can also be used to access information like the run name or output directory of the training using [Hydra's built-in resolvers](https://hydra.cc/docs/1.3/configure_hydra/intro/#resolvers-provided-by-hydra).

## `run`

`run` allows users to specify an ordered agenda of tasks to run, of which there are four types: `train` (which requires a `train` and at least one `val` dataset), `val` (which requires `val` dataset(s)), `test` (which requires `test` dataset(s)), `predict` (which requires `predict` datasets). 

Users can specify one or more of these run types in the config. A common use mode is to perform training, followed immediately by testing (using the best model checkpoint):
```yaml
run: [train, test]
```
If you want to see how the untrained model performs on the validation and test datasets at initialization before training, train, and then assess the trained model's performance:
```yaml
run: [val, test, train, val, test]
```

## `data`

`data` is the `DataModule` object to be used. Users are directed to the [API page](../api/datamodule.rst) of `nequip.data.datamodule` for the `nequip` supported `DataModule` classes. Custom datamodules that subclass from `nequip.data.datamodule.NequIPDataModule` can also be used.


## `trainer`

The `trainer` is meant to instantiate a `lightning.Trainer` object. To understand how to configure it, users are directed to `lightning.Trainer`'s [page](https://lightning.ai/docs/pytorch/stable/common/trainer.html). The sections on trainer [flags](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags) and its [API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) are especially important.

```{tip}
It is in the `lightning.Trainer` that users can specify [callbacks](https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks) used to influence the course of training. This includes the very important [ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint) callback that should be configured to save checkpoint files in the way the user so pleases. `nequip`'s own [callbacks](../api/callbacks.rst) can also be used here.
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

`training_module` defines the `NequIPLightningModule` (or its subclasses). Users are directed to its [API page](../api/lightning_module.rst) to learn how to configure it. It is here that the following parameters are defined.
 
 ### `model`
  It is under `model` that the deep equivariant potential model is configured, which includes the NequIP message-passing graph neural network model or the strictly local Allegro model. Refer to the [model documentation page](../api/model) to learn how to configure this section.

 ### `loss` and `metrics`
  All loss components and metrics are in the physcial units associated with the dataset. Note that this behavior of the loss is different from `nequip < 0.7.0`, where the loss would have a different scale. In `nequip >= 0.7.0`, the loss components are all in physical units. For example, if the dataset uses force units of eV/Å, a force mean-squared error (MSE) would have units of (eV/Å)².
  
  The loss function and metrics are configured using `MetricsManager` objects ([API docs](../api/metrics)). In addition to the metrics configured in the `MetricsManager`, it also computes a `weighted_sum` based on the `coeff` (coefficient) for each metric. For loss functions, this quantity is used as the loss function as a weighted sum of specified loss componenets. For metrics, the weighted sum could be useful for accounting for energy-force(-stress) balancing for monitoring. For example, `val0_epoch/weighted_sum` can be monitored and used to condition the behavior of learning rate scheduling or early stopping.

 ### `optimizer` and `lr_scheduler`

  The `optimizer` can be any PyTorch-compatible optimizer. Options from PyTorch can be found [here](https://pytorch.org/docs/stable/optim.html#algorithms). The [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam) optimizer, for example, can be configured as follows: 
```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.01
```
  The `lr_scheduler` is configured according to PyTorch Lightning's `lr_scheduler_config` (see [here](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers) for the full range of options). Consider the following use of `ReduceLROnPlateau` as an example.
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
  The `scheduler` is a PyTorch-compatible learning rate scheduler. Options from PyTorch can be found [here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

## `global_options`

For now, `global_options` is used to specify
 - `seed`, the global seed (in addition to the data seed and model seed)
 - `allow_tf32`, which controls whether TensorFloat-32 is used