# Config

The config file has five main sections -- `run`, `data`, `trainer`, `training_module`, `global_options`. These top level config entries must always be present. Before going into what each section entails, users are advised to take note of OmegaConf's [variable interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation) utilities, which may be a useful tool for managing configuring training and testing runs.

## `run`

`run` allows users to specify the run types, of which there are four -- `train` (which requires a `train` and at least one `val` dataset), `val` (which requires `val` dataset(s)), `test` (which requires `test` dataset(s)), `predict` (which requires `predict` datasets). 

Users can specify one or more of these run types in the config. A common use mode is to perform training, followed immediately by testing (using the best model checkpoint).
```yaml
run: [train, test]
```
If one seeks to check how the untrained model performs on the validation and test datasets before training, then assess the trained model's performance, one can use the following.
```yaml
run: [val, test, train, val, test]
```

> **_NOTE:_**  the `test` run type is the replacement for the role `nequip-evaluate` has in the pre-`0.7.0` `nequip` package.

## `data`

TODO: explain `DataModule`s and link to docs

## `trainer`

The `trainer` is meant to instantiate a `lightning.Trainer` object. To understand how to configure it, users are directed to `lightning.Trainer`'s [page](https://lightning.ai/docs/pytorch/stable/common/trainer.html). The sections on trainer [flags](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags) and its [API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) are especially important.

> **_NOTE:_**  it is in the `lightning.Trainer` that users can specify [`callbacks`](https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks) used to influence the course of training. This includes the very important [`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint) callback that should be configured to save checkpoint files in the way the user so pleases. `nequip`'s own [callbacks](../api/train.rst) can also be used here.

## `training_module`

TODO: explain `NequIPLightningModule` and `MetricsManager`

## `global_options`

TODO