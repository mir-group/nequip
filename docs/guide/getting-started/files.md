# File Types

There are a handful of core file types in the NequIP framework to be aware of.

## Config Files

Config files have the `.yaml` extension and describe [training](./workflow.md/#training) jobs that can be run with `nequip-train`. The structure of a config file is explained in greater detail in the ["Config File"](../configuration/config.md) page.

## Checkpoint Files

Checkpoint files have the `.ckpt` extension and are produced from [training](./workflow.md/#training) runs (`nequip-train` with the `train` run type). 

Checkpointing is controlled through PyTorch Lightning's {class}`~lightning.pytorch.callbacks.ModelCheckpoint` callback, which can be provided to the {class}`~lightning.pytorch.trainer.trainer.Trainer`. See relevant [section](../configuration/config.md/#trainer) of the "Config File" page for how to configure checkpointing behavior in the config file.

Checkpoint files can be used to [continue/restart](./workflow.md/#saving-and-restarting) interrupted training runs.

Checkpoint files may also be used to [fine-tune](./workflow.md#fine-tuning-packaged-models) a pre-trained model through the {func}`~nequip.model.ModelFromCheckpoint` [model loader](../../api/save_model.rst).

Checkpoint files can also be [compiled](./workflow.md/#compilation) to be used for inference.

```{tip}
It is safer to provide absolute paths to the checkpoint files when using {func}`~nequip.model.ModelFromCheckpoint` [model loader](../../api/save_model.rst).
```

```{warning}
Checkpoint files are not generally expected to survive version changes, i.e. it is unlikely to be able to load a checkpoint file from an older version of `nequip` (or an older version of an extension package such as {mod}`allegro`). To enable old models to be used in newer versions, one can [package](./workflow.md/#packaging) the model with `nequip-package`, which produces [package files](#package-files) that can be used in later versions.
```

## Package Files

Package files have the `.nequip.zip` extension and are produced by [packaging models](./workflow.md/#packaging) (i.e. `nequip-package`).

The package file is an archival format for storing and distributing models. It does not only contain the weights of the model, but also the very code required for the model to work, hence making it (mostly) version independent.

Package files can also be used to [fine-tune](./workflow.md#fine-tuning-packaged-models) the packaged pre-trained model through the {func}`~nequip.model.ModelFromPackage` [model loader](../../api/save_model.rst).

```{important}
It is safest to provide absolute paths when using {func}`~nequip.model.ModelFromPackage`.

In order to load checkpoint files produced by {func}`~nequip.model.ModelFromPackage` [model loader](../../api/save_model.rst) training runs (e.g. for fine-tuning), the package file must exist at the location at which it was first specified as an argument to {func}`~nequip.model.ModelFromPackage`. Hence, refrain from moving the package file once used for training runs. (At worse, it is possible to {func}`torch.load` the checkpoint, alter the package path before saving and using it.)
```

Package files can be [compiled](./workflow.md/#compilation) to be used for inference.

## Compiled Model Files

Compiled model files have either the `.nequip.pth` or `.nequip.pt2` extension, depending on whether the compilation mode is `torchscript` or `aotinductor`. They are produced by [`nequip-compile`](./workflow.md/#compilation).

Compiled model files are only used for inference and can be run with [integrations](../../integrations/all.rst) such as ASE and [LAMMPS](../../integrations/lammps/index.md).
