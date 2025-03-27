# File Types

There are a handful of core file types in the NequIP framework to be aware of. They are explained briefly as follows.

## Config Files

Config files have the `.yaml` extension and serve as the main way to perform [training](./workflow.md/#training) (i.e. `nequip-train`). The structure of a config file is explained in greater detail in the ["Config File"](./config.md) page.

## Checkpoint Files

Checkpoint files have the `.ckpt` extension and are produced from [training](./workflow.md/#training) runs (i.e. `nequip-train` with `train` run type). 

Checkpointing behavior is controlled through PyTorch Lightning's [ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint) callback, which is an argument to the `lightning.Trainer`'s [page](https://lightning.ai/docs/pytorch/stable/common/trainer.html). See relevant [section](./config.md/#trainer) of the "Config File" page for how to configure checkpointing behavior in the config file.

Checkpoint files can be used to [continue](./workflow.md/#restarts) interrupted training runs.

Checkpoint files may also be used to fine-tune a pre-trained model through the `ModelFromCheckpoint` [model loader](../api/save_model.rst).

Checkpoint files can also be [compiled](./workflow.md/#compilation) to be used for inference.

```{tip}
It is safer to provide absolute paths to the checkpoint files when using `ModelFromCheckpoint` [model loader](../api/save_model.rst).
```

```{warning}
Checkpoint files are not expected to survive version changes, i.e. it is unlikely to be able to load a checkpoint file from an older version of `nequip` (or an older version of an extension package such as `allegro`). To enable old models to be used in newer versions, one can [package](./workflow.md/#packaging) the model with `nequip-package`, which produces [package files](#package-files) that can be used in later versions.
```

## Package Files

Package files have the `.nequip.zip` extension and are produced by [packaging models](./workflow.md/#packaging) (i.e. `nequip-package`).

The package file is an archival format for storing and distributing models. It does not only contain the weights of the model, but also the very code required for the model to work, hence making it (mostly) version independent.

Package files can also be used to fine-tune the packaged pre-trained model through the `ModelFromPackage` [model loader](../api/save_model.rst).

Package files can be [compiled](./workflow.md/#compilation) to be used for inference.

```{tip}
It is safer to provide absolute paths to the package files when using `ModelFromPackage` [model loader](../api/save_model.rst).
```

```{important}
In order to load checkpoint files produced by `ModelFromPackage` [model loader](../api/save_model.rst) training runs (e.g. for fine-tuning), the package file must exist at the location at which it was first specified as an argument to `ModelFromPackage`. Hence, refrain from moving the package file once used for training runs. (At worse, it is possible to `torch.load` the checkpoint, alter the package path before saving and using it.)
```

## Compiled Model Files

Compiled model files have either the `.nequip.pth` or `.nequip.pt2` extension, depending on whether the compilation mode is `torchscript` or `aotinductor`. They are produced by [compiled](./workflow.md/#compilation) (i.e. `nequip-compile`).

Compiled model files are only used for inference tasks in our [integrations](../integrations/all.rst) such as [ASE](../integrations/ase.md) and [LAMMPS](../integrations/lammps.md).
