# The NequIP workflow

## Overview

At a glance, the NequIP workflow is as follows.

1. [**Train**](#training) &  [**test**](#testing) models with `nequip-train`, which produces a [checkpoint file](./files.md/#checkpoint-files).
2. [**Package**](#packaging) the model from the checkpoint file with `nequip-package`, which produces a [package file](./files.md/#package-files). Package files are the recommended format for distributing NequIP framework models as they are designed to be usable on different machines and code environments (e.g. with different `e3nn`, `nequip`, `allegro` versions than what the model was initially trained with).
3. [**Compile**](#compilation) the packaged model (or model from a checkpoint file) with `nequip-compile`, which produces a [compiled model file](./files.md/#compiled-model-files) that can be loaded for [**production simulations**](#production-simulations) in supported [integrations](../integrations/all.rst) such as [LAMMPS](../integrations/lammps.md) and [ASE](../integrations/ase.md).

## Training

The core command in NequIP is `nequip-train`, which takes in a YAML config file defining the dataset(s), model, and training hyperparameters, and then runs (or restarts) a training session. [Hydra](https://hydra.cc/) is used to manage the config files, and so many of the features and tricks from Hydra can be used if desired. `nequip-train` can be called as follows.

```bash
nequip-train -cp full/path/to/config/directory -cn config_name.yaml
```

Note that the flags `-cp` and `-cn` refer to the "config path" and "config name" respectively and are features of hydra's [command line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/). If one runs `nequip-train` in the same directory where the config file is located, the `-cp` part may be omitted. Note also that the full path is usually required if one uses `-cp`. Users who seek further configurability (e.g. using relative paths, multiple config files located in different directories, etc) are directed to the "[command line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/)" link to learn more.

Under the hood, the [Hydra](https://hydra.cc/) config utilities and the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework are used to facilitate training and testing in the NequIP infrastructure. One can think of the config as consisting of a set of classes to be instantiated with user-given parameters to construct objects required for training and testing to be performed. Hence, the API of these classes form the central source of truth in terms of what configurable parameters there are. These classes could come from

- `torch` in the case of [optimizers and learning rate scheduler](https://pytorch.org/docs/stable/optim.html), or
- `Lightning` such as Lightning's [trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) or Lightning's native [callbacks](https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks), or
- `nequip` itself such as the various [DataModules](../api/datamodule.rst), custom [callbacks](../api/callbacks.rst), etc

Users are advised to look at `configs/tutorial.yaml` to understand how the config file is structured, and then to look up what each of the classes do and what parameters they can take (be they on `torch`, `Lightning` or `nequip`'s docs). The documentation for `nequip` native classes can be found under [Python API](../api/nequip.rst).

Checkpointing behavior is controlled by `Lightning` and configuring it is the onus of the user. Checkpointing can be controlled by flags in Lightning's [trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) and can be specified even further with Lightning's [ModelCheckpoint callback](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint).

### Restarts

If a run is interrupted, one can continue training from a checkpoint file with the following command

```bash
nequip-train -cp full/path/to/config/directory -cn config_name.yaml ++ckpt_path='path/to/ckpt_file'
```

where we have used Hydra's [override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) (`++`). Note how one must still specify the config file used. Training from a checkpoint will always use the model from the checkpoint file, but other training hyperparameters (dataset, loss, metrics, callbacks, etc) is determined by the config file passed in the restart `nequip-train` (and can therefore be different from that of the original config used to generate the checkpoint). The restart will also resume from the last `run` stage (i.e. `train`, `val`, `test`, etc) that was running before the interruption.

```{warning}
In general, the config should not be modified between restarts.  There are no safety checks to guard against nonsensical changes to the config used for restarts. It is the user's responsibility to ensure that any changes made are intended and reasonable. Users are advised to have a working understanding of how [checkpointing](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html) works under the hood with a sense of what training states are preserved upon restarting from checkpoints if one seeks to restart a run with an altered config file. 

In general, it is safest to restart without changes to the original config. If one seeks to train a model from a checkpoint file with very different training hyperparameters or datasets (e.g. for fine-tuning), one can use the `ModelFromCheckpoint` [model loader](../api/save_model).
```

```{tip}
Working directories are managed by Hydra, and users can configure how these directories behave, as well as pass these directories to `Lightning` objects (e.g. so that model checkpoints are saved in the Hydra generated directories). Visit Hydra's [output/working directory page](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/) to learn more.
```

## Testing

Testing is also performed with `nequip-train` by adding `test` to the list of `run` parameters in the config. Testing requires test dataset(s) to be defined with the `DataModule` defined by the `data` key in the config.

There are two main ways users can use `test`.

- One can have testing be done automatically after training in the same `nequip-train` session by specifying `run: [train, test]` in the config. The `test` phase will use the `best` model checkpoint from the `train` phase.
- One can run tests from a checkpoint file by having `run: [test]` in the config and using the `ModelFromCheckpoint` [model loader](../api/save_model) to load a model from a checkpoint file.

One can use `nequip.train.callbacks.TestTimeXYZFileWriter` ([see API](../api/callbacks.rst)) as a callback to have `.xyz` files written with the predictions of the model on the test dataset(s). (This is the replacement for the role `nequip-evaluate` served before `nequip` version `0.7.0`)

## Packaging

The recommended way to archive a trained model is to `package` it.

```bash
nequip-package --ckpt-path path/to/ckpt_file --output-path path/to/packaged_model.nequip.zip
```

```{warning}
The output path MUST have the extension `.nequip.zip`.
```

```{tip}
To see command line options, one can use `nequip-package -h`
```

While checkpoint files are unlikely to survive breaking changes across code versions, the packaging infrastructure was designed to ensure that packaged models will continue to be usable as code versions change.
`nequip-package` will save not only the model and its weights, but also the very code that the model depends on.
The packaged model can thus be loaded and used independently from the model code in the Python environment's NequIP (and extensions such as Allegro).
Packaged models can be used not only for inference, but also fine-tuning, e.g. through the `ModelFromPackage` [model loader](../api/save_model). The checkpoint files produced by a fine-tuning `nequip-train` run is compatible with the rest of the framework and can be used in usual workflows, e.g. restarting training with `++ckpt_path path/to/ckpt`, use in `ModelFromCheckpoint` in the config file, `nequip-compile`, `nequip-package`, etc.

## Compilation

`nequip-compile` is the command used to compile a model (from a checkpoint file or a package file) for [production simulations](#production-simulations) with our various [integrations](../integrations/all.rst). There are two modes that users can use for compilation, `torchscript` and `aotinductor`. The latter `aotinductor` requires at least PyTorch 2.6.

The command to compile a TorchScript model is as follows.

```bash
nequip-compile \
--input-path path/to/ckpt_file/or/package_file \
--output-path path/to/compiled_model.nequip.pth \
--device (cpu/cuda) \
--mode torchscript
```

The command to compile an AOT Inductor model is as follows.

```bash
nequip-compile \
--input-path path/to/ckpt_file/or/package_file \
--output-path path/to/compiled_model.nequip.pt2 \
--device (cpu/cuda) \
--mode aotinductor \
--target target_integration
```

```{warning}
`--mode torchscript` imposes that the `--output-path` ends with a `.nequip.pth` extension.\
`--mode aotinductor` imposes that the `--output-path` ends with a `.nequip.pt2` extension.
```

```{tip}
To see command line options, one can use `nequip-compile -h`
```

```{important}
`nequip-compile` should be called on the device where the compiled model will be used on for the production simulation. While this constraint is not a hard requirement for TorchScript mode compilation, it is necessary for AOT Inductor mode compilation as the models are compiled specifically for a particular device.
```

```{important}
Note also that AOT Inductor mode compilation requires access to compilers (e.g. `gcc`) when running `nequip-compile`. Specifically, C++17 support is required, which requires `gcc` version 8 or higher (preferably >=11 where C++17 is the default), otherwise `filesystem` errors will occur. You can check your `gcc` version with `gcc --version`, and may need to upgrade or load a specific HPC module to get the required version.
```

```{tip}
If `--mode aotinductor` is used for [compilation](#compilation), the `nequip-compile` call must be configured in a manner specific to the intended integration. For supported integrations, a convenience flag is provided in the form of `--target`, which could be `--target ase` for compiled models to be used with ASE, or `--target pair_nequip` for compiled NequIP GNN models to be used in LAMMPS, or `--target pair_allegro` for compiled Allegro models to be used in LAMMPS.

The `--target` flag is a simplification over having to provide `--input-fields` and `--output-fields`. Developers designing new models or wanting to set up new integrations can manually provide `--input-fields` and `--output-fields`. New integration "target"s may be added through PRs or through NequIP extension packages. Engage with us on GitHub if you seek to do something like this.
```

```{tip}
If performing training and inference on separate machine, with possibly different Python environments, one can consider [packaging](#packaging) the trained model and transfering the packaged model to the inference machine where one can then `nequip-compile` the package file. Such an approach will be less prone to errors due to inconsistent Python environments, for example.
```

## Production Simulations

Once a model has been [trained](#training) in the NequIP framework, it can be [compiled](#compilation) for use in production simulations in our supported [integrations](../integrations/all.rst) with other codes and simulation engines, including [LAMMPS](../integrations/lammps.md) and [ASE](../integrations/ase.md).
