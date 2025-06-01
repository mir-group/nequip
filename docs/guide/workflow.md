# The NequIP Workflow

## Overview

At a glance, the NequIP workflow is as follows.

1. [**Train**](#training) models with `nequip-train`, which produces a [checkpoint file](./files.md/#checkpoint-files).
2. [**Test**](#testing) those models using `nequip-train`, sometimes as part of the same call to the command.
2. [**Package**](#packaging) the model from the checkpoint file with `nequip-package`, which produces a [package file](./files.md/#package-files). Package files are the recommended format for distributing NequIP framework models as they are designed to be usable on different machines and code environments (e.g. with different {mod}`e3nn`, {mod}`nequip`, {mod}`allegro` versions than what the model was initially trained with).
3. [**Compile**](#compilation) the packaged model (or model from a checkpoint file) with `nequip-compile`, which produces a [compiled model file](./files.md/#compiled-model-files) that can be loaded for [**production simulations**](#production-simulations) in supported [integrations](../integrations/all.rst) such as [LAMMPS](../integrations/lammps.md) and [ASE](../integrations/ase.md).

## Training

The core command in NequIP is `nequip-train`, which takes in a YAML config file defining the dataset(s), model, and training hyperparameters, and then runs (or restarts) a training session. [Hydra](https://hydra.cc/) is used to manage the config files, and so many of the features and tricks from Hydra can be used if desired. `nequip-train` can be called as follows.

```bash
nequip-train -cp full/path/to/config/directory -cn config_name.yaml
```

`nequip-train` uses the {class}`~lightning.pytorch.trainer.trainer.Trainer` from [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html#train-the-model) to run a training loop.

### Command line options
The command line interface of `nequip-train` is managed by Hydra, and complete details on its flexible syntax can be found in the [Hydra documentation](https://hydra.cc/docs/advanced/hydra-command-line-flags/).

The flags `-cp` and `-cn` refer to the "config path" and "config name" respectively. If one runs `nequip-train` in the same directory where the config file is located, the `-cp` flag may be omitted. Note also that the full path is usually required if one uses `-cp`. Users who seek further configurability (e.g. using relative paths, multiple config files located in different directories, etc) are directed to the "[command line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/)" page in the Hydra docs to learn more.

Working directories for output files from `nequip-train` are [managed by Hydra](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory), and users can configure how these directories are organized through [Hydra's options](https://hydra.cc/docs/configure_hydra/workdir/). 

### The config file
Under the hood, the [Hydra](https://hydra.cc/) config utilities and the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework are used to facilitate training and testing in the NequIP infrastructure. The config defines a hierarchy of objects, built by instantiating classes, usually specified in the config with `_target_`, with the parameters the user provides. The Python API of these classes exactly corresponds to the available configuration options in the config file. As a result, the Python API of these classes is the single source of truth defining valid configuration options. These classes could come from:

- {mod}`torch` itself, in the case of [optimizers](https://docs.pytorch.org/docs/stable/optim.html) and [learning rate schedulers](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate);
- {mod}`lightning`, such as Lightning's {class}`~lightning.pytorch.trainer.trainer.Trainer` or Lightning's native [callbacks](https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks);
- `nequip`, such as the various [DataModules](../api/datamodule.rst), custom [callbacks](../api/callbacks.rst), and so on.

Users are advised to look at the [tutorial configuration](https://github.com/mir-group/nequip/blob/main/configs/tutorial.yaml) to understand how the config file is structured, and then to look up what each of the classes do and what parameters they can take (be they on [PyTorch](https://pytorch.org/docs/stable/), [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) or [NequIP's API docs](../api/nequip.rst)). The documentation for `nequip`'s own classes can be found in the [Python API](../api/nequip.rst) section of this documentation. For detailed guidance on config structure, see the [Config File guide](./config.md).

```{tip}
Hydra's output directory can be accessed in the config file using variable interpolation, which is very useful, for example, to instruct `Lightning` to save checkpoints in Hydra's output directory:

    callbacks:
      - _target_: lightning.pytorch.callbacks.ModelCheckpoint
        dirpath: ${hydra:runtime.output_dir}
        ...

```

### Saving and restarting
Checkpointing behavior is controlled by {mod}`lightning` and configuring it is the onus of the user. Checkpointing can be controlled by flags in Lightning's {class}`~lightning.pytorch.trainer.trainer.Trainer` and can be specified even further with Lightning's {class}`~lightning.pytorch.callbacks.ModelCheckpoint` callback.

If a run is interrupted, one can continue training from a checkpoint file with the following command

```bash
nequip-train -cp full/path/to/config/directory -cn config_name.yaml ++ckpt_path='path/to/ckpt_file'
```

where we have used Hydra's [override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) (`++`). Note how one must still specify the config file used. Training from a checkpoint will always use the model from the checkpoint file, but other training hyperparameters (dataset, loss, metrics, callbacks, etc) are determined by the config file passed in the restart `nequip-train` (and can therefore be different from that of the original config used to generate the checkpoint). The restart will also resume from the last `run` stage (i.e. `train`, `val`, `test`, etc) that was running before the interruption.

```{warning}
DO NOT MODIFY THE CONFIG BETWEEN RESTARTS. There are no safety checks to guard against nonsensical changes to the config used for restarts, which can cause various problems during state restoration. It is safest to restart without changes to the original config. If one seeks to train a model from a checkpoint file with different training hyperparameters or datasets (e.g. for fine-tuning), one can use the {func}`~nequip.model.ModelFromCheckpoint` [model loader](../api/save_model). The only endorsed exception is raising the `max_epochs` argument of the {class}`~lightning.pytorch.trainer.trainer.Trainer` to extend the training run if it was interrupted because `max_epochs` was previously too small.
```

## Testing

Testing is also performed with `nequip-train` by adding `test` to the list of `run` parameters in the config. Testing requires test dataset(s) to be defined with the {class}`~nequip.data.datamodule.NequIPDataModule` defined by the `data` key in the config.

There are two main ways users can use `test`.

- One can have testing be done automatically after training in the same `nequip-train` session by specifying `run: [train, test]` in the config. The `test` phase will use the `best` model checkpoint from the `train` phase.
- One can run tests from a checkpoint file by having `run: [test]` in the config and using the {func}`~nequip.model.ModelFromCheckpoint` [model loader](../api/save_model) to load a model from a checkpoint file.

One can use the {class}`~nequip.train.callbacks.TestTimeXYZFileWriter` callback ([see API](../api/callbacks.rst)) to write out `.xyz` files containing the predictions of the model on the test dataset(s).

## Packaging

The recommended way to archive a trained model is to `package` it with the `build` option of `nequip-package`.

```bash
nequip-package build path/to/ckpt_file path/to/packaged_model.nequip.zip
```

One can inspect the metadata of the packaged model by using the `info` option.

```bash
nequip-package info path/to/pkg_file.nequip.zip
```

```{warning}
The output path MUST have the extension `.nequip.zip`.
```

```{tip}
To see command line options, one can use `nequip-package -h`. There are two options `build` and `info`, so one can get more detailed information with `nequip-package build -h` and `nequip-package info -h`.
```

While checkpoint files are unlikely to survive breaking changes across updates to the software, the packaging infrastructure is designed to allow packaged models to remain usable as the framework is updated.
`nequip-package` saves not only the model and its weights, but also a snapshot of the code that implements the model at the time the model is packaged.
The packaged model can thus be loaded and used independently even if new and different versions of NequIP (and extensions such as {mod}`allegro`) are later installed.

### Fine-tuning packaged models

Packaged models can be used for both inference and fine-tuning.  Fine-tuning uses the {func}`~nequip.model.ModelFromPackage` [model loader](../api/save_model) in the config for a new `nequip-train` run to use the model from the package as the starting point. The checkpoint files produced by this kind of fine-tuning `nequip-train` run can be used as usual and support restarting training with `++ckpt_path path/to/ckpt`, further fine-tuning using {func}`~nequip.model.ModelFromCheckpoint`, `nequip-compile`, `nequip-package`, etc.

## Compilation

`nequip-compile` is the command used to compile a model (either from a checkpoint file or a package file) for [production simulations](#production-simulations) with our various [integrations](../integrations/all.rst). There are two compiler modes: `torchscript` and `aotinductor`, which produce compiled model files with extensions `.nequip.pth` and `.nequip.pt2` respectively. We generally recommend the newer and faster `aotinductor`, but it requires PyTorch 2.6 or later. 

To compile a model with TorchScript:
```bash
nequip-compile \
  path/to/ckpt_file/or/package_file \
  path/to/compiled_model.nequip.pth \
  --device [cpu|cuda] \
  --mode torchscript
```

To compile a model with AOTInductor:
```bash
nequip-compile \
  path/to/ckpt_file/or/package_file \
  path/to/compiled_model.nequip.pt2 \
  --device [cpu|cuda] \
  --mode aotinductor \
  --target [ase|pair_nequip|pair_allegro|...]
```

```{important}
`nequip-compile` should be called on the same type of system and device where the compiled model will be used. This constraint may not be always be necessary for TorchScript compilation, but it is **required** for AOTInductor compilation, which specializes the model to a particular type of GPU, etc.
```

```{important}
AOTInductor requires access to compilers like `gcc` and `nvcc` when running `nequip-compile`. Specifically, C++17 support is required, which requires `gcc` version 8 or higher (preferably >=11 where C++17 is the default), otherwise errors involving the `filesystem` standard library will occur. You can check your `gcc` version with `gcc --version`, and may need to upgrade or load a specific module on your HPC system to get the required version.
```

```{tip}
If `--mode aotinductor` is used, the compiled model will be specific to a specified `--target` integration. For example, the framework provides `--target ase` for compiled models to be used with ASE, `--target pair_nequip` for compiled NequIP GNN models to be used in LAMMPS, or `--target pair_allegro` for compiled Allegro models to be used in LAMMPS.

The `--target` flag wraps the `--input-fields` and `--output-fields` options. Developers designing new models or wanting to set up new integrations can manually provide `--input-fields` and `--output-fields`. New integration "target"s may be added through PRs or through NequIP extension packages. Engage with us on GitHub if you seek to do something like this.
```

```{tip}
If performing training and inference on separate machines, with possibly different Python, CUDA, or hardware environments, consider [packaging](#packaging) the trained model and transferring the packaged model to the inference machine and running `nequip-compile` on it there.
```

## Production Simulations

Once a model has been [trained](#training) and [compiled](#compilation) it can be used to run production simulations in our supported [integrations](../integrations/all.rst) with other codes and simulation engines, including [LAMMPS](../integrations/lammps.md) and [ASE](../integrations/ase.md).
