# The `nequip` workflow

The `nequip` workflow has three steps:
 1. **Training**:  `nequip-train`, `nequip-benchmark`
 2. **Deploying**: `nequip-deploy`
 3. **Using deployed models**: [Integrations](../integrations/all.rst)


## Training

The core command in `nequip` is `nequip-train`, which takes in a YAML config file defining the dataset(s), model, and training hyperparameters, and then runs (or restarts) a training session. [Hydra](https://hydra.cc/) is used to manage the config files, and so many of the features and tricks from Hydra can be used if desired. `nequip-train` can be called as follows.
```bash
$ nequip-train -cp path/to/config/directory -cn config_name.yaml
```
Note that the flags `-cp` and `-cn` refer to the "config path" and "config name" respectively and are features of hydra's [command line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/). It is possible to use different flags to achieve the same effect if desired (follow the "command line flags" link to learn more).

Under the hood, the [Hydra](https://hydra.cc/) config utilities and the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework are used to facilitate training and testing in the NequIP infrastructure. One can think of the config as consisting of a set of classes to be instantiated with user-given parameters to construct objects required for training and testing to be performed. Hence, the API of these classes form the central source of truth in terms of what configurable parameters there are. These classes could come from 
 - `torch` in the case of [optimizers and learning rate scheduler](https://pytorch.org/docs/stable/optim.html), or 
 - `Lightning` such as Lightning's [trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) or Lightning's native [callbacks](https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks), or 
 - `nequip` itself such as the various [DataModules](../api/datamodule.rst), custom [callbacks](../api/train.rst), etc

Users are advised to look at `configs/tutorial.yaml` to understand how the config file is structured, and then to look up what each of the classes do and what parameters they can take (be they on `torch`, `Lightning` or `nequip`'s docs). The documentation for `nequip` native classes can be found under [Python API](../api/nequip.rst).

Checkpointing behavior is controlled by `Lightning` and configuring it is the onus of the user. Checkpointing can be controlled by flags in Lightning's [trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) and can be specified even further with Lightning's [ModelCheckpoint callback](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint).

One can continue training from a checkpoint file with the following command
```bash
nequip-train -cp path/to/config/directory -cn config_name.yaml ++ckpt_path="path/to/ckpt_file"
```
where we have used Hydra's [override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) (`++`). Note how one must still specify the config file used. Training from a checkpoint will always use the model from the checkpoint file, but other training hyperparameters (dataset, loss, metrics, callbacks, etc) is determined by the config file passed in the restart `nequip-train` (and can therefore be different from that of the original config used to generate the checkpoint).

Note that the working directories are managed by Hydra, and users can configure how these directories behave, as well as pass these directories to `Lightning` objects (e.g. so that model checkpoints are saved in the Hydra generated directories). Visit Hydra's [output/working directory page](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/) to learn more.

For molecular dynamics simulations in particular, the speed of the interatomic potential model can be very important. `nequip` provides the `nequip-benchmark` command to more easily let you get an approximate sense of the speed of different combinations of hyperparaters and system structures. The command takes the exactly same config file as `nequip-train`, but instead of training it initializes a random model with the given hyperparameter and benchmarks it on the a frame from the training set:
```bash
$ nequip-benchmark configs/minimal.yaml
``` 

## Deploying

Once you have trained a model, you must deploy it to create an archive of its trained parameters and metadata that can be used for simulations and other calculations:
```bash
$ nequip-deploy build path/to/training_session_directory deployed_model.pth
```
which generates `deployed_model.pth`.

You can get info about a deployed model using:
```bash
$ nequip-deploy info deployed_model.pth
```

## Using deployed models

### ...to run simulations and other calculations

There are many ways a deployed model can be used. Most often it can be [used for moelcular dynamics and other calculations in LAMMPS](../integrations/lammps.md). For integrations with other codes and simulation engines, see [Integrations](../integrations/all.rst).

### ...on a fixed set of inputs
Deployed models can be easily evaluated on a fixed set of inputs using `nequip-evaluate`, with automatic support for batching, which provides a significant speedup on GPUs:
```bash
$ nequip-evaluate --model path/to/deployed.pth --dataset-config inputs_definition.yaml --batch-size 50 --output model_predictions.xyz
```
`inputs_definition.yaml` here defines a dataset using exactly the same options as a training dataset (see [Dataset](./dataset.md)), but which contains the inputs to evaluate the model on. 

If you want to output fields from the model's prediction other than the defaults (energies, forces, virials/stress), you can request them in a comma-separated list with `--output-fields`:
```bash
$ nequip-evaluate --model path/to/deployed.pth --dataset-config inputs_definition.yaml --batch-size 50 --output model_predictions.xyz --output-fields node_features,edge_vectors
```
The output file, `model_predictions.xyz`, is in the extXYZ format and can be read with ASE or other tools. Nonstandard fields will show up in the `ase.Atoms` object in `.arrays` for per-atom quantities, or `.info` for system- or edge-level quantities.

You may need to lower the batch size for large models or large systems, and may be able to increase it and improve performance for smaller ones.

### ...at a low level in your own code
While LAMMPS, `nequip-evaluate`, or other integrations should be sufficient for the vast majority of usecases, deployed models can also be loaded as PyTorch TorchScript models to be called in your own code:
```python
import torch
import ase.io
from nequip.data import AtomicData, AtomicDataDict
from nequip.scripts.deploy import load_deployed_model, R_MAX_KEY

device = "cpu"  # "cuda" etc.
model, metadata = load_deployed_model(
    "path_to_deployed_model.pth,
    device=device,
)

# Load some input structure from an XYZ or other ASE readable file:
data = AtomicData.from_ase(ase.io.read("example_input_structure.xyz"), r_max=metadata[R_MAX_KEY])
data = data.to(device)

out = model(AtomicData.to_AtomicDataDict(data))

print(f"Total energy: {out[AtomicDataDict.TOTAL_ENERGY_KEY]}")
print(f"Force on atom 0: {out[AtomicDataDict.FORCE_KEY][0]}")
```


