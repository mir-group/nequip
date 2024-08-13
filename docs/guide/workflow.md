# The `nequip` workflow

The `nequip` workflow has three steps:
 1. **Training**:  `nequip-train`, `nequip-benchmark`
 2. **Deploying**: `nequip-deploy`
 3. **Using deployed models**: `nequip-evaluate`, [Integrations](../integrations/all.rst)

For details on the specific options of any of these commands, please run `nequip-* --help`.

## Training

The core command in `nequip` is `nequip-train`, which takes in a YAML configuration file defining the model, its hyperparaters, a training and validation set, and other options, and then runs (or restarts) a training session:
```bash
$ nequip-train configs/example.yaml
```
A good starting point can be found in `configs/example.yaml`; a comprehensive documentation of the options can be found in `configs/full.yaml`.

`nequip-train` logs out a variety of per-minibatch metrics throughout the training process (see [Loss functions and metrics](./loss.md)), reports metrics for the entire epoch at the end of each epoch to the log and to various interfaces, such as [Weights and Biases](../integrations/wandb.md).

At the end of each epoch `nequip` saves the current state of the model and training process, which enables seamless restarts running the same `nequip-train` command of interupted training runs as long as the `append: true` option is set.

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


