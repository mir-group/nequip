# The `nequip` workflow

The `nequip` workflow has three steps:
 1. **Training**:  `nequip-train`
 2. **Testing**: `nequip-train`
 3. **Deploying**: `nequip-deploy`
 4. **Using deployed models**: [Integrations](../integrations/all.rst)


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
nequip-train -cp path/to/config/directory -cn config_name.yaml ++ckpt_path='path/to/ckpt_file'
```
where we have used Hydra's [override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) (`++`). Note how one must still specify the config file used. Training from a checkpoint will always use the model from the checkpoint file, but other training hyperparameters (dataset, loss, metrics, callbacks, etc) is determined by the config file passed in the restart `nequip-train` (and can therefore be different from that of the original config used to generate the checkpoint).

Note that the working directories are managed by Hydra, and users can configure how these directories behave, as well as pass these directories to `Lightning` objects (e.g. so that model checkpoints are saved in the Hydra generated directories). Visit Hydra's [output/working directory page](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/) to learn more.


## Testing

Testing is also performed with `nequip-train` by adding `test` to the list of `run` parameters in the config. For example, to have testing be done automatically after training, one can specify `run: [train, test]` in the config. Testing requires test dataset(s) to be defined with the `DataModule` under `data` in the config. One can use `nequip.train.callbacks.TestTimeXYZFileWriter` ([see API](../api/train.rst)) as a callback to have `.xyz` files written with the predictions of the model on the test dataset(s). (This is the replacement for the role `nequip-evaluate` served before `nequip` version `0.7.0`)


## Deploying

Once you have trained a model, you must deploy it to create an archive of its trained parameters and metadata that can be used for simulations and other calculations:
```bash
nequip-deploy -cp path/to/config/directory -cn config_name.yaml ++mode=build ++ckpt_path='path/to/ckpt_file' ++out_file='path/to/deployed_model'
```

One can inspect the deployed model with the following command
```bash
nequip-deploy -cp path/to/config/directory -cn config_name.yaml ++mode=info ++model_path='path/to/deployed_model'
```


## Using deployed models

### ...to run simulations and other calculations

There are many ways a deployed model can be used. Most often it can be [used for moelcular dynamics and other calculations in LAMMPS](../integrations/lammps.md). For integrations with other codes and simulation engines, see [Integrations](../integrations/all.rst).

### ...at a low level in your own code
While LAMMPS, or other integrations should be sufficient for the vast majority of usecases, deployed models can also be loaded as PyTorch TorchScript models to be called in your own code:
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


