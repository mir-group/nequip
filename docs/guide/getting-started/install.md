# Installation

## Requirements
* Python >= 3.10
* PyTorch >= 2.2. PyTorch can be installed following the [instructions from their documentation](https://pytorch.org/get-started/locally/). Note that neither `torchvision` nor `torchaudio`, included in the default install command, are needed for NequIP.

```{note}
PyTorch >= 2.6 is required for PyTorch 2.0 compilation utilities including using `torch.compile` for training and AOTInductor compilation for integrations such as ASE and LAMMPS.
```

## Instructions

After installing `torch`, NequIP can be installed in the following ways.

1. from PyPI
    ```bash
    pip install nequip
    ```

2. from source, using the latest release
    ```bash
    git clone --depth 1 https://github.com/mir-group/nequip.git
    cd nequip
    pip install . 
    ```

3. from source, using the latest `develop` branch
    ```bash
    git clone https://github.com/mir-group/nequip.git
    cd nequip
    git checkout develop
    pip install . 
    ```

If you want to track your training runs with third-party services, like Weights and Biases, that are supported by [PyTorch Lightning's loggers](https://lightning.ai/docs/pytorch/stable/extensions/logging.html), you may need to install extra packages. For example, one needs to `pip install wandb` to use Lightning's {class}`~lightning.pytorch.loggers.WandbLogger`.

## Checking your installation

The easiest way to check if your installation is working is to train the tutorial model:
```bash
cd configs
python get_tutorial_data.py
nequip-train -cn tutorial.yaml
```

If you suspect something is wrong, encounter errors, or just want to confirm that everything is in working order, you can also run the unit tests:

```
pip install pytest
pytest tests/unit/
```

To run the full tests, including a set of longer/more intensive integration tests, run:
```
pytest tests/
```

If a GPU is present, the unit tests will use it.