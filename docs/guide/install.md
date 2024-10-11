# Installation

## Requirements
* Python >= 3.9
* PyTorch >= 1.13. PyTorch can be installed following the [instructions from their documentation](https://pytorch.org/get-started/locally/). Note that neither `torchvision` nor `torchaudio`, included in the default install command, are needed for NequIP.

**You must install PyTorch before installing NequIP, however it is not marked as a dependency of `nequip` to prevent `pip` from trying to overwrite your PyTorch installation.**

## Instructions

After installing `torch`, NequIP can be installed in the following ways.

1. from PyPI
    ```bash
    pip install nequip
    ```

2. from source, using the latest release
    ```bash
    git clone https://github.com/mir-group/nequip.git
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

Depending on the choice of [Lightning's loggers](https://lightning.ai/docs/pytorch/stable/extensions/logging.html), users may want to install the necessary logger. For example, one needs to `pip install wandb` to use Lightning's [WandbLogger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html#lightning.pytorch.loggers.WandbLogger).

## Installation Verification

The easiest way to check if your installation is working is to train a **toy** model:
```bash
cd configs
nequip-train -cn minimal.yaml
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