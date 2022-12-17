Installation
============

NequIP requires:

 * Python >= 3.6
 * PyTorch >= 1.8, <=1.11.*. PyTorch can be installed following the `instructions from their documentation <https://pytorch.org/get-started/locally/>`_. Note that neither ``torchvision`` nor ``torchaudio``, included in the default install command, are needed for NequIP.

To install:

 * We use `Weights&Biases <https://wandb.ai>`_ to keep track of experiments. This is not a strict requirement — you can use our package without it — but it may make your life easier. If you want to use it, create an account `here <https://wandb.ai/login?signup=true>`_ and install the Python package::

    pip install wandb

 * Install the latest stable NequIP::

    pip install https://github.com/mir-group/nequip/archive/main.zip

To install previous versions of NequIP, please clone the repository from GitHub and check out the appropriate tag (for example ``v0.3.3`` for version 0.3.3).

To install the current **unstable** development version of NequIP, please clone our repository and check out the ``develop`` branch.

Installation Issues
-------------------

The easiest way to check if your installation is working is to train a _toy_ model::

    nequip-train configs/minimal.yaml

If you suspect something is wrong, encounter errors, or just want to confirm that everything is in working order, you can also run the unit tests::

    pip install pytest
    pytest tests/unit/

To run the full tests, including a set of longer/more intensive integration tests, run::

    pytest tests/

If a GPU is present, the unit tests will use it.