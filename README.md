# NequIP

NequIP is an open-source code for building E(3)-equivariant interatomic potentials.

[![Documentation Status](https://readthedocs.org/projects/nequip/badge/?version=latest)](https://nequip.readthedocs.io/en/latest/?badge=latest)

![nequip](./nequip.png)

**PLEASE NOTE:** the NequIP code is under active development and is still in beta versions 0.x.x. In general changes to the patch version (the third number) indicate backward compatible beta releases, but please be aware that file formats and APIs may change. Bug reports are also welcomed in the GitHub issues!

## Installation

NequIP requires:

* Python >= 3.6
* PyTorch >= 1.8

To install:

* Install [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric), following [their installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and making sure to install with the correct version of CUDA. (Please note that `torch_geometric>=1.7.1)` is required.)

* Install our fork of [`pytorch_ema`](https://github.com/Linux-cpp-lisp/pytorch_ema) for using an Exponential Moving Average on the weights: 
```bash
$ pip install "git+https://github.com/Linux-cpp-lisp/pytorch_ema@context_manager#egg=torch_ema"
```

* We use [Weights&Biases](https://wandb.ai) to keep track of experiments. This is not a strict requirement — you can use our package without it — but it may make your life easier. If you want to use it, create an account [here](https://wandb.ai) and install the Python package:

```
pip install wandb
```

* Install NequIP

```
git clone https://github.com/mir-group/nequip.git
cd nequip
pip install . 
```

### Installation Issues

The easiest way to check if your installation is working is to train a toy model:
```bash
$ nequip-train configs/minimal.yaml
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

Note: the integration tests have hung in the past on certain systems that have GPUs. If this happens to you, please report it along with information on your software environment in the [Issues](https://github.com/mir-group/nequip/issues)!

## Usage

**! PLEASE NOTE:** the first few training epochs/calls to a NequIP model can be painfully slow. This is expected behaviour as the [profile-guided optimization of TorchScript models](https://program-transformations.github.io/slides/pytorch_neurips.pdf) takes a number of calls to warm up before optimizing the model. This occurs regardless of whether the entire model is compiled because many core components from e3nn are compiled and optimized through TorchScript.

### Basic network training

To train a network, you run `nequip-train` with a YAML config file that describes your data set, model hyperparameters, and training options. 

```bash
$ nequip-train configs/example.yaml
```

A number of example configuration files are provided:
 - [`configs/minimal.yaml`](configs/minimal.yaml): A minimal example of training a toy model on force data.
 - [`configs/minimal_eng.yaml`](configs/minimal_eng.yaml): The same, but for a toy model that predicts and trains on only energy labels.
 - [`configs/example.yaml`](configs/example.yaml): Training a more realistic model on forces and energies.
 - [`configs/full.yaml`](configs/full.yaml): A complete configuration file containing all available options along with documenting comments.

Training runs can be restarted using `nequip-restart`; training that starts fresh or restarts depending on the existance of the working directory can be launched using `nequip-requeue`. All `nequip-*` commands accept the `--help` option to show their call signatures and options.

### Evaluating trained models (and their error)

The `nequip-evaluate` command can be used to evaluate a trained model on a specified dataset, optionally computing error metrics or writing the results to an XYZ file for further processing.

The simplest command is:
```bash
$ nequip-evaluate --train-dir /path/to/training/session/
```
which will evaluate the original training error metrics over any part of the original dataset not used in the training or validation sets.

For more details on this command, please run `nequip-evaluate --help`.

### Deploying models

The `nequip-deploy` command is used to deploy the result of a training session into a model that can be stored and used for inference.
It compiles a NequIP model trained in Python to [TorchScript](https://pytorch.org/docs/stable/jit.html).
The result is an optimized model file that has no dependency on the `nequip` Python library, or even on Python itself:
```bash
nequip-deploy build path/to/training/session/ where/to/put/deployed_model.pth
```
For more details on this command, please run `nequip-deploy --help`.

### Using models in Python

Both deployed and undeployed models can be used in Python code; for details, see the end of the [Developer's tutorial](https://deepnote.com/project/2412ca93-7ad1-4458-972c-5d5add5a667e) mentioned again below.

An ASE calculator is also provided in `nequip.dynamics`.

### LAMMPS Integration 

NequIP is integrated with the popular Molecular Dynamics code [LAMMPS](https://www.lammps.org/) which allows for MD simulations over large time- and length-scales and gives users access to the full suite of LAMMPS features. 

The interface is implemented as `pair_style nequip`. Using it requires two simple steps: 

1. Deploy a trained NequIP model, as discussed above.
```
nequip-deploy build path/to/training/session/ path/to/deployed.pth
```
The result is an optimized model file that has no Python dependency and can be used by standalone C++ programs such as LAMMPS.

2. Change the LAMMPS input file to the nequip `pair_style` and point it to the deployed NequIP model:

```
pair_style	nequip
pair_coeff	* * deployed.pth
```

For installation instructions, please see the [`pair_nequip` repository](https://github.com/mir-group/pair_nequip).


## Developer's tutorial 

A more in-depth introduction to the internals of NequIP can be found in the [tutorial notebook](https://deepnote.com/project/2412ca93-7ad1-4458-972c-5d5add5a667e). This notebook discusses theoretical background as well as the Python interfaces that can be used to train and call models.

Please note that for most common usecases, including customized models, the `nequip-*` commands should be prefered for training models.

## References & citing

The theory behind NequIP is described in our preprint (1). NequIP's backend builds on e3nn, a general framework for building E(3)-equivariant neural networks (2). If you use this repository in your work, please consider citing NequIP (1) and e3nn (3):

 1. https://arxiv.org/abs/2101.03164
 2. https://e3nn.org
 3. https://doi.org/10.5281/zenodo.3724963

## Authors

NequIP is being developed by:

 - Simon Batzner
 - Albert Musaelian
 - Lixin Sun
 - Anders Johansson
 - Mario Geiger
 - Tess Smidt

under the guidance of [Boris Kozinsky at Harvard](https://bkoz.seas.harvard.edu/).

## Contact & questions

If you have questions, please don't hesitate to reach out at batzner[at]g[dot]harvard[dot]edu. 

If you find a bug or have a proposal for a feature, please post it in the [Issues](https://github.com/mir-group/nequip/issues).
If you have a question, topic, or issue that isn't obviously one of those, try our [GitHub Disucssions](https://github.com/mir-group/nequip/discussions).
