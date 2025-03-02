# NequIP

NequIP is an open-source code for building E(3)-equivariant interatomic potentials.

**:red_circle: PLEASE NOTE :red_circle:**:  Please note that a major update to this code is in the final stages of development and as a result support for the current release is a lower priority.

[![Documentation Status](https://readthedocs.org/projects/nequip/badge/?version=latest)](https://nequip.readthedocs.io/en/latest/?badge=latest)

![nequip](./logo.png)

**PLEASE NOTE:** the NequIP code is under active development and is still in beta versions 0.x.x. In general changes to the patch version (the third number) indicate backward compatible beta releases, but please be aware that file formats and APIs may change. Bug reports are also welcomed in the GitHub issues!

## Install

Installation instructions can be found [here](docs/guide/install.md).

## Tutorial 

The best way to learn how to use NequIP is through the [Colab Tutorial](https://bit.ly/mrs-nequip). This will run entirely on Google's cloud virtual machine; you do not need to install or run anything locally. 

## Usage

This [document](docs/guide/workflow.md) explains the `nequip` workflow, from training and testing models to compiling models for inference in our integrations such as [ASE](docs/integrations/ase.md) and [LAMMPS](docs/integrations/lammps.md).

 [`configs/tutorial.yaml`](configs/tutorial.yaml) is a complete configuration file containing all available options along with documenting comments. This file is **for reference**.

## Plugins / extending `nequip`

`nequip` is a modular framework and extension packages can provide new model components, architectures, etc. The main extension package(s) currently available are:
 - [Allegro](https://github.com/mir-group/allegro): implements the highly parallelizable Allegro model architecture.

Details on writing and using plugins can be found in the [Allegro tutorial](https://colab.research.google.com/drive/1yq2UwnET4loJYg_Fptt9kpklVaZvoHnq) and in [`nequip-example-extension`](https://github.com/mir-group/nequip-example-extension/).

## References & citing

The theory behind NequIP is described in our [article](https://www.nature.com/articles/s41467-022-29939-5) (1). 
NequIP's backend builds on [`e3nn`](https://e3nn.org), a general framework for building E(3)-equivariant 
neural networks (2). If you use this repository in your work, please consider citing `NequIP` (1) and `e3nn` (3):

 1. https://www.nature.com/articles/s41467-022-29939-5
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

under the guidance of [Boris Kozinsky at Harvard](https://mir.g.harvard.edu/).

## Contact, questions, and contributing

If you find a bug or have a proposal for a feature, please post it in the [Issues](https://github.com/mir-group/nequip/issues).
If you have a question, topic, or issue that isn't obviously one of those, try our [GitHub Discussions](https://github.com/mir-group/nequip/discussions).

If you want to contribute to the code, please read ["Contributing to NequIP"](docs/dev/contributing.md).

We can also be reached by email at allegro-nequip@g.harvard.edu.
