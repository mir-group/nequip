# NequIP

NequIP is an open-source deep learning package for learning interatomic potentials using E(3)-equivariant convolutions.

### Requirements

* Python, v3.8+
* PyTorch, v1.8
* Numpy, v1.19.5
* Scipy, v1.6.0
* ASE, v3.20.1

In particular, please be sure to install Python 3.8 and Pytorch 1.8. 

### Installation

* Install [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric), make sure to install this with your correct version of CUDA/CPU: 

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
```

where ```${CUDA}``` should be replaced by either ```cpu```, ```cu101```, ```cu102```, or ```cu111``` depending on your PyTorch installation, for details see [here](https://github.com/rusty1s/pytorch_geometric). 

Then install Pytorch-Geometric from source (do not install it via ```pip install torch-geometric```)

```
pip install git+https://github.com/rusty1s/pytorch_geometric.git
```

* Install [e3nn](https://github.com/e3nn/e3nn): 

```
pip install git+https://github.com/e3nn/e3nn.git 
```

* We use [Weights&Biases](https://wandb.ai) to keep track of experiments. This is not a strict requirement, you can use our software without this, but it may make your life easier. If you want to use it, create an account [here](https://wandb.ai) and install it: 

```
pip install wandb
```

* Install NequIP

```
pip install git+https://github.com/mir-group/nequip.git
```

### Installation Issues

We recommend running the tests using ```pytest```: 

```
pip install pytest
pytest ./tests
```

One some platforms, the installation may complain about the scikit learn installation. If that's the case, specifically install the following scikit-learn version:

```
pip install -U scikit-learn==0.23.0
```

That should fix it.

### Tutorial 

The best way to learn how to use NequIP is [through the tutorial notebook hosted here](https://deepnote.com/project/2412ca93-7ad1-4458-972c-5d5add5a667e) 

### Training a network

To train a network, all you need to is run train.py with a config file that describes your data set and network, for example: 

```
python scripts/train.py configs/example.yaml
```

### References

The theory behind NequIP is described in our preprint [1]. NequIP's backend builds on e3nn, a general framework for building E(3)-equivariant neural networks [2]. 

    [1] https://arxiv.org/abs/2101.03164
    [2] https://github.com/e3nn/e3nn

### Authors

NequIP is being developed by:

    - Simon Batzner
    - Anders Johansson
    - Albert Musaelian
    - Lixin Sun
    - Mario Geiger
    - Tess Smidt

under the guidance of Boris Kozinsky at Harvard.


### Contact

If you have questions, please don't hesitate to reach out at batzner[at]g[dot]harvard[dot]edu. 


### Citing

If you use this repository in your work, plase consider citing us with the following pre-print: 

    [1] https://arxiv.org/abs/2101.03164

