# Trimmed-down `pytorch_geometric`

NequIP uses the data format and code of the excellent [`pytorch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/) [1, 2] framework. We use, however, only a very limited subset of that library: the most basic graph data structures. 

To avoid adding a large number of unnecessary second-degree dependencies, and to simplify installation, we include and modify here the small subset of `torch_geometric` that is neccessary for our code.

We are grateful to the developers of PyTorch Geometric for their ongoing and very useful work on graph learning with PyTorch.

  [1]  Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric (Version 2.0.1) [Computer software]. https://github.com/pyg-team/pytorch_geometric
  [2]  https://arxiv.org/abs/1903.02428