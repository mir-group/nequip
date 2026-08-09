# GPU Kernel Modifiers

NequIP GNN models support several GPU kernel modifiers that can significantly speed up both training and inference by replacing standard operations with optimized implementations.
The resulting train-time and inference-time speed-ups (and reduction in GPU memory demand) are benchmarked in the [NequIP foundation potentials paper](https://doi.org/10.48550/arXiv.2607.28461).

## Available GPU Kernel Modifiers

|  | <center>**OpenEquivariance**</center> | <center>**CuEquivariance** (alpha) </center> |
|--|---------------------------------------|-------------------------------------|
| <center>**Modifier Name**</center> | <center>`enable_OpenEquivariance`</center> | <center>`enable_CuEquivariance`</center> |
| <center>**Supported Devices**</center> | <center>NVIDIA GPUs, AMD GPUs (HIP)</center> | <center>NVIDIA GPUs</center> |
| <center>**Training**</center> | <center>✅ Stable</center> | <center>🔨 Work in progress</center> |
| <center>**[ASE](../../integrations/ase.md) (TorchScript)**</center> | <center>✅ Stable</center> | <center>🔨 Work in progress</center> |
| <center>**[ASE](../../integrations/ase.md) (AOT Inductor)**</center> | <center>✅ Stable (requires PyTorch >= 2.9)</center> | <center>🔨 Work in progress</center> |
| <center>**[LAMMPS ML-IAP](../../integrations/lammps/mliap.md)**</center> | <center>✅ Stable</center> | <center>🔨 Work in progress</center> |

```{toctree}
:maxdepth: 1

openequivariance
cuequivariance
```
