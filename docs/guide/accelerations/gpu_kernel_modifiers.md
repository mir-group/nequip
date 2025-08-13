# GPU Kernel Modifiers

NequIP GNN models support several GPU kernel modifiers that can significantly speed up both training and inference by replacing standard operations with optimized implementations.

## Available GPU Kernel Modifiers

|  | <center>**OpenEquivariance**</center> | <center>**CuEquivariance**</center> |
|--|---------------------------------------|-------------------------------------|
| <center>**Modifier Name**</center> | <center>`enable_OpenEquivariance`</center> | <center>`enable_CuEquivariance`</center> |
| <center>**Supported Devices**</center> | <center>NVIDIA GPUs, AMD GPUs (HIP)</center> | <center>NVIDIA GPUs</center> |
| <center>**Training**</center> | <center>✅ Stable</center> | <center>🔨 Work in progress</center> |
| <center>**[ASE](../../integrations/ase.md) (TorchScript)**</center> | <center>✅ Stable</center> | <center>✅ Stable</center> |
| <center>**[ASE](../../integrations/ase.md) (AOT Inductor)**</center> | <center>🔨 Work in progress</center> | <center>✅ Stable</center> |
| <center>**[LAMMPS ML-IAP](../../integrations/lammps/mliap.md)**</center> | <center>✅ Stable</center> | <center>✅ Stable</center> |

```{toctree}
:maxdepth: 1

openequivariance
cuequivariance
```