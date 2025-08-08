# LAMMPS

[LAMMPS](https://docs.lammps.org/Manual.html) is a production-grade molecular dynamics engine.
The NequIP framework provides two different integrations for LAMMPS.

## `pair_nequip_allegro`

The [`pair_nequip_allegro` page](pair_styles.md) provides details on using the LAMMPS plugins hosted in the [`pair_nequip_allegro` repository](https://github.com/mir-group/pair_nequip_allegro)  provides pair styles to use NequIP framework interatomic potentials directly in LAMMPS.
This repository contains:

- `pair_nequip` for the NequIP message-passing GNN model
- `pair_allegro` for the strictly local Allegro model


## ML-IAP

LAMMPS [ML-IAP](https://docs.lammps.org/Packages_details.html#pkg-ml-iap) is a separate LAMMPS integration that provides a different interface for using machine learning potentials.
The NequIP framework provides a wrapper and workflow tools for ML-IAP.

See [ML-IAP](mliap.md) for detailed instructions.

## Summary

The following table summarizes key differences between the integration approaches:

| <center>Feature</center> | <center>[`pair_nequip_allegro`](pair_styles.md)</center> | <center>ML-IAP</center> |
|:-------:|:-------------------------:|:------:|
| <center>**LAMMPS Compilation**</center> | <center>See [`pair_nequip_allegro` repo](https://github.com/mir-group/pair_nequip_allegro)</center> | <center>See ["Building ML-IAP"](mliap.md#building-ml-iap)</center> |
| <center>**Multirank Support**</center> | <center>`pair_nequip`: Single rank only<br>`pair_allegro`: Multirank</center> | <center>Multirank </center> |
| <center>**Model Preparation**</center> | <center>[`nequip-compile`](../../guide/getting-started/workflow.md#compilation)</center> | <center>[`nequip-prepare-lmp-mliap`](mliap.md#preparing-models-with-nequip-prepare-lmp-mliap)</center> |
| <center>**Acceleration Modifiers**</center> | <center>`pair_nequip`: None<br>`pair_allegro`: [`enable_TritonContracter`](https://nequip.readthedocs.io/projects/allegro/en/latest/guide/triton.html)</center> | <center>NequIP: [`enable_OpenEquivariance`](../../guide/accelerations/openequivariance.md)<br>Allegro: [`enable_CuEquivarianceContracter`](https://nequip.readthedocs.io/projects/allegro/en/latest/guide/cuequivariance.html), [`enable_TritonContracter`](https://nequip.readthedocs.io/projects/allegro/en/latest/guide/triton.html)</center> |

```{toctree}
:maxdepth: 1

pair_styles
mliap
```
