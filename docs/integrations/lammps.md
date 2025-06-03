# LAMMPS

[LAMMPS](https://docs.lammps.org/Manual.html) is a production-grade molecular dynamics engine. 
The [`pair_nequip_allegro`](https://github.com/mir-group/pair_nequip_allegro) repository is an interface to use NequIP framework interatomic potentials in LAMMPS.
This repository provides two pair styles: `pair_nequip` is for the NequIP message-passing GNN model, which is limited to one MPI rank; `pair_allegro` is for the strictly local Allegro model, which supports parallel execution and MPI in LAMMPS.

```{important}
You must [compile](../guide/getting-started/workflow.md#compilation) the NequIP framework models before using them in LAMMPS. If you compile with `--mode aotinductor`, you must specify `--target pair_nequip` if you intend to use the model with the `pair_nequip` pair style, or `--target pair_allegro` if you intend to use the model with the `pair_allegro` pair style. The `allegro` package must be installed in your Python environment in order to perform `nequip-compile` with the `--target pair_allegro` flag.
```