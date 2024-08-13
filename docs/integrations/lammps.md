# LAMMPS

[LAMMPS](TODO) is an extremely powerful, scalable, flexible, and production-grade molecular dynamics engine. The `nequip` framework provides [`pair_nequip`](TODO), a general interface for the use of `nequip` interatomic potential models in LAMMPS.
For faster and more scalable simulations, we provide [`pair_allegro`](TODO), a more specialized integration for Allegro models which takes advantage of the scalable Allegro architecture to run simulations on more than one GPU (including up to [thousands](TODO)) for larger systems and faster timesteps.

For details on installing and using `pair_nequip` and `pair_allegro`, please refer to the corresponding repositories linked above.