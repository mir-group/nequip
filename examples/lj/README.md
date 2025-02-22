Lennard-Jones Custom Module Example
===================================

> [!WARNING]
> This example is outdated, open a GitHub issue if a custom module example is needed.

Note: for production simulations, a more appropriate Lennard-Jones energy term is provided in `nequip.model.PairPotentialTerm` / `nequip.model.PairPotential`.

Run commands with
```
PYTHONPATH=`pwd`:$PYTHONPATH nequip-* ...
```
so that the model from `lj.py` can be imported.

For example, to create a deployed LJ model `lj.pth`:
```bash
PYTHONPATH=`pwd`:$PYTHONPATH nequip-deploy build --model lj.yaml lj.pth
```