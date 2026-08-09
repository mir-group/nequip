# Foundation Potentials

Foundation potentials with the NequIP/Allegro architectures are hosted on [nequip.net](https://www.nequip.net/) and automatically accessible through the `nequip` codebase. These are pretrained, wide-purpose interatomic potentials covering most of the periodic table, which can be used directly for [production simulations](./workflow.md#production-simulations) or [fine-tuned](../training-techniques/fine_tuning.md) on your own data.
These models are described in the [NequIP foundation potentials paper](https://doi.org/10.48550/arXiv.2607.28461):

> Seán R. Kavanagh, Chuin Wei Tan, Menghang Wang, Marc L. Descoteaux, Gabriel de Miranda Nascimento, Ulrik Unneberg, Laura Zichi, Francesco Libbi, Norma Rivano, Austin Glover, Vivek Bharadwaj, Anders Johansson, William C. Witt, Albert Musaelian, Boris Kozinsky. <br/>
> "Fast and Accurate Foundation Models for Equivariant Machine-Learned Interatomic Potentials." <br/>
> arXiv:2607.28461 (2026). <br/>
> https://doi.org/10.48550/arXiv.2607.28461

## Available models

[nequip.net](https://www.nequip.net/) lists the currently available models, their architectures, training data, and hyperparameters.
They come in both message-passing (`NequIP-OAM-{S,M,L,XL}`) and strictly local ([`Allegro-OAM-*`](https://github.com/mir-group/allegro)) flavors, and in a range of model sizes trading off accuracy against speed.
The `OAM` models are trained on the OMat24 dataset and then fine-tuned on the subsampled Alexandria (sAlex) and MPtrj datasets.

Models are referred to by an ID of the form `nequip.net:group-name/model-name:version`, e.g. `nequip.net:mir-group/NequIP-OAM-L:0.1`, which can also be passed anywhere a [package file](./files.md#package-files) is expected. If accessed through the `nequip` commands, they are downloaded and cached automatically — see [Compiling models from nequip.net](./workflow.md#compiling-models-from-nequipnet).

## Using a foundation potential for simulations

[Compile](./workflow.md#compilation) the model for the [integration](../../integrations/all.rst) you want to run it with, then use it as usual:

```bash
nequip-compile \
  nequip.net:mir-group/NequIP-OAM-L:0.1 \
  path/to/compiled_model.nequip.pt2 \
  --device cuda \
  --mode aotinductor \
  --target ase \
  --modifiers enable_OpenEquivariance  # recommended GPU kernel acceleration for NequIP models
```

See [ASE](../../integrations/ase.md) and [LAMMPS](../../integrations/lammps/index.md) for running production simulations, and [Accelerations](../accelerations/index.rst) for the [GPU kernel](../accelerations/gpu_kernel_modifiers.md) and [precision](../accelerations/precision.md) options that make the biggest difference to inference speed.

## Fine-tuning a foundation potential

Foundation potentials are distributed as [package files](./files.md#package-files) and can be used as the starting point for a new `nequip-train` run, which is usually far cheaper and more data-efficient than training from scratch.
See the [Fine-Tuning](../training-techniques/fine_tuning.md) page for the full recipe and tips, including how to set the cutoff radius, atom types, and per-type energy shifts of your fine-tuning dataset.

## Training your own foundation potential

The [architecture presets](../configuration/model.md#architecture-presets) ({func}`~nequip.model.PresetNequIPGNNModel`) reproduce the model sizes of the NequIP foundation potentials, and the accelerations documented under [Accelerations](../accelerations/index.rst) — [train-time compilation](../accelerations/pt2_compilation.md), [GPU kernel modifiers](../accelerations/gpu_kernel_modifiers.md), [mixed precision](../accelerations/precision.md), and [multi-GPU training](../accelerations/ddp_training.md) — are what make training on ultra-large datasets affordable. Their combined effect is benchmarked in the [foundation potentials paper](https://doi.org/10.48550/arXiv.2607.28461).

```{note}
Please cite the [NequIP foundation potentials paper](https://doi.org/10.48550/arXiv.2607.28461) (and the [NequIP infrastructure paper](https://doi.org/10.1039/D5DD00423C)) if you fine-tune or use the NequIP/Allegro foundation potentials.
See [References & citing](../../introduction/intro.md#references--citing) for the full citation details.
```
