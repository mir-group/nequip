# Fine-Tuning

Fine-tuning allows you to adapt a pretrained model to new datasets. This is useful when you have a model trained on one system and want to adapt it to a similar but different system.

Pretrained models for fine-tuning can be found at [nequip.net](https://www.nequip.net/) in the [packaged file format](../getting-started/files.md#package-files).


## Loading a Packaged Model

Instead of building a model from scratch, use {func}`~nequip.model.ModelFromPackage` to load a pretrained model:

```yaml
model:
  _target_: nequip.model.ModelFromPackage
  package_path: path/to/model.nequip.zip
```


## Extracting Model Type Names

When fine-tuning a packaged model, you need to know what atom types the original model was trained on so that data processing is performed consistently with the pretrained model.
Use the `type_names_from_package` resolver to extract this information:

```yaml
model_type_names: ${type_names_from_package:path/to/model.nequip.zip}

data:
  # ...
  transforms:
    - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
      model_type_names: ${model_type_names}
      # ...
  stats_manager:
    type_names: ${model_type_names}
```

This pattern ensures that your data transforms and statistics are configured with the same atom types as the pretrained model.

Note that {class}`~nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper` has a `chemical_species_to_atom_type_map` argument that is omitted in the example above.
This argument defaults to an identity mapping when the type names from the package correspond exactly to chemical species (e.g., "H", "C", "O"), which is the common case for models on [nequip.net](https://www.nequip.net/).
If the pretrained model used custom type names (e.g., "my_H", "carbon"), you would need to explicitly provide the mapping.


## Modifying Per-Type Energy Shifts and Scales

To account for different DFT (or other simulation) settings, which can lead to different reference energies between datasets, users can modify the per-atom-type energy `shifts` and `scales` by specifying `modify_PerTypeScaleShift` with {func}`~nequip.model.modify` (see API for {class}`~nequip.nn.atomwise.PerTypeScaleShift`).
You only need to specify the `shifts` and/or `scales` for atom types in the fine-tuning dataset (no need to specify values for all atom types of the original model).

```{important}
Model modification only works when the atom types in the new dataset are a subset of the original dataset used to train the model. You cannot add new atom types this way.
```

It is recommended to set `shifts` to the isolated atom energies of your fine-tuning dataset's DFT settings.
One can use the original scales (i.e. don't set `scales` when applying the `modify_PerTypeScaleShift` modifier) or initialize `scales` from dataset statistics of the fine-tuning dataset.

```yaml
model:
  _target_: nequip.model.modify
  modifiers:
    - modifier: modify_PerTypeScaleShift
      # isolated atom energies computed with the settings of the fine-tuning dataset
      # (these are random values for illustrative purposes)
      shifts:
        C: -153.123
        H: -13.456
        O: -432.789
      # optionally set scales from dataset statistics
      # scales: ${training_data_stats:per_type_forces_rms}
      shifts_trainable: false
      scales_trainable: false
  model:
    _target_: nequip.model.ModelFromPackage
    package_path: /path/to/pretrained/model.nequip.zip
```

See [Dataset Statistics](../configuration/data.md#dataset-statistics) for more details on configuring dataset statistics.

## Considerations for Fine-Tuning Training
There are a number of considerations and changes you may want to make to training setup and hyperparameters when fine-tuning, rather than training from scratch. This is an active area of research within the field and the `NequIP` userbase.

Key differences to training from scratch are:

- **Decrease the learning rate**: It is typically best to use a lower learning rate for fine-tuning a pre-trained model, compared to the optimal LR for from-scratch training. 
- **Update energy shifts**: As discussed above, you will likely want to update the atomic energy shifts of the model to match the settings (and thus absolute energies) of your data, to ensure smooth fine-tuning.
- **Fixed model hyperparameters**: When fine-tuning, the architecture of the pre-trained model (number of layers _l_-max, radial cutoff etc. – e.g. provided on [nequip.net](https://www.nequip.net/)) cannot be modified. When comparing the performance of fine-tuning and from-scratch training, it is advised to use the same model hyperparameters for appropriate comparison.