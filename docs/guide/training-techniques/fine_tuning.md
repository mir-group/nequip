# Fine-Tuning

Fine-tuning allows you to adapt a pretrained model to new datasets. This is useful when you have a model trained on one system and want to adapt it to a similar but different system.


## Modifying Per-Type Energy Shifts and Scales

To account for different DFT (or other simulation) settings, which can lead to difference reference energies between datasets, users can modify the per-atom-type energy `shifts` and `scales` by specifying `modify_PerTypeScaleShift` with {func}`~nequip.model.modify` (see API for {class}`~nequip.nn.atomwise.PerTypeScaleShift`).
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
      # scales from dataset statistics
      scales: ${training_data_stats:per_type_forces_rms}
      # or use the scales from the original model
      # scales: null
      shifts_trainable: false
      scales_trainable: false
  model:
    _target_: nequip.model.ModelFromPackage
    package_path: /path/to/pretrained/model.nequip.zip
```

See [Dataset Statistics](../configuration/data.md#dataset-statistics) for more details on configuring dataset statistics.
