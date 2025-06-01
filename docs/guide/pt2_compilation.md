# PyTorch 2.0 Compiled Training

PyTorch 2.0 compilation utilities are provided to accelerate training **provided PyTorch >= 2.6.0 is installed**.
To use `torch.compile` to accelerate training, set `compile_mode: compile` under the `model` section in `training_module` of the config file. For example,
```yaml
model:
    _target_: nequip.model.NequIPGNNModel
    compile_mode: compile
    # other model hyperparameters
```
or
```yaml
model:
    _target_: allegro.model.AllegroModel
    compile_mode: compile
    # other model hyperparameters    
```
Note that `compile_mode` can only be `compile` (use `torch.compile`), or `eager` (no compilation used). If `compile_mode` is unspecified, it defaults to `eager`.
It will take a bit of time (around a minute or more) for the model to be compiled with `torch.compile` before training proceeds, but the speed-ups are worth it.

```{warning}
Train-time compilation will not work if any of the batch dimensions are all never >=2 over all batches. Batch dimensions include the number of frames, atoms, and edges. It is unlikely for this to happen for the "atom" or "edge" batch dimension, but a practical scenario where it could happen for the "frame" batch dimension is when one trains with both `train` and `val` `batch_size: 1` (perhaps for a dataset where each frame contains many atoms).
```

```{warning}
At present we advise against using train-time compilation on CPUs. As of PyTorch 2.6.0, there are known cases of **CPU**-specific train-time compilation issues for certain configurations of NequIP and Allegro models. Be cautious when trying to use train-time compilation with CPUs. If you encounter such issues, please open a GitHub issue.
```