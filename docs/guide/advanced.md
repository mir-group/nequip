# Advanced Training Techniques

## PyTorch 2.0 Compiled Training

As of `nequip-0.7.0`, PyTorch 2.0 compilation utilities are supported to accelerate training. To use `torch.compile` to accelerate training, one has to set `compile_mode: compile` under the `model` section in `training_module` of the config file. For example,
```
model:
    _target_: nequip.model.NequIPGNNModel
    compile_mode: compile
    # other model hyperparameters
```
or
```
model:
    _target_: allegro.model.AllegroModel
    compile_mode: compile
    # other model hyperparameters    
```
Note that `compile_mode` can only be `compile` (use `torch.compile`), `script` (use TorchScript), or `null` (no compilation used). If `compile_mode` is unspecified, it defaults to `script`. 

The startup time for `torch.compile` is longer than TorchScript, but the speed-ups are usually better. 

```{warning}
Train-time compilation will not work if any of the batch dimensions are all never >=2 over all batches. Batch dimensions include the number of frames, atoms, and edges. It is unlikely for this to happen for the "atom" or "edge" batch dimension, but a practical scenario where it could happen for the "frame" batch dimension is when one trains with both `train` and `val` `batch_size: 1` (perhaps for a dataset where each frame contains many atoms). For those interested, this limitation is related to the [0/1 specialization problem](https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ&tab=t.0#heading=h.ez923tomjvyk).
```

```{warning}
We advise using train-time compilation with GPUs. As of PyTorch 2.6.0, there are known cases of **CPU**-specific train-time compilation issues for certain configurations of NequIP and Allegro models. Be cautious when trying to use train-time compilation with CPUs. If you encounter such issues, please open a GitHub issue.
```

## Data-distributed Parallel Training

As of `nequip-0.7.0`, the NequIP training infrastructure is built upon PyTorch Lightning's abstractions. One feature we benefit from is thus their API for enabling data-distributed parallel (DDP) training. See [Lightning's docs](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel) for how to set it up through Lightning's `Trainer`.