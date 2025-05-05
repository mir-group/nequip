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

## Distributed Data Parallel Training

As of `nequip-0.7.0`, the NequIP training infrastructure is built upon PyTorch Lightning's abstractions. One feature we benefit from is thus their API for enabling distributed data parallel (DDP) training. There are two ways of setting up multi-rank training runs.

1. If train-time compilation is not used, one can use PyTorch Lightning's [DDPStrategy](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html#lightning.pytorch.strategies.DDPStrategy).
See [Lightning's docs](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel) for how to set it up through Lightning's [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api).
**NOTE** that it is usually not necessary to explicitly set the [DDPStrategy](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html#lightning.pytorch.strategies.DDPStrategy) as an input to the Lightning [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) if the cluster environment is set up to facilitate Lightning's automatic detection of cluster variables, and the main aspect that deserves user attention is configuring the job submission script and the relevant `Trainer` arguments (`num_nodes` and sometimes `devices`) correctly.
In general, one can refer to PyTorch Lightning's docs and other Lightning-based references to set up multi-rank training with Lightning's `DDPStrategy`.
It is likely that one may need to set cluster-specific environment variables and set up the multi-rank training run differently depending on the cluster, devices, etc.
A useful resource for SLURM-managed clusters is PyTorch Lightning's [docs for SLURM-managed clusters](https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html), which details, for instance the need to use `srun nequip-train ...`, and which SLURM variables correspond to arguments in the `trainer` section of the config file.
A minimal SLURM example for doing DDP training with 2 nodes with 4 GPUs per node (8 GPUs in total) is shown as follows.

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
... other slurm variables

# ... set up (e.g. module load, activate Python env, etc)

# ... cluster specific set up such as network interface 
# (e.g. MASTER_PORT, MASTER_ADDR, NCCL_SOCKET_IFNAME)

srun nequip-train -cn config.yaml ++trainer.num_nodes=${SLURM_NNODES}
```

2. If train-time compilation is used, one **must** use NequIP's custom ``nequip.train.SimpleDDPStrategy`` ([API docs](../../api/ddp)) in place of PyTorch Lightning's [DDPStrategy](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html#lightning.pytorch.strategies.DDPStrategy).
``nequip.train.SimpleDDPStrategy`` shares the same interface as Lightning's [DDPStrategy](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html#lightning.pytorch.strategies.DDPStrategy), so Lightning's docs are relevant if there is ever a need to set its arguments (which is typically not necessary, but may be useful for certain clusters).
``nequip.train.SimpleDDPStrategy`` can also be used if train-time compilation is not used.
Here's an example of how one can use this strategy in the config file.

```bash
trainer:
  _target_: lightning.Trainer
  # other trainer arguments
  devices: ${num_devices}
  num_nodes: ${num_nodes}
  strategy:
    _target_: nequip.train.SimpleDDPStrategy
```

The main difference is that NequIP's custom ``nequip.train.SimpleDDPStrategy`` only performs weight gradient syncing once after the complete backwards pass on each rank, while PyTorch Lightning's ``DDPStrategy`` uses [``torch.nn.parallel.DistributedDataParallel``](https://pytorch.org/docs/stable/notes/ddp.html), which has more logic to sync the gradients in buckets.

```{warning}
NOTE that the `batch_size` configured under the dataloaders in the `data` [section of the config](config.md/#data) refers to the per-rank batch size, so using multiple ranks will lead to an effective batch size that is the per-rank batch size times the number of ranks.

As increasing the number of ranks (while holding the per-rank batch size constant) increases the effective batch size, one should consider adjusting other hyperparameters that one would typically adjust when raising the batch size, such as the learning rate (see [Lightning's docs](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_faq.html#how-should-i-adjust-the-learning-rate-when-using-multiple-devices) for similar advice).

It may be helpful to use a combination of OmegaConf's [variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation), [environment variable resolver](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html#oc-env) and NequIP's custom arithmetic resolver `int_div` to dynamically configure these parameters based on the runtime environment. 
For example, to get the world size as a SLURM environment variable and set the per-rank batch size as the desired effective global batch size divided by the world size, one can use something like

`batch_size: ${int_div:${effective_global_batch_size},${oc.env:SLURM_NTASKS}}` 

where `effective_global_batch_size` is set elsewhere and is interpolated here.
```


```{warning}
Be very careful when reporting validation or test metrics calculated in a DDP setting. The [``DistributedSampler``](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) may duplicate data samples on some devices to make sure all devices have the same batch size if the number of frames in the dataset cannot be evenly distributed to all devices. Either ensure that the data samples can be evenly distributed to all ranks, or perform validation/testing on a single rank. See [Lightning's docs](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-loop) for similar advice.
```
