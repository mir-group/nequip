# Distributed Data Parallel Training

There are two ways to set up multi-rank distributed data parallel (DDP) training runs, depending on whether you are using [PyTorch 2.0 compiled training](pt2_compilation.md) or not.

## Without train-time compilation

If train-time compilation is not used, you can use PyTorch Lightning's {class}`~lightning.pytorch.strategies.DDPStrategy`.

See [Lightning's docs](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel) for how to set it up through Lightning's {class}`~lightning.pytorch.trainer.trainer.Trainer`. 

**NOTE** that it is usually not necessary to explicitly set the {class}`~lightning.pytorch.strategies.DDPStrategy` as an input to the Lightning {class}`~lightning.pytorch.trainer.trainer.Trainer` if the cluster environment is set up to facilitate Lightning's automatic detection of cluster variables. The main aspect that deserves user attention is configuring the job submission script and the relevant `Trainer` arguments (`num_nodes` and sometimes `devices`) correctly.

In general, you can refer to PyTorch Lightning's docs and other Lightning-based references to set up multi-rank training with Lightning's {class}`~lightning.pytorch.strategies.DDPStrategy`. You may need to set cluster-specific environment variables and set up the multi-rank training run differently depending on the cluster, devices, etc.

### SLURM Example

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

## With train-time compilation

If [train-time compilation](pt2_compilation.md) is used, you **must** use NequIP's custom {class}`~nequip.train.SimpleDDPStrategy` in place of PyTorch Lightning's {class}`~lightning.pytorch.strategies.DDPStrategy`.

{class}`~nequip.train.SimpleDDPStrategy` shares the same interface as Lightning's {class}`~lightning.pytorch.strategies.DDPStrategy`, so Lightning's docs are relevant if there is ever a need to set its arguments (which is typically not necessary, but may be useful for certain clusters). {class}`~nequip.train.SimpleDDPStrategy` can also be used if train-time compilation is not used.

Here's an example of how to use this strategy in the config file:

```yaml
trainer:
  _target_: lightning.Trainer
  # other trainer arguments
  devices: ${num_devices}
  num_nodes: ${num_nodes}
  strategy:
    _target_: nequip.train.SimpleDDPStrategy
```

The main difference is that NequIP's custom {class}`~nequip.train.SimpleDDPStrategy` only performs weight gradient syncing once after the complete backwards pass on each rank, while PyTorch Lightning's {class}`~lightning.pytorch.strategies.DDPStrategy` uses {class}`torch.nn.parallel.DistributedDataParallel`, which has more logic to sync the gradients in buckets.

## Important Considerations

### Batch Size and Learning Rate Scaling

The `batch_size` configured under the dataloaders in the `data` [section of the config](config.md/#data) refers to the **per-rank batch size**. When using multiple ranks, this leads to an **effective batch size that is the per-rank batch size times the number of ranks**.

As increasing the number of ranks (while holding the per-rank batch size constant) increases the effective batch size, you should consider adjusting other hyperparameters that you would typically adjust when raising the batch size, such as the learning rate (see [Lightning's docs](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_faq.html#how-should-i-adjust-the-learning-rate-when-using-multiple-devices) for similar advice).

It may be helpful to use a combination of {mod}`omegaconf`'s [variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation), [environment variable resolver](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html#oc-env) and NequIP's custom arithmetic resolver `int_div` to dynamically configure these parameters based on the runtime environment. 

For example, to get the world size as a SLURM environment variable and set the per-rank batch size as the desired effective global batch size divided by the world size, you can use:

```yaml
batch_size: ${int_div:${effective_global_batch_size},${oc.env:SLURM_NTASKS}}
```

where `effective_global_batch_size` is set elsewhere and is interpolated here.

### Validation and Test Metrics

When using DDP, the {class}`torch.utils.data.distributed.DistributedSampler` may duplicate data samples on some devices to ensure all devices have the same batch size if the number of frames in the dataset cannot be evenly distributed to all devices.

Be very careful when reporting validation or test metrics in DDP settings, as data duplication can lead to incorrect metrics. Either ensure data samples can be evenly distributed to all ranks, or perform validation/testing on a single rank.

For more details on handling validation in distributed settings, see [Lightning's documentation](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-loop).