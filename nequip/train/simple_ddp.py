# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from lightning.pytorch.strategies import DDPStrategy


class SimpleDDPStrategy(DDPStrategy):
    """Effectively Lightning's ``DDPStrategy``, but doing manual gradient syncs instead of using PyTorch's ``DistributedDataParallel`` wrapper.

    .. note::
        To use train-time compilation with multi-rank training, this strategy must be used in place of PyTorch Lightning's ``DDPStrategy``.

    Example use in the config file::

      trainer:
        _target_: lightning.Trainer
        # other trainer arguments
        strategy:
          _target_: nequip.train.SimpleDDPStrategy
    """

    def configure_ddp(self) -> None:
        pass

    def post_backward(self, closure_loss: torch.Tensor) -> None:
        """
        Manual syncing of gradients after the backwards pass.
        """
        # cat all gradients into a single tensor for efficiency
        grad_tensors = []
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_tensors.append(param.grad.data.view(-1))

        if grad_tensors:
            # cat and reduce
            flat_grads = torch.cat(grad_tensors)
            # NOTE: averaging (i.e. summing and dividing by number of ranks) is consistent with PyTorch Lightning's `DDPStrategy`
            # in the training loop, we account for this by multiplying the loss by the number of ranks before the backwards call
            if torch.distributed.get_backend() == "gloo":
                torch.distributed.all_reduce(
                    flat_grads, op=torch.distributed.ReduceOp.SUM
                )
                flat_grads /= torch.distributed.get_world_size()
            else:
                torch.distributed.all_reduce(
                    flat_grads, op=torch.distributed.ReduceOp.AVG
                )

            # copy reduced gradients back
            offset = 0
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    numel = param.grad.numel()
                    param.grad.data.copy_(
                        flat_grads[offset : offset + numel].view_as(param.grad.data)
                    )
                    offset += numel
