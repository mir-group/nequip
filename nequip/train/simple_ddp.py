# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from lightning.pytorch.strategies import DDPStrategy


class SimpleDDPStrategy(DDPStrategy):
    """Effectively Lightning's :class:`~lightning.pytorch.strategies.DDPStrategy`, but doing manual gradient syncs instead of using PyTorch's :class:`~torch.nn.parallel.DistributedDataParallel` wrapper.

    .. note::
        To use train-time compilation with multi-rank training, this strategy must be used in place of PyTorch Lightning's :class:`~lightning.pytorch.strategies.DDPStrategy`.

    Example use in the config file:

    .. code-block:: yaml

      trainer:
        _target_: lightning.Trainer
        # other trainer arguments
        strategy:
          _target_: nequip.train.SimpleDDPStrategy
    """

    def configure_ddp(self) -> None:
        pass

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        """Move the model to its device *before* restoring the checkpoint.

        ``OpenEquivariance`` modules can wedge if state is restored onto them
        before the device move; move-then-restore avoids it.
        """
        return True

    def load_checkpoint(self, checkpoint_path, weights_only=None):
        """Load the resume checkpoint straight onto this rank's GPU, ``weights_only=False``.

        ``map_location=self.root_device`` loads the entire checkpoint (model, optimizer,
        EMA state) onto this rank's GPU, overriding the saved device index. ``weights_only``
        is forced to ``False`` to allow the optimizer/EMA pickled state through.
        """
        if weights_only is None:
            weights_only = False
        torch.cuda.empty_cache()
        return self.checkpoint_io.load_checkpoint(
            checkpoint_path, map_location=self.root_device, weights_only=weights_only
        )

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
