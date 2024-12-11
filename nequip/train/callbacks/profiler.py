import torch
import lightning
from lightning.pytorch.callbacks import Callback
from nequip.data import AtomicDataDict
from nequip.train import NequIPLightningModule

class Profiler(Callback):
    """Proxy class for `TensorBoard Profiler <https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html>`_.

    Example usage in config:
    ::

        trainer:
          ...
          callbacks:
          - _target_: nequip.train.callbacks.Profiler
            trace_output: "./proflog"

    Args:
        trace_output (str): directory where profile data is stored
    """

    def __init__(self, trace_output='proflog'):
        super().__init__()
        self.prof = torch.profiler.profile(
                     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                     on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_output),
                     record_shapes=True,
                     profile_memory=True,
                     with_stack=True
                 )

    def on_train_start(self, trainer, pl_module):
        self.prof.start()
    def on_train_end(self, trainer, pl_module):
        self.prof.stop()
    def on_train_batch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        batch: AtomicDataDict.Type,
        batch_idx: int,
    ):
        """"""
        self.prof.step()
