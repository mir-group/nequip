import lightning
from lightning.pytorch.callbacks import Callback

from nequip.train import NequIPLightningModule

from typing import List, Dict, Union


class LossCoefficientScheduler(Callback):
    """Schedule loss coefficients over a training run.

    The ``LossCoefficientScheduler`` takes a single argument ``schedule``, which is a list of dicts with keys ``epoch`` (``int``) and ``coeffs`` (``List[float]``). ``coeffs`` must follow the same order as that in the loss ``MetricsManager``.

    When the trainer's epoch counter matches any of the ``epoch`` arguments provided, the loss coefficients will be changed to the corresponding ``coeffs``.

    The ``coeffs`` will be normalized to sum up to 1 in line with the convention of ``MetricsManager``.

    Example usage in config where there are two loss contributions:
    ::

        callbacks:
          - _target_: nequip.train.callbacks.LossCoefficientScheduler
            schedule:
              - epoch: 100
                coeffs: [1,2]
              - epoch: 200
                coeffs: [1,4]

    Args:
        schedule (list): list of dicts with keys ``epoch`` and ``coeffs``
    """

    def __init__(self, schedule: List[Dict[str, Union[int, List[float]]]]):
        self.change_epochs = [pair["epoch"] for pair in schedule]
        self.coeffs_list = [pair["coeffs"] for pair in schedule]
        # sanity check - epochs are >= 0
        assert all([change_epoch >= 0 for change_epoch in self.change_epochs])
        # sanity check - ensure same number of coeffs
        num_coeffs = [len(coeffs) for coeffs in self.coeffs_list]
        assert num_coeffs.count(num_coeffs[0]) == len(num_coeffs)

    def on_train_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        """"""
        # only change loss coefficients at the designated "change epoch"
        if trainer.current_epoch not in self.change_epochs:
            return
        # set the loss coefficients
        idx = self.change_epochs.index(trainer.current_epoch)
        pl_module.loss.set_coeffs(self.coeffs_list[idx])
