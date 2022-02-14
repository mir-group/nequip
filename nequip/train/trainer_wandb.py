import wandb

from .trainer import Trainer


class TrainerWandB(Trainer):
    """Trainer class that adds WandB features"""

    def end_of_epoch_log(self):
        Trainer.end_of_epoch_log(self)
        wandb.log(self.mae_dict)

    def init(self):
        super().init()

        if not self._initialized:
            return

        if self.kwargs.get("wandb_watch", False):
            wandb_watch_kwargs = self.kwargs.get("wandb_watch_kwargs", {})
            wandb.watch(self.model, **wandb_watch_kwargs)
