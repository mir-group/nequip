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

        wandb_watch = self.kwargs.get("wandb_watch", False)
        if wandb_watch is not False:
            if wandb_watch is True:
                wandb_watch = {}
            wandb.watch(self.model, **wandb_watch)
