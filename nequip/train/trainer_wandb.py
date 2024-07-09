import wandb

from .trainer import Trainer


class TrainerWandB(Trainer):
    """Trainer class that adds WandB features"""

    def end_of_epoch_log(self):
        super().end_of_epoch_log()
        
        if self.distributed and dist.get_rank() != 0:
            # We should only interface with wandb if we are rank 0
            return
        
        wandb.log(self.mae_dict)

    def init(self):
        super().init()

        if not self._initialized:
            return

        if self.distributed and dist.get_rank() != 0:
            # We should only interface with wandb if we are rank 0
            return
        
        # upload some new fields to wandb
        wandb.config.update({"num_weights": self.num_weights})

        if self.kwargs.get("wandb_watch", False):
            if self.distributed:
                raise NotImplementedError("wandb_watch is not implemented with DDP")
            wandb_watch_kwargs = self.kwargs.get("wandb_watch_kwargs", {})
            wandb.watch(self.model, **wandb_watch_kwargs)
