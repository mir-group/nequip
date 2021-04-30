""" Nequip.train.trainer

Todo:

isolate the loss function from the training procedure
enable wandb resume
make an interface with ray

"""

import wandb

from .trainer import Trainer


class TrainerWandB(Trainer):
    """Class to train a model to minimize forces"""

    def __init__(self, **kwargs):
        Trainer.__init__(self, **kwargs)
        wandb.config.update(self.output.updated_dict(), allow_val_change=True)

    def end_of_epoch_log(self):
        Trainer.end_of_epoch_log(self)
        wandb.log(self.mae_dict)

    def init_model(self):

        Trainer.init_model(self)

        # TO DO, this will trigger pickel failure
        # we may need to go back to state_dict method for saving
        # wandb.watch(self.model)
