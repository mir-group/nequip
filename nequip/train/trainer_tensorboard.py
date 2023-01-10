from torch.utils.tensorboard import SummaryWriter

from .trainer import Trainer


class TrainerTensorBoard(Trainer):
    """Trainer class that adds WandB features"""

    def end_of_epoch_log(self):
        Trainer.end_of_epoch_log(self)
        # wandb.log(self.mae_dict)
        for k, v in self.mae_dict.items():
            self.tb_writer.add_scalar(k, v, self.iepoch)
        self.tb_writer.flush()

    def init(self):
        super().init()

        if not self._initialized:
            return

        self.tb_writer = SummaryWriter(
            log_dir=f"{self.output.root}/{self.output.run_name}/tb_summary"
        )
