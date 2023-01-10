from torch.utils.tensorboard import SummaryWriter

from .trainer import Trainer, TRAIN, VALIDATION


class TrainerTensorBoard(Trainer):
    """Trainer class that adds WandB features"""

    def end_of_epoch_log(self):
        Trainer.end_of_epoch_log(self)
        kwargs = dict(
            global_step=self.iepoch, walltime=self.mae_dict["cumulative_wall"]
        )
        for k, v in self.mae_dict.items():
            terms = k.split("_")
            if terms[0] in [TRAIN, VALIDATION]:
                header = "/".join(terms[1:])
                self.tb_writer.add_scalar(f"{header}/{terms[0]}", v, **kwargs)
            elif k not in ["cumulative_wall", "epoch"]:
                self.tb_writer.add_scalar(k, v, **kwargs)
        self.tb_writer.flush()

    def init(self):
        super().init()

        if not self._initialized:
            return

        self.tb_writer = SummaryWriter(
            log_dir=f"{self.output.root}/tb_summary/{self.output.run_name}",
        )
