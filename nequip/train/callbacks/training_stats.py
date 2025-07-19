# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import lightning
from lightning.pytorch.callbacks import Callback
from nequip.train import NequIPLightningModule


class TrainingStatsMonitor(Callback):
    """Monitor and log detailed training statistics including weights, gradients, and optimizer states.

    This callback provides comprehensive logging of training dynamics to help diagnose training instabilities.
    It tracks weight and gradient statistics as well as internal optimizer states for Adam/AdamW optimizers.

    Args:
        log_freq (int): frequency (in batches) to log statistics
        log_weights (bool): whether to log weight statistics
        log_gradients (bool): whether to log gradient statistics
        log_optimizer_states (bool): whether to log optimizer internal states (Adam/AdamW only)
    """

    def __init__(
        self,
        log_freq: int = 100,
        log_weights: bool = True,
        log_gradients: bool = True,
        log_optimizer_states: bool = True,
    ):
        self.log_freq = log_freq
        self.log_weights = log_weights
        self.log_gradients = log_gradients
        self.log_optimizer_states = log_optimizer_states
        self._prefix = "training_stats"

        # track step count for logging frequency
        self.step_count = 0

        # cache for parameter ID to name mapping
        self._param_id_to_name_cache = None

    def _should_log(self) -> bool:
        """"""
        return self.step_count % self.log_freq == 0

    def _log_weight_stats(self, pl_module: NequIPLightningModule) -> None:
        """"""
        if not self.log_weights:
            return

        stats_dict = {}

        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.data is not None:
                with torch.no_grad():
                    abs_tensor = param.data.abs()
                    stats_dict.update(
                        {
                            f"weights.min/{name}": param.data.min().item(),
                            f"weights.max/{name}": param.data.max().item(),
                            f"weights.mean/{name}": param.data.mean().item(),
                            f"weights.std/{name}": param.data.std().item(),
                            f"weights.absmin/{name}": abs_tensor.min().item(),
                            f"weights.absmax/{name}": abs_tensor.max().item(),
                        }
                    )

        if stats_dict:
            # add prefix to all keys
            prefixed_stats = {f"{self._prefix}.{k}": v for k, v in stats_dict.items()}
            pl_module.log_dict(prefixed_stats, sync_dist=True)

    def _log_gradient_stats(self, pl_module: NequIPLightningModule) -> None:
        """"""
        if not self.log_gradients:
            return

        stats_dict = {}

        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is not None:
                with torch.no_grad():
                    stats_dict.update(
                        {
                            f"gradients.absmax/{name}": param.grad.abs().max().item(),
                            f"gradients.rms/{name}": torch.sqrt(
                                torch.mean(param.grad**2)
                            ).item(),
                        }
                    )

        if stats_dict:
            # add prefix to all keys
            prefixed_stats = {f"{self._prefix}.{k}": v for k, v in stats_dict.items()}
            pl_module.log_dict(prefixed_stats, sync_dist=True)

    def _log_optimizer_states(
        self, trainer: lightning.Trainer, pl_module: NequIPLightningModule
    ) -> None:
        """"""
        if not self.log_optimizer_states:
            return

        optimizers = trainer.optimizers
        if not optimizers:
            return

        # build parameter ID to name mapping if not cached
        if self._param_id_to_name_cache is None:
            self._param_id_to_name_cache = {
                id(param): name for name, param in pl_module.named_parameters()
            }
        stats_dict = {}

        for opt_idx, optimizer in enumerate(optimizers):
            if not hasattr(optimizer, "state_dict"):
                continue

            state_dict = optimizer.state_dict()
            if "state" not in state_dict:
                continue

            for param, param_state in optimizer.state.items():
                param_name = self._param_id_to_name_cache[id(param)]

                # only log for Adam/AdamW optimizers
                if "exp_avg" in param_state and "exp_avg_sq" in param_state:
                    opt_suffix = f"_{opt_idx}" if len(optimizers) > 1 else ""

                    # log momentum estimates (exp_avg)
                    if isinstance(param_state["exp_avg"], torch.Tensor):
                        exp_avg = param_state["exp_avg"]
                        with torch.no_grad():
                            stats_dict.update(
                                {
                                    f"optimizer{opt_suffix}.exp_avg.absmax/{param_name}": exp_avg.abs()
                                    .max()
                                    .item(),
                                    f"optimizer{opt_suffix}.exp_avg.rms/{param_name}": torch.sqrt(
                                        torch.mean(exp_avg**2)
                                    ).item(),
                                }
                            )

                    # log variance estimates (sqrt of exp_avg_sq)
                    if isinstance(param_state["exp_avg_sq"], torch.Tensor):
                        exp_avg_sq = param_state["exp_avg_sq"]
                        with torch.no_grad():
                            sqrt_exp_avg_sq = torch.sqrt(exp_avg_sq)
                            stats_dict.update(
                                {
                                    f"optimizer{opt_suffix}.sqrt_exp_avg_sq.min/{param_name}": sqrt_exp_avg_sq.min().item(),
                                    f"optimizer{opt_suffix}.sqrt_exp_avg_sq.max/{param_name}": sqrt_exp_avg_sq.max().item(),
                                    f"optimizer{opt_suffix}.sqrt_exp_avg_sq.mean/{param_name}": sqrt_exp_avg_sq.mean().item(),
                                }
                            )

        if stats_dict:
            # add prefix to all keys
            prefixed_stats = {f"{self._prefix}.{k}": v for k, v in stats_dict.items()}
            pl_module.log_dict(prefixed_stats, sync_dist=True)

    def on_after_backward(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        """"""
        if self._should_log():
            self._log_gradient_stats(pl_module)

    def on_before_optimizer_step(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        optimizer,
        optimizer_idx: int = 0,
    ) -> None:
        """"""
        if self._should_log():
            self._log_weight_stats(pl_module)
            self._log_optimizer_states(trainer, pl_module)

        # always increment step count
        self.step_count += 1
