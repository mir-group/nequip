# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from torch.optim.swa_utils import _group_tensors_by_device_and_dtype

import lightning
from .lightning import NequIPLightningModule
from nequip.utils import RankedLogger

import warnings
from typing import List, Optional, Union, Callable, Any

logger = RankedLogger(__name__, rank_zero_only=True)


class EMALightningModule(NequIPLightningModule):
    """
    An exponential moving average (EMA) of the model weights are maintained. Validation and test metrics will be that of the EMA weight model. If EMA is used, models loaded from checkpoint files (except during restarts) will always be the model with EMA weights. Specifically, the EMA models will be the ones loaded in the ``NequIPCalculator``, compiled with ``nequip-compile``, or packaged with ``nequip-package``.

    Args:
        ema_decay (float): decay constant for the exponential moving average (EMA) of model weights (default ``0.999``)
    """

    def __init__(
        self,
        ema_decay: float = 0.999,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # set up EMA model and perform initialization copy
        self.ema = EMAWeights(self.model, decay=ema_decay)
        self.ema.update_parameters(self.model)
        # because `self.ema` is an `nn.Module` with appropriate `set_extra_state` and `get_extra_state` methods,
        # the correct states should be loaded upon restarts

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[
            torch.optim.Optimizer, lightning.pytorch.core.optimizer.LightningOptimizer
        ],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        """"""
        # note that this function is meant to be overriden by subclasses
        optimizer.step(closure=optimizer_closure)
        # this function is overriden to do the EMA update
        self.ema.update_parameters(self.model)

    # The EMA model and the base model have their weights swapped before and after validation, testing, prediction tasks
    # The purpose is to take advantage of the compiled state of the base model (torchscript or torch.compile)
    # the purpose of the assert is to safeguard against unexpected load scenarios where
    # `self.model` holds the EMA weights and `self.ema` holds the raw weights (we don't expect this situation to happen, but just in case ...)
    # we assume that checkpoints will not be saved in between the start and end of `val`, `test`, `predict`
    # which should be true since Lightning's `ModelCheckpoint` only has a `on_validation_end` hook (no `on_validation_epoch_end` for example), i.e. the hooks we use for the switching should be sufficient for correct behavior
    def _assert_ema_status_and_switch(
        self, expect_ema_module_holds_ema_weights: bool, evaluation_mode: str
    ):
        # a little verbose, but reads easier
        if expect_ema_module_holds_ema_weights != self.ema.is_holding_ema_weights:
            # no need to complicate error statement logic to mention the bool states
            # they can be inferred from the `evaluation_mode`
            raise RuntimeError(
                f"Conflicting EMA states found at {evaluation_mode}. If using `nequip-train` from a checkpoint, the checkpoint is likely corrupted. Otherwise, there is something wrong and a GitHub issue should be reported."
            )
        self.ema.swap_parameters(self.model)

    def on_validation_start(self):
        """"""
        self._assert_ema_status_and_switch(True, "start of validation")

    def on_validation_end(self):
        """"""
        self._assert_ema_status_and_switch(False, "end of validation")

    def on_test_start(self):
        """"""
        self._assert_ema_status_and_switch(True, "start of testing")

    def on_test_end(self):
        """"""
        self._assert_ema_status_and_switch(False, "end of testing")

    def on_predict_start(self):
        """"""
        self._assert_ema_status_and_switch(True, "start of prediction")

    def on_predict_end(self):
        """"""
        self._assert_ema_status_and_switch(False, "end of prediction")

    @property
    def evaluation_model(self) -> torch.nn.Module:
        # === load up EMA weights ===
        # logging for sanity checking, especially useful for diamond inheritance subclasses involving EMA
        logger.info("Loading EMA weights for evaluation model.")
        # we expect `self.model` to contain the raw weights
        self._assert_ema_status_and_switch(True, "loading for evaluation")
        return self.model


class EMAWeights(torch.nn.Module):
    """Exponential moving average (EMA) weights of a base model.

    Simplified version of the `official PyTorch implementation <https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html#torch.optim.swa_utils.AveragedModel>`_, made specific to EMA, with inspiration from `torch-ema <https://github.com/fadel/pytorch_ema>`_.

    Note that this EMA implementation does not sync buffers and only updates weights.
    All methods of this module assume that the base model weights and the EMA weights are on the same device.

    Args:
        model (torch.nn.Module): base model (this module will make copies of its weights)
        decay (float): the EMA decay factor
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        super().__init__()

        if decay < 0.0 or decay > 1.0:
            raise ValueError(
                f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
            )
        self.decay = decay
        # store EMA weights as buffers
        _params = [torch.empty_like(p) for p in model.parameters()]
        for idx, param in enumerate(_params):
            self.register_buffer(f"ema_weight_{idx}", param)
        self.num_ema_weights = len(_params)
        del _params
        # counter for weight updates
        self.num_updates = 0
        # flag to control if this module is holding EMA weights or raw model weights
        # latter is true when `swap_parameters` is called
        self.is_holding_ema_weights = True

    @property
    def ema_weights(self):
        # we adopt a similar solution from the following discussion
        # https://discuss.pytorch.org/t/why-no-nn-bufferlist-like-function-for-registered-buffer-tensor/18884/10
        # since it makes the code that comes below more readable
        # by imitating a `ParameterList` for buffers
        return [
            getattr(self, f"ema_weight_{idx}") for idx in range(self.num_ema_weights)
        ]

    def forward(self, *args, **kwargs):
        """"""
        raise RuntimeError(
            "This module only carries EMA weights, but not a model forward implementation."
        )

    def swap_parameters(self, model: torch.nn.Module) -> None:
        """Swaps parameters between the EMA model and the base model.

        This function swaps the EMA and base model parameters. This function can be used at the start and end of validation or test epoch such that the EMA model is used for validation and testing, and the weight swapping occurs minimally. The reason for wanting to use the base model instead of this EMA model is because the base model may be optimized via TorchScript or ``torch.compile`` so we just transfer the EMA weights to the base model to take advantage of its compilation.

        Args:
            model (torch.nn.Module): base model
        """
        with torch.no_grad():
            for p_self, p_model in zip(self.ema_weights, model.parameters()):
                self._swap_tensors(p_self, p_model)
        self.is_holding_ema_weights = not self.is_holding_ema_weights

    def _swap_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)
        del tmp

    def update_parameters(self, model: torch.nn.Module) -> None:
        """Update model parameters.

        Args:
            model (torch.nn.Module): base model
        """
        assert (
            self.is_holding_ema_weights
        ), "EMA module is not holding EMA weights. If using `nequip-train` from a checkpoint, the checkpoint is likely corrupted. Otherwise, there is something wrong and a GitHub issue should be reported."

        ema_param_detached: List[Optional[torch.Tensor]] = []
        model_param_detached: List[Optional[torch.Tensor]] = []
        for p_ema, p_model in zip(self.ema_weights, model.parameters()):
            ema_param_detached.append(p_ema.detach())
            model_param_detached.append(p_model.detach())
            # initialization copy handled in this loop
            if self.num_updates == 0:
                p_ema.detach().copy_(p_model.detach())

        # EMA update
        if self.num_updates > 0:
            grouped_tensors = _group_tensors_by_device_and_dtype(
                [ema_param_detached, model_param_detached]
            )
            for (device, _), (
                [self_params, model_params],
                _,
            ) in grouped_tensors.items():
                # NOTE: decay used is not just the decay provided but is a function of the number of updates
                # to prevent the weights from early training from contaminating the EMA weights later on
                # from https://github.com/fadel/pytorch_ema
                decay = min(
                    self.decay, (1 + self.num_updates) / (10 + self.num_updates)
                )
                with torch.no_grad():
                    # foreach lerp only handles float and complex
                    if torch.is_floating_point(self_params[0]) or torch.is_complex(
                        self_params[0]
                    ):
                        torch._foreach_lerp_(self_params, model_params, 1 - decay)
                    else:
                        for p_ema, p_model in zip(self_params, model_params):
                            p_ema.copy_(p_ema * decay + p_model * (1 - decay))

        # NOTE: official PyTorch implementation kept buffers in sync with the source model
        # but we don't expect to update our buffers anyway
        # we can always add the syncing back if we expect to update buffers

        self.num_updates += 1

    def set_extra_state(self, state):
        """"""
        self.num_updates = state["num_updates"]
        self.is_holding_ema_weights = state["is_holding_ema_weights"]
        assert (
            self.is_holding_ema_weights
        ), "EMA module loaded in a state where it does not contain EMA weights -- the checkpoint file is likely corrupted."

        # handle possibility of restarts overwriting `decay`
        state_dict_decay = state["decay"]
        if self.decay != state_dict_decay:
            warnings.warn(
                f"EMA decay parameter loaded from state dict ({state_dict_decay}) is different from EMA decay parameter set ({self.decay}) -- make sure this is intended (e.g. you have intentionally overriden `ema_decay` in a restart). The current decay value ({self.decay}) will be used."
            )

    def get_extra_state(self):
        """"""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "is_holding_ema_weights": self.is_holding_ema_weights,
        }
