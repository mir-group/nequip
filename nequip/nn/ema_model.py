"""
Custom implementation of an EMA model based on
https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html#torch.optim.swa_utils.AveragedModel

Reason for custom implementation is because FX codegen doesn't play well with the `deepcopy` of PyTorch's implementation,
leading to the creation of additional buffers, resulting in a buffer mismatch between the base model and the EMA model.
An error is then thrown when the PyTorch implementation tries to sync the EMA model's buffers with base model's buffers.

Main differences between PyTorch's implementation and this one:
 - this implementation is more concise and specific to EMA
 - this implementation does not sync buffers (we could add that if we expect buffers to evolve during training)
"""

import torch
from torch.optim.swa_utils import _group_tensors_by_device_and_dtype
import warnings
from typing import List, Optional


class ExponentialMovingAverageModel(torch.nn.Module):
    """Model whose weights are an exponential moving average (EMA) of the weights of a base model.

    Simplified version of the `official PyTorch implementation <https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html#torch.optim.swa_utils.AveragedModel>`_, made specific to EMA, with inspiration from `torch-ema <https://github.com/fadel/pytorch_ema>`_.

    Note that this EMA implementation does not sync buffers and only updates weights.

    Args:
        model (torch.nn.Module): copy of base model (DO NOT PASS IN THE BASE MODEL)
        decay (float): the EMA decay factor
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
    ):
        super().__init__()

        if decay < 0.0 or decay > 1.0:
            raise ValueError(
                f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
            )
        self.decay = decay

        self.module = model
        # counter for weight updates
        self.n_averaged = 0
        # whether the module is currently storing EMA weights or base model weights
        self.ema_weights = True

    def forward(self, *args, **kwargs):
        """"""
        # we will not not actually use this in the NequIPLightningModule
        # because we want to use the base model, which is optimized via torchscript or torch.compile
        # so we load the EMA weights into the base model (and back) for validation and test epochs
        return self.module(*args, **kwargs)

    def swap_parameters(self, model: torch.nn.Module) -> None:
        """Swaps parameters between the EMA model and the base model.

        This function swaps the EMA and base model parameters. This function can be used at the start and end of validation or test epoch such that the EMA model is used for validation and testing, and the weight swapping occurs minimally. The reason for wanting to use the base model instead of this EMA model is because the base model may be optimized via torchscript or torch.compile so we just transfer the EMA weights to the base model to take advantage of its compilation.

        Args:
            model (torch.nn.Module): base model
        """
        for p_self, p_model in zip(self.parameters(), model.parameters()):
            self._swap_tensors(p_self, p_model)
        self.ema_weights = not self.ema_weights

    def _swap_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
        with torch.no_grad():
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
        ema_param_detached: List[Optional[torch.Tensor]] = []
        model_param_detached: List[Optional[torch.Tensor]] = []
        for p_ema, p_model in zip(self.parameters(), model.parameters()):
            p_model_ = p_model.detach().to(p_ema.device)
            ema_param_detached.append(p_ema.detach())
            model_param_detached.append(p_model_)
            # initialization copy handled in this loop
            if self.n_averaged == 0:
                p_ema.detach().copy_(p_model_)

        # EMA update
        if self.n_averaged > 0:
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
                decay = min(self.decay, (1 + self.n_averaged) / (10 + self.n_averaged))
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

        self.n_averaged += 1

    def set_extra_state(self, state):
        self.n_averaged = state["n_averaged"]
        self.ema_weights = state["ema_weights"]
        if self.decay != state["decay"]:
            warnings.warn(
                "EMA decay parameter set is different from the one loaded -- make sure this is intended."
            )
        self.decay = state["decay"]

    def get_extra_state(self):
        return {
            "decay": self.decay,
            "n_averaged": self.n_averaged,
            "ema_weights": self.ema_weights,
        }
