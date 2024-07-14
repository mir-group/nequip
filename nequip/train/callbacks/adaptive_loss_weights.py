from dataclasses import dataclass

from nequip.train import Trainer

from nequip.train._key import ABBREV
import torch

# Making this a dataclass takes care of equality operators, handing restart consistency checks


@dataclass
class SoftAdapt:
    """Adaptively modify `loss_coeffs` through a training run using the SoftAdapt scheme (https://arxiv.org/abs/2403.18122)

    To use this in a training, set in your YAML file:

        end_of_batch_callbacks:
         - !!python/object:nequip.train.callbacks.adaptive_loss_weights.SoftAdapt {"batches_per_update": 20, "beta": 1.0}

    This funny syntax tells PyYAML to construct an object of this class.

    Main hyperparameters are:
        - how often the loss weights are updated, `batches_per_update`
        - how sensitive the new loss weights are to the change in loss components, `beta`
    """

    # user-facing parameters
    batches_per_update: int = None
    beta: float = None
    eps: float = 1e-8  # small epsilon to avoid division by zero
    # attributes for internal tracking
    batch_counter: int = -1
    prev_losses: torch.Tensor = None
    cached_weights = None

    def __call__(self, trainer: Trainer):

        # --- CORRECTNESS CHECKS ---
        assert self in trainer.callback_manager.callbacks["end_of_batch"]
        assert self.batches_per_update >= 1

        # track batch number
        self.batch_counter += 1

        # empty list of cached weights to store for next cycle
        if self.batch_counter % self.batches_per_update == 0:
            self.cached_weights = []

        # --- MAIN LOGIC THAT RUNS EVERY EPOCH ---

        # collect loss for each training target
        losses = []
        for key in trainer.loss.coeffs.keys():
            losses.append(trainer.batch_losses[f"loss_{ABBREV.get(key, key)}"])
        new_losses = torch.tensor(losses)

        # compute and cache new loss weights over the update cycle
        if self.prev_losses is None:
            self.prev_losses = new_losses
            return
        else:
            # compute normalized loss change
            loss_change = new_losses - self.prev_losses
            loss_change = torch.nn.functional.normalize(
                loss_change, dim=0, eps=self.eps
            )
            self.prev_losses = new_losses
            # compute new weights with softmax
            exps = torch.exp(self.beta * loss_change)
            self.cached_weights.append(exps.div(exps.sum() + self.eps))

        # --- average weights over previous cycle and update ---
        if self.batch_counter % self.batches_per_update == 1:
            softadapt_weights = torch.stack(self.cached_weights, dim=-1).mean(-1)
            counter = 0
            for key in trainer.loss.coeffs.keys():
                trainer.loss.coeffs.update({key: softadapt_weights[counter]})
                counter += 1
