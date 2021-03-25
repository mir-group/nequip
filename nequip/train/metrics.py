import logging
from typing import Union, List

import torch.nn
from ._loss import find_loss_function

metric_names = ["error"]
loss_funcs = ["L1Loss"]

class Metrics:
    """ Only scalar errors are supported atm. 
    """
    def __init__(
        self,
        funcs: dict,
        atomic_weight_on: bool = False,
    ):

        self.atomic_weight_on = atomic_weight_on
        self.funcs = {}
        for key, func in funcs.items():
            # TO DO: classification

            if func in metric_names:
                idx = metric_names.index(func)
            elif func in loss_funcs:
                idx = loss_funcs.index(func)
            else:
                raise NotImplementedError("other metrics are not implemented yet")

            loss_name = loss_funcs[idx]

            self.funcs[key] = find_loss_function(loss_name, {})

    def __call__(self, pred: dict, ref: dict):

        metrics = {}
        for key, func in self.funcs.items():
            metrics[key] = func(
                    pred=pred,
                    ref=ref,
                    key=key,
                    atomic_weight_on=self.atomic_weight_on,
                    reduction="sum",
                )

        return metrics
