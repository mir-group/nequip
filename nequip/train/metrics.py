import logging
from typing import Union, List

import torch.nn
from ._loss import find_loss_function

loss_to_metrics = {"mseloss": "rmse", "l1loss": "mae"}
metrics_to_loss = {"rmse": "MSELoss", "mae": "L1Loss"}


class Metrics:
    def __init__(
        self,
        funcs: dict,
        atomic_weight_on: bool = False,
    ):
        for key, func_list in funcs.items():
            for func in func_list:
                assert func.lower() in [
                    "mseloss",
                    "l1loss",
                    "mae",
                    "rmse",
                ], "other metrics are not implemented yet"

        self.atomic_weight_on = atomic_weight_on
        self.funcs = {}
        for key, func_list in funcs.items():
            self.funcs[key] = {}
            for name in func_list:
                metric_name = loss_to_metrics.get(name.lower(), name.lower())
                func_name = metrics_to_loss.get(name.lower(), name)
                self.funcs[key][metric_name] = find_loss_function(func_name)

    def __call__(self, pred: dict, ref: dict):

        metrics = {}
        for key, funcs in self.funcs.items():
            metrics[key] = {
                name: func(
                    pred=pred,
                    ref=ref,
                    key=key,
                    atomic_weight_on=self.atomic_weight_on,
                    mean=False,
                )
                for name, func in funcs.items()
            }

        return metrics
