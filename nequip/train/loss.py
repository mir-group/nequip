import logging
from typing import Union, List

import torch.nn
from ._loss import find_loss_function
from ._key import ABBREV

from torch_runstats import RunningStats, Reduction


class Loss:
    """
    assemble loss function based on key(s) and coefficient(s)

    Args:
        coeffs (dict, str): keys with coefficient and loss function name
        weight (bool): if True, the results will be weighted with the key: AtomicDataDict.WEIGHTS_KEY+key

    Example input dictionaries

    ```python
    'total_energy'
    ['total_energy', 'forces']
    {'total_energy': 1.0}
    {'total_energy': (1.0)}
    {'total_energy': (1.0, 'MSELoss'), 'forces': (1.0, 'L1Loss', param_dict)}
    {'total_energy': (1.0, user_define_callables), 'force': (1.0, 'L1Loss', param_dict)}
    {'total_energy': (1.0, 'MSELoss'),
     'force': (1.0, 'Weighted_L1Loss', param_dict)}
    ```

    If atomic_weight_on is True, all the loss function will be weighed by ref[AtomicDataDict.WEIGHTS_KEY+key] (if it exists)
    The loss function can be a loss class name that is exactly the same (case sensitive) to the ones defined in torch.nn.
    It can also be a user define class type that
        - takes "reduction=none" as init argument
        - uses prediction tensor and reference tensor for its call functions,
        - outputs a vector with the same shape as pred/ref

    """

    def __init__(
        self,
        coeffs: Union[dict, str, List[str]],
        atomic_weight_on: bool = False,
        coeff_schedule: str = "constant",
    ):

        self.atomic_weight_on = atomic_weight_on
        self.coeff_schedule = coeff_schedule
        self.coeffs = {}
        self.funcs = {}

        mseloss = find_loss_function("MSELoss", {})
        if isinstance(coeffs, str):
            self.coeffs[coeffs] = 1.0
            self.funcs[coeffs] = mseloss
        elif isinstance(coeffs, list):
            for key in coeffs:
                self.coeffs[key] = 1.0
                self.funcs[key] = mseloss
        elif isinstance(coeffs, dict):
            for key, value in coeffs.items():
                logging.debug(f" parsing {key} {value}")
                coeff = 1.0
                func = "MSELoss"
                func_params = {}
                if isinstance(value, (float, int)):
                    coeff = value
                elif isinstance(value, str) or callable(value):
                    func = value
                elif isinstance(value, (list, tuple)):
                    # list of [func], [func, param], [coeff, func], [coeff, func, params]
                    if isinstance(value[0], (float, int)):
                        coeff = value[0]
                        if len(value) > 1:
                            func = value[1]
                        if len(value) > 2:
                            func_params = value[2]
                    else:
                        func = value[0]
                        if len(value) > 1:
                            func_params = value[1]
                else:
                    raise NotImplementedError(
                        f"expected float, list or tuple, but get {type(value)}"
                    )
                logging.debug(f" parsing {coeff} {func}")
                self.coeffs[key] = coeff
                self.funcs[key] = find_loss_function(
                    func,
                    func_params,
                )
        else:
            raise NotImplementedError(
                f"loss_coeffs can only be str, list and dict. got {type(coeffs)}"
            )

        for key, coeff in self.coeffs.items():
            self.coeffs[key] = torch.as_tensor(coeff, dtype=torch.get_default_dtype())

    def __call__(self, pred: dict, ref: dict):

        loss = 0.0
        contrib = {}
        for key in self.coeffs:
            _loss = self.funcs[key](
                pred=pred,
                ref=ref,
                key=key,
                atomic_weight_on=self.atomic_weight_on,
                mean=True,
            )
            contrib[key] = _loss
            loss = loss + self.coeffs[key] * _loss

        return loss, contrib


class LossStat:
    def __init__(self, keys):
        self.loss_stat = {"total": RunningStats(dim=tuple(), reduction=Reduction.MEAN)}

    def __call__(self, loss, loss_contrib):
        results = {}
        results["loss"] = self.loss_stat["total"].accumulate_batch(loss)
        for k, v in loss_contrib.items():
            if k not in self.loss_stat:
                self.loss_stat[k] = RunningStats(dim=tuple(), reduction=Reduction.MEAN)
                self.loss_stat[k].to(v.get_device())
            if k != "total":
                results["loss_" + ABBREV.get(k, k)] = self.loss_stat[
                    k
                ].accumulate_batch(v)
            else:
                results["loss"] = self.loss_stat[k].accumulate_batch(v)
        return results

    def reset(self):
        for v in self.loss_stat.values():
            v.reset()

    def to(self):
        for v in self.loss_stat.values():
            v.to()

    def current_result(self):
        results = {
            "loss_" + ABBREV.get(k, k): v.current_result().item()
            for k, v in self.loss_stat.items()
            if k != "total"
        }
        results["loss"] = self.loss_stat["total"].current_result().item()
        return results
