import logging
from typing import Union, List

import torch.nn
from ._loss import find_loss_function


class Loss:
    """
    assemble loss function based on key(s) and coefficient(s)

    Args:
        coeffs (dict, str): keys with coefficient and loss function name
        reduction (str): whether the loss is weighted or not
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

        if isinstance(coeffs, str):
            self.coeffs[coeffs] = 1.0
            mseloss = find_loss_function("MSELoss")
            self.funcs[coeffs] = mseloss
        elif isinstance(coeffs, list):
            mseloss = find_loss_function("MSELoss")
            for key in coeffs:
                self.coeffs[key] = 1.0
                self.funcs[key] = mseloss
        elif isinstance(coeffs, dict):
            for key, value in coeffs.items():
                logging.debug(f" parsing {key} {value}")
                func = ["MSELoss"]
                if isinstance(value, (float, int)):
                    coeff = value
                elif isinstance(value, str) or callable(value):
                    coeff = 1.0
                    func = [value]
                elif isinstance(value, (list, tuple)):
                    if isinstance(value[0], (float, int)):
                        coeff = value[0]
                        func = ["MSELoss"] if len(value) == 1 else value[1:]
                    else:
                        coeff = 1.0
                        func = value
                else:
                    raise NotImplementedError(
                        f"expected float, list or tuple, but get {type(value)}"
                    )
                logging.debug(f" parsing {coeff} {func}")
                self.coeffs[key] = coeff
                self.funcs[key] = find_loss_function(*func)
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
            l = self.funcs[key](
                pred=pred, ref=ref, key=key, atomic_weight_on=self.atomic_weight_on
            )

            contrib.update(c)
            loss = loss + self.coeffs[key] * l

        return loss, contrib
