from collections import OrderedDict
from copy import deepcopy
from typing import Mapping


class EarlyStopping:
    """
    Early stop conditions

    There are three early stopping conditions:

    1. a value lower than a defined lower bound
    2. a value higher than a defined upper bound
    3. a value hasn't decreased for x epochs within delta range

    Args:

    lower_bounds (dict): define the key and lower bound for condition 1
    upper_bounds (dict): define the key and lower bound for condition 2
    patiences (dict): defined the x epochs for condition 3
    delta (dict): defined the delta range for condition 3. defaults are 0.0
    cumulative_delta (bool): if True, the minimum value recorded for condition 3
                             will not be updated when the newer value only decreases
                             for a tiny value (< delta). default False
    """

    def __init__(
        self,
        lower_bounds: dict = {},
        upper_bounds: dict = {},
        patiences: dict = {},
        delta: dict = {},
        cumulative_delta: bool = False,
    ):

        self.patiences = deepcopy(patiences)
        self.lower_bounds = deepcopy(lower_bounds)
        self.upper_bounds = deepcopy(upper_bounds)
        self.cumulative_delta = cumulative_delta

        self.delta = {}
        self.counters = {}
        self.minimums = {}
        for key, pat in self.patiences.items():
            self.patiences[key] = int(pat)
            self.counters[key] = 0
            self.minimums[key] = None
            self.delta[key] = delta.get(key, 0.0)

            if pat < 1:
                raise ValueError(
                    f"Argument patience for {key} should be positive integer."
                )
            if self.delta[key] < 0.0:
                raise ValueError("Argument delta should not be a negative number.")

        for key in self.delta:
            if key not in self.patiences:
                raise ValueError(f"patience for {key} should be defined")

    def __call__(self, metrics) -> None:

        stop = False
        stop_args = "Early stopping:"
        debug_args = None

        # check whether key in metrics hasn't reduced for x epochs
        for key, pat in self.patiences.items():

            value = metrics[key]
            minimum = self.minimums[key]
            delta = self.delta[key]

            if minimum is None:
                self.minimums[key] = value
            elif value >= (minimum - delta):
                if not self.cumulative_delta and value > minimum:
                    self.minimums[key] = value
                self.counters[key] += 1
                debug_args = f"EarlyStopping: {self.counters[key]} / {pat}"
                if self.counters[key] >= pat:
                    stop_args += f" {key} has not reduced for {pat} epochs"
                    stop = True
            else:
                self.minimums[key] = value
                self.counters[key] = 0

        for key, bound in self.lower_bounds.items():
            if metrics[key] < bound:
                stop_args += f" {key} is smaller than {bound}"
                stop = True

        for key, bound in self.upper_bounds.items():
            if metrics[key] > bound:
                stop_args += f" {key} is larger than {bound}"
                stop = True

        return stop, stop_args, debug_args

    def state_dict(self) -> "OrderedDict[dict, dict]":
        return OrderedDict([("counters", self.counters), ("minimums", self.minimums)])

    def load_state_dict(self, state_dict: Mapping) -> None:
        self.counters = state_dict["counters"]
        self.minimums = state_dict["minimums"]
