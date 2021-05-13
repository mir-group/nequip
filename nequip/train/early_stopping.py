from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Mapping, Optional, cast


class EarlyStopping:
    """
    Early stop conditions

    There are three early stopping conditions:

    1. a value lower than a defined lower bound
    2. a value higher than a defined upper bound
    3. a value hasn't decreased for x epochs within min_delta range

    Args:

    lower_bounds (dict): define the key and lower bound for condition 1
    upper_bounds (dict): define the key and lower bound for condition 2
    patiences (dict): defined the x epochs for condition 3
    min_delta (dict): defined the delta range for condition 3. defaults are 0.0
    cumulative_delta (bool): if True, the minimum value recorded for condition 3
                             will not be updated when the newer value only decreases
                             for a tiny value (< min_delta). default False
    """

    def __init__(
        self,
        lower_bounds: dict = {},
        upper_bounds: dict = {},
        patiences: dict = {},
        min_delta: dict = {},
        cumulative_delta: bool = False,
    ):

        self.patiences = deepcopy(patiences)
        self.lower_bounds = deepcopy(lower_bounds)
        self.upper_bounds = deepcopy(upper_bounds)
        self.cumulative_delta = cumulative_delta

        # self.keys = set(list(self.lower_bounds.keys())) + set(list(self.upper_bounds.keys()))+set(list(self.patiences.keys()))

        self.min_delta = {}
        self.counter = {}
        self.minimums = {}
        for key, pat in self.patiences.items():
            self.patiences[key] = int(pat)
            self.counter[key] = 0
            self.minimums[key] = None
            self.min_delta[key] = min_delta.get(key, 0.0)

            if pat < 1:
                raise ValueError(f"Argument patience for {key} should be positive integer.")
            if self.min_delta[key] < 0.0:
                raise ValueError("Argument min_delta should not be a negative number.")

        for key in self.min_delta:
            if key not in self.patiences:
                raise ValueError(f"patience for {key} should be defined")

    def __call__(self, metrics) -> None:

        stop = False
        stop_args = "Early stopping:"
        debug_args = None

        # check whether key in metrics hasn't reduced for x epochs
        for key, pat in self.patiences.items():

            value = metrics[key]
            minimums = self.minimums[key]
            min_delta = self.min_delta[key]

            if minimums is None:
                minimums = value
            elif value >= (minimums - self.min_delta[key]):
                if not self.cumulative_delta and value > minimums:
                    self.minimums[key] = value
                self.counter[key] += 1
                debug_args = f"EarlyStopping: {self.counter[key]} / {pat}"
                if self.counter[key] >= pat:
                    stop_args += " {key} has not reduced for {pat} epochs")
                    stop = True
            else:
                self.minimums[key] = value
                self.counter[key] = 0

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
        return OrderedDict(
            [("counter", self.counter), ("minimums", self.minimums)]
        )

    def load_state_dict(self, state_dict: Mapping) -> None:
        self.counter = state_dict["counter"]
        self.minimums = state_dict["minimums"]
