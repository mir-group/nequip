from copy import deepcopy
from hashlib import sha1
from typing import Union, Sequence, Tuple

import yaml

import torch
import numpy as np

from nequip.data import AtomicDataDict
from torch_runstats import RunningStats, Reduction

from ._loss import find_loss_function
from ._key import ABBREV

metrics_to_reduction = {
    "mae": Reduction.MEAN,
    "mean": Reduction.MEAN,
    "rmse": Reduction.RMS,
}


class Metrics:
    """Computes all mae, rmse needed for report

    Args:
        components: a list or a tuples of definition.

    Example:

    ```
    components = [(key1, "rmse"), (key2, "mse")]
    ```

    You can also offer more details with a dictionary. The keys can be any keys for RunningStats or

    report_per_component (bool): if True, report the mean on each component (equivalent to mean(axis=0) in numpy),
                                 otherwise, take the mean across batch and all components for vector data.
    functional: the function to compute the error. It has to be exactly the same as the one defined in torch.nn.
                Callables are also allowed.
                default: "L1Loss"
    PerSpecies: whether to compute the estimation for each species or not

    the keys are case-sensitive.


    ```
    components = (
        (
            AtomicDataDict.FORCE_KEY,
            "rmse",
            {"PerSpecies": True, "functional": "L1Loss", "dim": 3},
        ),
        (AtomicDataDict.FORCE_KEY, "mae", {"dim": 3}),
    )
    ```

    """

    def __init__(
        self, components: Sequence[Union[Tuple[str, str], Tuple[str, str, dict]]]
    ):

        self.running_stats = {}
        self.params = {}
        self.funcs = {}
        self.kwargs = {}
        self.stratified_stats = (
            {}
        )  # need to be stored separately, as needs all data labels at once

        for component in components:

            key, reduction, params = Metrics.parse(component)

            params["stratify"] = params.get(
                "stratify", False
            )  # can be either 'XX%_range', 'XX%_population' or int/float (raw unit for separation)
            params["PerSpecies"] = params.get("PerSpecies", False)
            params["PerAtom"] = params.get("PerAtom", False)

            param_hash = Metrics.hash_component(component)

            functional = params.get("functional", "L1Loss")

            if key not in self.kwargs:
                self.funcs[key] = {}
                self.kwargs[key] = {}
                self.params[key] = {}

            if key not in self.running_stats and not params.get("stratify", False):
                self.running_stats[key] = {}  # default is to flatten the array

            if key not in self.stratified_stats and params.get("stratify", False):
                self.stratified_stats[key] = {}

            # store for initialization
            kwargs = deepcopy(params)
            kwargs.pop("functional", "L1Loss")
            kwargs.pop("stratify")
            kwargs.pop("PerSpecies")
            kwargs.pop("PerAtom")

            # by default, report a scalar that is mae and rmse over all component
            self.kwargs[key][param_hash] = dict(
                reduction=metrics_to_reduction.get(reduction, reduction),
            )
            self.kwargs[key][param_hash].update(kwargs)
            self.params[key][param_hash] = (reduction, params)
            self.funcs[key][param_hash] = find_loss_function(functional, {})

    def init_runstat(self, params, error: torch.Tensor):
        """
        Initialize Runstat Counter based on the shape of the error matrix

        Args:
        params (dict): dictionary of additional arguments
        error (torch.Tensor): error matrix
        """

        kwargs = deepcopy(params)
        # automatically define the dimensionality
        if "dim" not in kwargs:
            kwargs["dim"] = error.shape[1:]

        if "reduce_dims" not in kwargs:
            if not kwargs.pop("report_per_component", False):
                kwargs["reduce_dims"] = tuple(range(len(error.shape) - 1))

        rs = RunningStats(**kwargs)
        rs.to(device=error.device)
        return rs

    @staticmethod
    def hash_component(component):
        buffer = yaml.dump(component).encode("ascii")
        return sha1(buffer).hexdigest()

    @staticmethod
    def parse(component):
        # parse the input list
        reduction = "mae"
        params = {}

        if isinstance(component, str):
            key = component
        elif len(component) == 1:
            key = component[0]
        elif len(component) == 2:
            key, reduction = component
        elif len(component) == 3:
            key, reduction, _params = component
            params = {k: deepcopy(v) for k, v in _params.items()}
        else:
            raise ValueError(
                f"tuple should have a max length of 3 but {len(component)} is given"
            )
        return key, reduction, params

    def __call__(self, pred: dict, ref: dict):

        metrics = {}
        N = None
        for key in self.funcs.keys():
            for param_hash, kwargs in self.kwargs[key].items():
                func = self.funcs[key][param_hash]
                error = func(
                    pred=pred,
                    ref=ref,
                    key=key,
                    mean=False,
                )

                _, params = self.params[key][param_hash]
                stratify = params["stratify"]
                per_species = params["PerSpecies"]
                per_atom = params["PerAtom"]

                if not stratify:
                    # initialize the internal run_stat base on the error shape
                    if param_hash not in self.running_stats[key]:
                        self.running_stats[key][param_hash] = self.init_runstat(
                            params=kwargs, error=error
                        )

                    stat = self.running_stats[key][param_hash]

                params = {}
                if per_species:
                    if stratify:
                        raise NotImplementedError(
                            "Stratify is not implemented for per_species"
                        )
                    # TODO, this needs OneHot component. will need to be decoupled
                    params = {
                        "accumulate_by": pred[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
                    }
                if per_atom:
                    if N is None:
                        N = torch.bincount(ref[AtomicDataDict.BATCH_KEY]).unsqueeze(-1)
                    error_N = error / N
                else:
                    error_N = error

                if stat.dim == () and not per_species:
                    error_N = error_N.flatten()

                if (  # just need error and ref value, note that forces are not stratified by xyz
                    stratify  # norm (just raw x, y, z values used)
                ):
                    if param_hash not in self.stratified_stats[key]:
                        self.stratified_stats[key][param_hash] = {
                            "error": error_N,
                            "ref_val": ref[key],
                        }
                    self.stratified_stats[key][param_hash]["error"] = torch.cat(
                        (self.stratified_stats[key][param_hash]["error"], error_N)
                    )
                    self.stratified_stats[key][param_hash]["ref_val"] = torch.cat(
                        (self.stratified_stats[key][param_hash]["ref_val"], ref[key])
                    )

                else:
                    metrics[(key, param_hash)] = stat.accumulate_batch(
                        error_N, **params
                    )

        return metrics

    def reset(self):
        for stats in self.running_stats.values():
            for stat in stats.values():
                stat.reset()

    def to(self, device):
        for stats in self.running_stats.values():
            for stat in stats.values():
                stat.to(device=device)

    def current_result(self, verbose=False):
        """
        Return the current result of the metrics.

        Args:
            verbose (bool):
                If True, prints information about stratified metrics (i.e. ranges).
                Default: False
        """

        metrics = {}
        for key, stats in self.running_stats.items():
            for param_hash, stat in stats.items():
                metrics[(key, param_hash)] = stat.current_result()

        for key, stats in self.stratified_stats.items():
            for (
                param_hash,
                stratified_stat_dict,
            ) in stats.items():  # compute the stratified error:
                reduction, params = self.params[key][param_hash]
                errors = stratified_stat_dict["error"]
                ref_vals = stratified_stat_dict["ref_val"]
                stratified_metric_dict = {}

                if isinstance(params.get("stratify"), str) and "range" in params.get(
                    "stratify"
                ):  # stratify by range (given as percent string)
                    min_max_range = (ref_vals.max() - ref_vals.min()).cpu().numpy()
                    range_separation = (
                        float(params["stratify"].strip("%_range")) / 100
                    ) * min_max_range
                    if verbose:
                        print(
                            f"Stratifying {key} errors by {key} range, in increments of "
                            f"{params['stratify'].strip('_range')} (= {range_separation:.3f}), with"
                            f" min-max dataset range of {min_max_range:.3f}"
                        )

                    num_strata = np.ceil(min_max_range / range_separation).astype(int)
                    format = (  # .1% if 1/num_strata is not an integer, otherwise .0% (no decimal)
                        ".1%"
                        if not np.isclose(
                            (1 / num_strata) * 100, round((1 / num_strata) * 100)
                        )
                        else ".0%"
                    )

                    for i in range(num_strata):
                        mask = (ref_vals >= (i * range_separation) + ref_vals.min()) & (
                            ref_vals < ((i + 1) * range_separation) + ref_vals.min()
                        )
                        masked_errors = errors[mask]
                        if len(masked_errors) > 0:
                            if reduction == "rms":
                                stat = masked_errors.square().mean().sqrt()
                            elif reduction in ["mean", "mae"]:
                                stat = masked_errors.mean()
                            else:
                                raise NotImplementedError(
                                    f"reduction {reduction} not implemented"
                                )

                            stratified_metric_dict[
                                (
                                    f"{i/num_strata:{format}}"
                                    f"-{(i+1)/num_strata:{format}}"
                                )
                            ] = stat

                        else:
                            stratified_metric_dict[
                                f"{i/num_strata:{format}}-{(i+1)/num_strata:{format}}"
                            ] = torch.tensor(float("nan"))

                # elif params.get("population", False):  # stratify by population
                # ... # TODO

                metrics[(key, param_hash)] = stratified_metric_dict

        return metrics

    def flatten_metrics(self, metrics, type_names=None):

        flat_dict = {}
        skip_keys = []
        for k, value in metrics.items():

            key, param_hash = k
            reduction, params = self.params[key][param_hash]

            short_name = ABBREV.get(key, key)
            if hasattr(self.funcs[key][param_hash], "get_name"):
                short_name = self.funcs[key][param_hash].get_name(short_name)

            per_atom = params["PerAtom"]
            suffix = "/N" if per_atom else ""
            item_name = f"{short_name}{suffix}_{reduction}"

            per_species = params["PerSpecies"]
            stratify = params["stratify"]

            if stratify:  # then value is a dict of {stratum_idx: value}
                for stratum_idx, v in value.items():
                    name = f"{stratum_idx}_{item_name}"
                    flat_dict[name] = v.item()
                    skip_keys.append(name)

            else:
                stat = self.running_stats[key][param_hash]

                if per_species:
                    stat = self.running_stats[key][param_hash]
                    if stat.output_dim == tuple():
                        if type_names is None:
                            type_names = [i for i in range(len(value))]
                        for id_ele, v in enumerate(value):
                            if type_names is not None:
                                flat_dict[f"{type_names[id_ele]}_{item_name}"] = (
                                    v.item()
                                )
                            else:
                                flat_dict[f"{id_ele}_{item_name}"] = v.item()

                        flat_dict[f"psavg_{item_name}"] = value.mean().item()
                    else:
                        for id_ele, vec in enumerate(value):
                            ele = type_names[id_ele]
                            for idx, v in enumerate(vec):
                                name = f"{ele}_{item_name}_{idx}"
                                flat_dict[name] = v.item()
                                skip_keys.append(name)

                else:
                    if stat.output_dim == tuple():  # a scalar
                        flat_dict[item_name] = value.item()
                    else:  # a vector
                        for idx, v in enumerate(value.flatten()):
                            flat_dict[f"{item_name}_{idx}"] = v.item()

        return flat_dict, skip_keys
