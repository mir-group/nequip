from typing import Union

from nequip.data import AtomicDataDict
from torch_runstats import RunningStats, Reduction

from ._loss import find_loss_function
from ._key import ABBREV

metrics_to_reduction = {"mae": Reduction.MEAN, "rmse": Reduction.RMS}


class Metrics:
    """Only scalar errors are supported atm."""

    def __init__(
        self,
        components: Union[list, tuple],
    ):

        self.running_stats = {}
        self.per_species = {}
        self.funcs = {}
        for component in components:

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
                key, reduction, params = component
            else:
                raise ValueError(
                    f"tuple should have a max length of 3 but {len(component)} is given"
                )

            functional = params.pop("functional", "L1Loss")

            # default is to flatten the array
            per_species = params.pop("PerSpecies", False)
            dim = params.pop("dim", tuple())

            if key not in self.running_stats:
                self.running_stats[key] = {}
                self.per_species[key] = {}
                self.funcs[key] = find_loss_function(functional, {})

            self.running_stats[key][reduction] = RunningStats(
                dim=dim,
                reduction=metrics_to_reduction.get(reduction, reduction),
                **params,
            )
            self.per_species[key][reduction] = per_species

    def __call__(self, pred: dict, ref: dict):

        metrics = {}
        for key, func in self.funcs.items():
            error = func(
                pred=pred,
                ref=ref,
                key=key,
                atomic_weight_on=False,
                mean=False,
            )

            for reduction, stat in self.running_stats[key].items():

                params = {}
                if self.per_species[key][reduction]:
                    params = {"accumulate_by": ref[AtomicDataDict.SPECIES_INDEX_KEY]}

                if stat._dim == ():
                    metrics[(key, reduction)] = stat.accumulate_batch(
                        error.flatten(), **params
                    )
                else:
                    metrics[(key, reduction)] = stat.accumulate_batch(error, **params)

        return metrics

    def reset(self):
        for stats in self.running_stats.values():
            for stat in stats.values():
                stat.reset()

    def current_result(self):

        metrics = {}
        for key, stats in self.running_stats.items():
            for reduction, stat in stats.items():
                if stat._dim == tuple() or stat._dim == (1,):
                    metrics[(key, reduction)] = stat.current_result().item()
                else:
                    metrics[(key, reduction)] = stat.current_result()
        return metrics

    def flatten_metrics(self, metrics, allowed_species=None):

        flat_dict = {}
        skip_keys = []
        for k, value in metrics.items():

            key, reduction = k
            short_name = ABBREV.get(key, key)

            item_name = f"{short_name}_{reduction}"

            stat = self.running_stats[key][reduction]
            per_species = self.per_species[key][reduction]
            if per_species:

                element_names = (
                    list(range(value.shape[0]))
                    if allowed_species is None
                    else list(allowed_species.keys())
                )

                if stat._dim == tuple():
                    for id_ele, v in enumerate(value):
                        flat_dict[f"{element_name[id_ele]}_{item_name}"] = v

                    flat_dict[f"all_{item_name}"] = value.mean().item()
                else:
                    for id_ele, vec in enumerate(value):
                        ele = element_names[id_ele]
                        for idx, v in enumerate(vec):
                            name = f"{ele}_{item_name}_{idx}"
                            flat_dict[name] = v
                            skip_keys.append(name)

            else:
                if stat._dim == tuple():
                    # a scalar
                    flat_dict[item_name] = value
                else:
                    # a vector
                    for idx, v in enumerate(value):
                        flat_dict[f"{item_name}_{idx}"] = v
        return flat_dict, skip_keys
