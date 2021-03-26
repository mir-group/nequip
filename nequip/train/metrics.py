from typing import Union

from nequip.data import AtomicDataDict
from nequip.utils import RunningStats
from nequip.utils.stats import Reduction

from ._loss import find_loss_function

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
            reduction = Reduction.MEAN
            params = {}

            if len(component) == 1:
                key = component
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

    def final_stat(self):

        metrics = {}
        for key, func in self.funcs.items():
            for reduction, stat in self.running_stats.items():
                metrics[(key, reduction)] = self.running_stats[key][
                    reduction
                ].current_result()
        return metrics
