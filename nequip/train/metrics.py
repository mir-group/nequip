from ._loss import find_loss_function
from nequip.utils import RunningStats
from nequip.utils.stats import Reduction

metrics_to_reduction = {"mae": Reduction.MEAN, "rmse": Reduction.RMS}


class Metrics:
    """Only scalar errors are supported atm."""

    def __init__(
        self,
        components: dict,
    ):

        self.running_stats = {}
        self.funcs = {}
        for component in components:

            # parse the input list
            mode = "average"
            functional = "L1Loss"
            reduction = Reduction.MEAN

            if len(component) == 2:
                key, dim = component
            elif len(component) == 3:
                key, dim, reduction = component
            elif len(component) == 4:
                key, dim, reduction, mode = component
            else:
                key, dim, reduction, mode, functional = component

            if key not in self.running_stats:
                self.running_stats[key] = {}
                self.funcs[key] = find_loss_function(functional, {})
            self.running_stats[key][reduction] = RunningStats(
                dim=dim, reduction=metrics_to_reduction.get(reduction, reduction)
            )

    def __call__(self, pred: dict, ref: dict):

        metrics = {}
        for key, func in self.funcs.items():
            error = func(
                pred=pred,
                ref=ref,
                key=key,
                atomic_weight_on=self.atomic_weight_on,
                reduction="sum",
            )
            for reduction, stat in self.running_stats.items():
                metrics[(key, reduction)] = self.running_stats[key][
                    reduction
                ].accumulate_batch(error)
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
