from typing import Union
from ._loss import find_loss_function
from nequip.utils import RunningStats
from nequip.utils.stats import Reduction

metrics_to_reduction = {"mae": Reduction.MEAN, "rmse": Reduction.RMS}


class Metrics:
    """Only scalar errors are supported atm."""

    def __init__(
        self,
        components: Union[list, tuple],
    ):

        self.running_stats = {}
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
            dim = params.pop("dim", 1)

            if key not in self.running_stats:
                self.running_stats[key] = {}
                self.funcs[key] = find_loss_function(functional, {})

            self.running_stats[key][reduction] = RunningStats(
                dim=dim,
                reduction=metrics_to_reduction.get(reduction, reduction),
                **params,
            )
        print(self.running_stats)

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
                print("---")
                print(error.shape, stat._dim)

                # if stat.per_species:
                #     metrics[(key, reduction)] = stat.accumulate_batch(
                #         error.mean(dim=tuple(i for i in range(1, len(error.shape))))
                #     )
                if stat._dim != error.shape[1:]:
                    res_dim = tuple(
                        (i for i in range(1 + len(stat._dim), len(error.shape)))
                    )
                    squeeze_mat = error.mean(dim=res_dim)
                    print("squeeze", error.shape, stat._dim, res_dim)
                    print("squeeze", error.mean(dim=res_dim).shape, stat._dim)
                    metrics[(key, reduction)] = stat.accumulate_batch(
                        error.mean(dim=res_dim)
                    )
                else:
                    print("un squeeze", error.shape, stat._dim)
                    metrics[(key, reduction)] = stat.accumulate_batch(error)
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
