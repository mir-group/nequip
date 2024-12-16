import torch
from torchmetrics import Metric
from nequip.data import AtomicDataDict, BaseModifier

from typing import List, Dict, Union, Callable, Final

_METRICS_MANAGER_INPUT_KEYS: Final[str] = [
    "field",
    "metric",
    "name",
    "coeff",
    "per_type",
    "ignore_nan",
]


class MetricsManager(torch.nn.ModuleList):
    """Manages ``nequip`` metrics that can be applied to ``AtomicDataDict`` to compute error metrics.

    This class manages both metrics for loss functions and metrics for monitoring and reporting purposes. The main input argument ``metrics`` is a list of dictionaries, where each dictionary contains the following keys.

    There are two mandatory keys.

      - ``field`` refers to the quantity of interest for metric computation. It has two formats.

         - a ``str`` for a ``nequip`` defined field (e.g. ``total_energy``, ``forces``, ``stress``), or
         - a ``Callable`` that performs some additional operations before returning a ``torch.Tensor``
           for metric computation (e.g. ``nequip.data.PerAtomModifier``).
      - ``metric`` is a ``torchmetrics.Metric``. Users are expected to mostly use ``nequip.train.MeanSquaredError`` and ``nequip.train.MeanAbsoluteError`` for MSEs (for loss), RMSEs, and MAEs (for monitoring).

    The remaining keys are optional.

      - ``per_type`` is a ``bool`` (defaults to ``False`` if not provided). If ``True``, node fields (such as ``forces``) will have their metrics computed separately for each atom type based on the ``type_names`` argument. A simple average over the per-type metrics will be used as the "effective" metric returned. There are some subtleties. 1) During batch steps, the per-type metric for a particular type may be ``NaN`` (by design) if the batch does not contain that atom type. Correspondingly, the per-batch effective metric does not account for that particular type, and is only averaged over the number of atom types that were in that batch. 2) The per-epoch effective metric will always consider all atom types configured in its computation (i.e. when taking the simple average) since all atom types are expected to have contributed to at least one batch step over an epoch.

      - ``coeff`` is a ``float`` that determines the relative weight of the metric contribution to an overall ``weighted_sum`` of metrics. A ``weighted_sum`` is automatically computed if any input dictionary contains a float-valued ``coeff``. This feature is important for loss functions for multitask problems (e.g. training on ``total_energy`` and ``forces`` simultaneously), and for constructing an effective validation metrics for monitoring (e.g. for early stopping and lr scheduling). Entries without ``coeff`` will still have their metrics computed, but they will not be incorporated into the ``weighted_sum``. An example for the utility of this feature is for monitoring additional metrics of the same nature (e.g. energy MSE and MAE) but only wanting one of them be in the effective metric used for lr scheduling. Note that these coefficients will be **automatically normalized** to sum up to one, e.g. if one has an energy metric with ``coeff=3`` and a force metric with ``coeff=1``, the ``weighted_sum`` is computed with coefficients ``[0.75, 0.25]`` for the energy and force metric respectively.

      - ``ignore_nan`` is a ``bool`` (defaults to ``False`` if not provided). This should be set to true if one expects the underlying ``target`` data to contain ``NaN`` entries. An example use case is when one has a dataset with ``stress`` labels for only a portion of the dataset. One can still train on ``stress`` for data that contain it and the others can be set as ``NaN`` entries to be handled appropriately during metric computation with this key.

      - ``name`` is the name that the metric is logged as. Default names are used if not provided, but it is recommended for users to set custom names for clarity and control.

    Args:
        metrics (list): list of dictionaries with keys ``field``, ``metric``, ``per_type``, ``coeff``, ``ignore_nan``, and ``name``
        type_names (list): required for ``per_type`` metrics (if this class is instantiated in ``nequip.train.NequIPLightningModule``, which is the case if one uses ``nequip-train``, this is automatically handled such that users need not explicitly fill in this field in the config)
    """

    def __init__(
        self,
        metrics: List[
            Dict[str, Union[float, str, Dict[str, Union[str, Callable]], Metric]]
        ],
        type_names: List[str] = None,
    ):
        super().__init__()

        # === sanity checks ===
        for metric_dict in metrics:
            for key in metric_dict.keys():
                assert (
                    key in _METRICS_MANAGER_INPUT_KEYS
                ), f"unrecognized key `{key}` found as input in `MetricsManager`"

        self.num_metrics = len(metrics)
        # === MANDATORY dict keys ===
        self.fields = [
            (
                BaseModifier(metric["field"])
                if isinstance(metric["field"], str)
                else metric["field"]
            )
            for metric in metrics
        ]
        for metric in metrics:
            self.append(metric["metric"])

        # === OPTIONAL dict keys and logic based on dict.get(key, None) ===
        # == ignore Nan ==
        self.ignore_nans = [metric.get("ignore_nan", False) for metric in metrics]
        assert all(isinstance(item, bool) for item in self.ignore_nans)

        # == process names ==
        self.names = []
        for idx in range(self.num_metrics):
            name = metrics[idx].get("name", None)
            if name is None:
                name = "_".join([str(self.fields[idx]), str(self[idx])])
            assert name != "weighted_sum"  # special name check, just in case
            self.names.append(name)
        assert len(self.names) == len(
            set(self.names)
        ), f"Repeated names found ({self.names}) -- names must be unique. It is recommended to give custom names instead of relying on the automatic naming."

        # === per_type metrics ===
        self.per_type = [metric.get("per_type", False) for metric in metrics]
        if any(self.per_type):
            assert (
                type_names is not None
            ), "`type_names` must be provided if any `per_type=True`"
        self.type_names = type_names

        for idx in range(self.num_metrics):
            if self.per_type[idx]:
                field_type = self.fields[idx].type
                # TODO: potentially implement per_type edge metrics
                if field_type != "node":
                    raise RuntimeError(
                        f"`per_type` metrics only supported for node fields, but {field_type} field found for {self.names[idx]}."
                    )
                # set up per_type metrics as a ModuleList
                # one copy of the base Metric for each type in forward() and compute()
                ptm_list = torch.nn.ModuleList([])
                if field_type == "node":
                    num_types = len(self.type_names)
                elif field_type == "edge":
                    num_types = len(self.type_names) * len(self.type_names)
                for _ in range(num_types):
                    ptm_list.append(self[idx].clone())
                self[idx] = ptm_list

        # == process coeffs ==
        self.coeffs = [metric.get("coeff", None) for metric in metrics]
        self.do_weighted_sum = False
        self.set_coeffs(self.coeffs)  # normalize coefficients if not all None

        # convenient to cache metrics computed last for callbacks
        self.metrics_values_step = []
        self.metrics_values_epoch = []

    def forward(
        self,
        preds: AtomicDataDict.Type,
        target: AtomicDataDict.Type,
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Computes and accumulates metrics (intended for use at batch steps).
        """
        self.metrics_values_step = []
        if self.do_weighted_sum:
            weighted_sum = 0.0
        metric_dict = {}
        for idx in range(self.num_metrics):
            preds_field, target_field = self.fields[idx](preds, target)

            if self.per_type[idx]:
                metric = 0
                num_contributing_types = 0
                for type_idx, type_name in enumerate(self.type_names):
                    # index out each type
                    selector = torch.eq(preds[AtomicDataDict.ATOM_TYPE_KEY], type_idx)
                    per_type_preds = preds_field[selector]
                    per_type_target = target_field[selector]

                    # mask out NaNs (based on target)
                    if self.ignore_nans[idx]:
                        notnan_mask = ~torch.isnan(per_type_target)
                        per_type_preds = torch.masked_select(
                            per_type_preds, notnan_mask
                        )
                        per_type_target = torch.masked_select(
                            per_type_target, notnan_mask
                        )

                    pt_metric = self[idx][type_idx](per_type_preds, per_type_target)
                    pt_metric_name = (
                        prefix + "_".join([self.names[idx], type_name]) + suffix
                    )
                    metric_dict.update({pt_metric_name: pt_metric})
                    # account for batches without atom type
                    assert pt_metric.numel() == 1
                    if not torch.isnan(pt_metric):
                        metric = metric + pt_metric
                        num_contributing_types += 1
                assert num_contributing_types <= len(self.type_names)
                metric = metric / num_contributing_types
            else:
                # mask out NaNs (based on target)
                if self.ignore_nans[idx]:
                    notnan_mask = ~torch.isnan(target_field)
                    preds_field = torch.masked_select(preds_field, notnan_mask)
                    target_field = torch.masked_select(target_field, notnan_mask)
                metric = self[idx](preds_field, target_field)

            metric_dict.update({prefix + self.names[idx] + suffix: metric})
            self.metrics_values_step.append(metric.item())

            if self.do_weighted_sum:
                if self.coeffs[idx] is not None:
                    weighted_sum = weighted_sum + metric * self.coeffs[idx]
        if self.do_weighted_sum:
            metric_dict.update({prefix + "weighted_sum" + suffix: weighted_sum})
        return metric_dict

    def compute(self, prefix: str = "", suffix: str = ""):
        """
        Aggregates accumulated metrics (intended for use at the end of an epoch).
        """
        self.metrics_values_epoch = []
        if self.do_weighted_sum:
            weighted_sum = 0.0
        metric_dict = {}
        for idx in range(self.num_metrics):
            if self.per_type[idx]:
                metric = 0
                for type_idx, type_name in enumerate(self.type_names):
                    ps_metric = self[idx][type_idx].compute()
                    ps_metric_name = (
                        prefix + "_".join([self.names[idx], type_name]) + suffix
                    )
                    metric_dict.update({ps_metric_name: ps_metric})
                    metric = metric + ps_metric
                metric = metric / len(self.type_names)
            else:
                metric = self[idx].compute()
            metric_dict.update({prefix + self.names[idx] + suffix: metric})
            self.metrics_values_epoch.append(metric.item())
            if self.do_weighted_sum:
                if self.coeffs[idx] is not None:
                    weighted_sum = weighted_sum + metric * self.coeffs[idx]
        if self.do_weighted_sum:
            metric_dict.update({prefix + "weighted_sum" + suffix: weighted_sum})

        return metric_dict

    def reset(self):
        for idx in range(self.num_metrics):
            if self.per_type[idx]:
                for type_idx in range(len(self.type_names)):
                    self[idx][type_idx].reset()
            else:
                self[idx].reset()

    def set_coeffs(self, coeffs: List[float]) -> None:
        """
        Sanity checks and normalizes coefficients to one before setting the new coefficients.
        """
        assert len(coeffs) == len(self.coeffs)
        if not all([coeff is None for coeff in coeffs]):
            # normalize coefficients to sum up to 1 wherever provided
            tot = sum([c if c is not None else 0.0 for c in coeffs])
            coeffs = [c / tot if c is not None else None for c in coeffs]
            self.do_weighted_sum = True
        self.coeffs = coeffs

    def get_extra_state(self) -> None:
        """"""
        return {
            "coeffs": self.coeffs,
            "metrics_values_step": self.metrics_values_step,
            "metrics_values_epoch": self.metrics_values_epoch,
        }

    def set_extra_state(self, state: Dict) -> None:
        """"""
        self.set_coeffs(state["coeffs"])
        self.metrics_values_step = state["metrics_values_step"]
        self.metrics_values_epoch = state["metrics_values_epoch"]
