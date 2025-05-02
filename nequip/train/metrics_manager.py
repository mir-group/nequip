# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from torchmetrics import Metric
from nequip.data import AtomicDataDict, BaseModifier, PerAtomModifier
from .metrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    RootMeanSquaredError,
)

from typing import List, Dict, Union, Callable, Final, Optional

_METRICS_MANAGER_INPUT_KEYS: Final[List[str]] = [
    "field",
    "metric",
    "name",
    "coeff",
    "per_type",
    "ignore_nan",
]

_METRICS_MANAGER_MANDATORY_INPUT_KEYS: Final[List[str]] = [
    "metric",
]


class MetricsManager(torch.nn.ModuleDict):
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
        self.metrics: Dict[str, Dict[str, Union[float, str, bool]]] = {}
        per_type_encountered = False
        for metric_dict in metrics:
            # === sanity checks ===
            # == check that keys present are expected ==
            for key in metric_dict.keys():
                assert (
                    key in _METRICS_MANAGER_INPUT_KEYS
                ), f"unrecognized key `{key}` found as input in `MetricsManager`"

            # == check that mandatory keys are present ==
            for mandatory_key in _METRICS_MANAGER_MANDATORY_INPUT_KEYS:
                assert (
                    mandatory_key in metric_dict.keys()
                ), f"Each dictionary in `MetricManager`'s `metrics` argument must possess at least the following mandatory keys: {_METRICS_MANAGER_MANDATORY_INPUT_KEYS}"

            # === field ===
            # field can be
            # - str: we will wrap it with `BaseModifier`
            # - Callable: it should be a subclass of `BaseModifier`
            # - None: this is a special metric that will have a different code path than the rest
            field = metric_dict.get("field", None)
            field = BaseModifier(field) if isinstance(field, str) else field
            if field is not None:
                assert isinstance(field, BaseModifier)

            # === metric ===
            metric = metric_dict["metric"]

            # === names ===
            name = metric_dict.get("name", None)
            if name is None:
                name = (
                    "_".join([str(field), str(metric)])
                    if field is not None
                    else str(metric)
                )
            assert (
                name != "weighted_sum"
            ), "`weighted_sum` is a specially reserved metric name that should not be configured."
            assert (
                name not in self.metrics.keys()
            ), f"Repeated names found ({name}) -- names must be unique. It is recommended to give custom names instead of relying on the automatic naming."

            # === sanity check `field=None` case ===
            if field is None:
                for key in ("ignore_nan", "per_type"):
                    assert (
                        key not in metric_dict
                    ), f"When field is not provided or `field: None`, `{key}` should not be provided."
                # NOTE that we still go through the `ignore_nan` and `per_type` handling below so they get the default values
                # so special metrics will have to handle NaNs on their own, and cannot be used with the automatic `per_type` system

            # == ignore Nan ==
            ignore_nan = metric_dict.get("ignore_nan", False)
            assert isinstance(
                ignore_nan, bool
            ), f"`ignore_nan` should be a bool, but found {ignore_nan} of type {type(ignore_nan)}"

            # == per_type metrics ==
            per_type = metric_dict.get("per_type", False)
            # sanity check that `type_names` is provided
            # to only do the assert once
            if per_type and not per_type_encountered:
                per_type_encountered = True
                assert (
                    type_names is not None
                ), "`type_names` must be provided if any `per_type=True`"
                self.type_names = type_names
            # logic to proliferate metrics objects
            if per_type:
                if field.type != "node":
                    raise RuntimeError(
                        f"`per_type` metrics only supported for node fields, but {field.type} field found for {name}."
                    )
                # set up per_type metrics as a ModuleList
                # one copy of the base Metric for each type in forward() and compute()
                ptm_list = torch.nn.ModuleList([])
                for _ in range(len(self.type_names)):
                    ptm_list.append(metric.clone())
                metric = ptm_list

            # === construct dict entry ===
            self.metrics.update(
                {
                    name: {
                        "field": field,
                        "coeff": metric_dict.get("coeff", None),
                        "ignore_nan": ignore_nan,
                        "per_type": per_type,
                    }
                }
            )
            self.update({name: metric})

            # == clean up for safety ==
            if field is not None:
                del ignore_nan, per_type
            del field, metric, name

        # === process coefficients ===
        self.do_weighted_sum = False
        self.set_coeffs(
            {k: v["coeff"] for k, v in self.metrics.items()}
        )  # normalize coefficients if not all None

        # convenient to cache metrics computed last for callbacks
        self.metrics_values_step = {k: None for k in self.metrics.keys()}
        self.metrics_values_epoch = {k: None for k in self.metrics.keys()}

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
        self.metrics_values_step = {k: None for k in self.metrics.keys()}
        if self.do_weighted_sum:
            weighted_sum = 0.0
        metric_dict = {}

        for metric_name, metric_params in self.metrics.items():
            field: Optional[Callable] = metric_params["field"]

            if field is not None:
                per_type: bool = metric_params["per_type"]
                ignore_nan: bool = metric_params["ignore_nan"]

                preds_field, target_field = field(preds, target)
                if per_type:
                    metric = 0
                    num_contributing_types = 0
                    for type_idx, type_name in enumerate(self.type_names):
                        # index out each type
                        selector = torch.eq(
                            preds[AtomicDataDict.ATOM_TYPE_KEY], type_idx
                        )
                        per_type_preds = preds_field[selector]
                        per_type_target = target_field[selector]

                        # mask out NaNs (based on target)
                        if ignore_nan:
                            notnan_mask = ~torch.isnan(per_type_target)
                            per_type_preds = torch.masked_select(
                                per_type_preds, notnan_mask
                            )
                            per_type_target = torch.masked_select(
                                per_type_target, notnan_mask
                            )

                        pt_metric = self[metric_name][type_idx](
                            per_type_preds, per_type_target
                        )
                        pt_metric_name = (
                            prefix + "_".join([metric_name, type_name]) + suffix
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
                    if ignore_nan:
                        notnan_mask = ~torch.isnan(target_field)
                        preds_field = torch.masked_select(preds_field, notnan_mask)
                        target_field = torch.masked_select(target_field, notnan_mask)
                    metric = self[metric_name](preds_field, target_field)
            else:
                # custom metric that takes in two AtomicDataDict objects
                metric = self[metric_name](preds, target)

            metric_dict.update({prefix + metric_name + suffix: metric})
            self.metrics_values_step.update({metric_name: metric.item()})

            if self.do_weighted_sum:
                coeff: Optional[float] = metric_params["coeff"]
                if coeff is not None:
                    weighted_sum = weighted_sum + metric * coeff

        if self.do_weighted_sum:
            metric_dict.update({prefix + "weighted_sum" + suffix: weighted_sum})

        return metric_dict

    def compute(self, prefix: str = "", suffix: str = ""):
        """
        Aggregates accumulated metrics (intended for use at the end of an epoch).
        """
        self.metrics_values_epoch = {k: None for k in self.metrics.keys()}
        if self.do_weighted_sum:
            weighted_sum = 0.0
        metric_dict = {}
        for metric_name, metric_params in self.metrics.items():
            if metric_params["per_type"]:
                metric = 0
                for type_idx, type_name in enumerate(self.type_names):
                    ps_metric = self[metric_name][type_idx].compute()
                    ps_metric_name = (
                        prefix + "_".join([metric_name, type_name]) + suffix
                    )
                    metric_dict.update({ps_metric_name: ps_metric})
                    metric = metric + ps_metric
                metric = metric / len(self.type_names)
            else:
                metric = self[metric_name].compute()
            metric_dict.update({prefix + metric_name + suffix: metric})
            self.metrics_values_epoch.update({metric_name: metric.item()})
            if self.do_weighted_sum:
                coeff = metric_params["coeff"]
                if coeff is not None:
                    weighted_sum = weighted_sum + metric * coeff
        if self.do_weighted_sum:
            metric_dict.update({prefix + "weighted_sum" + suffix: weighted_sum})

        return metric_dict

    def reset(self):
        for metric_name, metric_params in self.metrics.items():
            if metric_params["per_type"]:
                for type_idx in range(len(self.type_names)):
                    self[metric_name][type_idx].reset()
            else:
                self[metric_name].reset()

    def set_coeffs(self, coeff_dict: Dict[str, Optional[float]]) -> None:
        """
        Sanity checks and normalizes coefficients to one before setting the new coefficients.
        If some metrics are unspecified, the ``coeff`` will be assumed to be ``None``.
        """
        # add `None` if metric coeff is unspecified
        for metric_name in self.metrics.keys():
            if metric_name not in coeff_dict.keys():
                coeff_dict[metric_name] = None
        # normalize the coeffs
        if not all([v is None for k, v in coeff_dict.items()]):
            # normalize coefficients to sum up to 1 wherever provided
            tot = sum([v if v is not None else 0.0 for k, v in coeff_dict.items()])
            for name, metric_dict in self.metrics.items():
                metric_dict["coeff"] = (
                    coeff_dict[name] / tot if coeff_dict[name] is not None else None
                )
            self.do_weighted_sum = True
        else:
            for name, metric_dict in self.metrics.items():
                metric_dict["coeff"] = None
            self.do_weighted_sum = False

    def get_extra_state(self) -> None:
        """"""
        return {
            "coeff_dict": {k: v["coeff"] for k, v in self.metrics.items()},
            "metrics_values_step": self.metrics_values_step,
            "metrics_values_epoch": self.metrics_values_epoch,
        }

    def set_extra_state(self, state: Dict) -> None:
        """"""
        self.set_coeffs(state["coeff_dict"])
        self.metrics_values_step = state["metrics_values_step"]
        self.metrics_values_epoch = state["metrics_values_epoch"]


def EnergyForceLoss(
    coeffs: Dict[str, float] = {
        AtomicDataDict.TOTAL_ENERGY_KEY: 1.0,
        AtomicDataDict.FORCE_KEY: 1.0,
    },
    per_atom_energy: bool = True,
    type_names=None,
):
    """Simplified ``MetricsManager`` wrapper for a **loss** term containing energy and forces mean squared errors (MSEs).

    The loss component names are ``per_atom_energy_mse`` OR `total_energy_mse` (depending on whether ``per_atom_energy`` is ``True`` or ``False``), and `forces_mse`, which are the names to refer to when neeeded, e.g. when scheduling loss component coefficients.

    Example usage in config:
    ::

        training_module:
          _target_: nequip.train.NequIPLightningModule

          loss:
            _target_: nequip.train.EnergyForceLoss
            per_atom_energy: true
            coeffs:
              total_energy: 1.0
              forces: 1.0

    Args:
        coeffs (Dict[str, float]): ``dict`` that stores the relative weight of energy and forces to the overall loss (default ``{'total_energy': 1.0, 'forces': 1.0}``)
        per_atom_energy (bool, optional): whether to normalize the total energy by the number of atoms (default ``True``)
    """

    metrics = [
        {
            "name": "per_atom_energy_mse" if per_atom_energy else "total_energy_mse",
            "field": (
                PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY)
                if per_atom_energy
                else AtomicDataDict.TOTAL_ENERGY_KEY
            ),
            "coeff": coeffs[AtomicDataDict.TOTAL_ENERGY_KEY],
            "metric": MeanSquaredError(),
        },
        {
            "name": "forces_mse",
            "field": AtomicDataDict.FORCE_KEY,
            "coeff": coeffs[AtomicDataDict.FORCE_KEY],
            "metric": MeanSquaredError(),
        },
    ]
    return MetricsManager(metrics, type_names=type_names)


_EF_METRICS_COEFFS_KEYS: Final[List[str]] = [
    "total_energy_rmse",
    "per_atom_energy_rmse",
    "forces_rmse",
    "total_energy_mae",
    "per_atom_energy_mae",
    "forces_mae",
]

_EFS_METRICS_COEFFS_KEYS: Final[List[str]] = _EF_METRICS_COEFFS_KEYS + [
    "stress_rmse",
    "stress_mae",
]


def EnergyForceMetrics(
    coeffs: Dict[str, float] = {
        "total_energy_rmse": 1.0,
        "per_atom_energy_rmse": None,
        "forces_rmse": 1.0,
        "total_energy_mae": None,
        "per_atom_energy_mae": None,
        "forces_mae": None,
    },
    type_names=None,
):
    """Simplified ``MetricsManager`` wrapper for a **metric** term containing energy and force mean absolute errors (MAEs) and root mean squared errors (RMSEs).

    Example usage in config:
    ::

        training_module:
          _target_: nequip.train.NequIPLightningModule

          val_metrics:
            _target_: nequip.train.EnergyForceMetrics
            coeffs:
              total_energy_rmse: 1.0
              per_atom_energy_rmse: null
              forces_rmse: 1.0
              total_energy_mae: null
              per_atom_energy_mae: null
              forces_mae: null

    Args:
        coeffs (Dict[str, float]): ``dict`` that stores the relative contribution of the different energy and forces metrics to the ``weighted_sum`` version of the metric as in ``nequip.train.MetricsManager`` (default ``{'total_energy_rmse': 1.0, 'per_atom_energy_rmse': None, 'forces_rmse': 1.0, 'total_energy_mae': None, 'per_atom_energy_mae': None, 'forces_mae': None}``)
    """
    assert all(
        [k in _EF_METRICS_COEFFS_KEYS for k in coeffs.keys()]
    ), f"Unrecognized key found in `coeffs`, only the following are recognized: {_EF_METRICS_COEFFS_KEYS}"
    metrics = [
        {
            "name": "total_energy_rmse",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("total_energy_rmse", None),
        },
        {
            "name": "total_energy_mae",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("total_energy_mae", None),
        },
        {
            "name": "per_atom_energy_rmse",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("per_atom_energy_rmse", None),
        },
        {
            "name": "per_atom_energy_mae",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("per_atom_energy_mae", None),
        },
        {
            "name": "forces_rmse",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("forces_rmse", None),
        },
        {
            "name": "forces_mae",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("forces_mae", None),
        },
    ]
    return MetricsManager(metrics, type_names=type_names)


def EnergyForceStressLoss(
    coeffs: Dict[str, float] = {
        AtomicDataDict.TOTAL_ENERGY_KEY: 1.0,
        AtomicDataDict.FORCE_KEY: 1.0,
        AtomicDataDict.STRESS_KEY: 1.0,
    },
    per_atom_energy: bool = True,
    type_names=None,
):
    """Simplified ``MetricsManager`` wrapper for a **loss** term containing energy, forces and stress mean squared errors (MSEs).

    The loss component names are ``per_atom_energy_mse`` OR ``total_energy_mse`` (depending on whether ``per_atom_energy`` is ``True`` or ``False``), ``forces_mse``, and ``stress_mse``, which are the names to refer to when neeeded, e.g. when scheduling loss component coefficients.

    Example usage in config:
    ::

        training_module:
          _target_: nequip.train.NequIPLightningModule

          loss:
            _target_: nequip.train.EnergyForceStressLoss
            per_atom_energy: true
            coeffs:
              total_energy: 1.0
              forces: 1.0
              stress: 1.0

    Args:
        coeffs (Dict[str, float]): ``dict`` that stores the relative weight of energy and forces to the overall loss (default ``{'total_energy': 1.0, 'forces': 1.0, 'stress': 1.0}``)
        per_atom_energy (bool, optional): whether to normalize the total energy by the number of atoms (default ``True``)
    """

    metrics = [
        {
            "name": "per_atom_energy_mse" if per_atom_energy else "total_energy_mse",
            "field": (
                PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY)
                if per_atom_energy
                else AtomicDataDict.TOTAL_ENERGY_KEY
            ),
            "coeff": coeffs[AtomicDataDict.TOTAL_ENERGY_KEY],
            "metric": MeanSquaredError(),
        },
        {
            "name": "forces_mse",
            "field": AtomicDataDict.FORCE_KEY,
            "coeff": coeffs[AtomicDataDict.FORCE_KEY],
            "metric": MeanSquaredError(),
        },
        {
            "name": "stress_mse",
            "field": AtomicDataDict.STRESS_KEY,
            "coeff": coeffs[AtomicDataDict.STRESS_KEY],
            "metric": MeanSquaredError(),
        },
    ]
    return MetricsManager(metrics, type_names=type_names)


def EnergyForceStressMetrics(
    coeffs: Dict[str, float] = {
        "total_energy_rmse": 1.0,
        "per_atom_energy_rmse": None,
        "forces_rmse": 1.0,
        "stress_rmse": 1.0,
        "total_energy_mae": None,
        "per_atom_energy_mae": None,
        "forces_mae": None,
        "stress_mae": None,
    },
    type_names=None,
):
    """Simplified ``MetricsManager`` wrapper for a **metric** term containing energy, force and stress mean absolute errors (MAEs) and root mean squared errors (RMSEs).

    Example usage in config:
    ::

        training_module:
          _target_: nequip.train.NequIPLightningModule

          val_metrics:
            _target_: nequip.train.EnergyForceStressMetrics
            coeffs:
              total_energy_rmse: 1.0
              per_atom_energy_rmse: null
              forces_rmse: 1.0
              stress_rmse: 1.0
              total_energy_mae: null
              per_atom_energy_mae: null
              forces_mae: null
              stress_mae: null

    Args:
        coeffs (Dict[str, float]): ``dict`` that stores the relative contribution of the different energy and forces metrics to the ``weighted_sum`` version of the metric as in ``nequip.train.MetricsManager`` (default ``{'total_energy_rmse': 1.0, 'per_atom_energy_rmse': None, 'forces_rmse': 1.0, 'stress_rmse': 1.0, 'total_energy_mae': None, 'per_atom_energy_mae': None, 'forces_mae': None, 'stress_mae': None}``)
    """
    assert all(
        [k in _EFS_METRICS_COEFFS_KEYS for k in coeffs.keys()]
    ), f"Unrecognized key found in `coeffs`, only the following are recognized: {_EFS_METRICS_COEFFS_KEYS}"
    metrics = [
        {
            "name": "total_energy_rmse",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("total_energy_rmse", None),
        },
        {
            "name": "total_energy_mae",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("total_energy_mae", None),
        },
        {
            "name": "per_atom_energy_rmse",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("per_atom_energy_rmse", None),
        },
        {
            "name": "per_atom_energy_mae",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("per_atom_energy_mae", None),
        },
        {
            "name": "forces_rmse",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("forces_rmse", None),
        },
        {
            "name": "forces_mae",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("forces_mae", None),
        },
        {
            "name": "stress_rmse",
            "field": AtomicDataDict.STRESS_KEY,
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("stress_rmse", None),
        },
        {
            "name": "stress_mae",
            "field": AtomicDataDict.STRESS_KEY,
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("stress_mae", None),
        },
    ]
    return MetricsManager(metrics, type_names=type_names)
