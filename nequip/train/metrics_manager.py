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
    """Manages metrics for ``AtomicDataDict`` objects in loss functions and monitored error metrics.

    This class handles both loss computation and monitored error metrics through a unified interface.
    Each metric is defined by a dictionary with mandatory and optional keys.

    The ``metrics`` parameter is a list of dictionaries defining individual metrics.
    Each dictionary can contain keys: ``field``, ``metric``, ``coeff``, ``name``, ``per_type``, ``ignore_nan``.
    The ``type_names`` parameter is a list of atom type names, required if ANY metric uses ``per_type=True``.
    When this class is instantiated in :class:`~nequip.train.NequIPLightningModule`
    (which is the case when using ``nequip-train``), ``type_names`` is automatically
    provided from the dataset configuration and users need not specify it explicitly.

    Mandatory Keys
    --------------
    ``metric`` : ``torchmetrics.Metric``
        The metric class to compute (e.g., :class:`~nequip.train.MeanSquaredError`,
        :class:`~nequip.train.MeanAbsoluteError`, :class:`~nequip.train.RootMeanSquaredError`).

    Optional Keys
    -------------
    ``field`` : ``str`` or ``Callable``, optional
        Data field to extract for metric computation.
        Can be:

        - ``str``: A nequip field name (e.g., ``'total_energy'``, ``'forces'``, ``'stress'``)
        - ``Callable``: A field modifier like :class:`~nequip.data.PerAtomModifier`
        - ``None``: For custom metrics that need full :class:`AtomicDataDict` objects

        When ``None``, the metric receives ``(preds, target)`` as complete data dictionaries
        instead of extracted field values.
        The metric must handle all data processing, type filtering, and NaN handling internally.

    ``coeff`` : ``float``, optional
        Weight for ``weighted_sum`` calculation.
        If any metric has a coefficient, ``weighted_sum`` is computed as the weighted average of all metrics with coefficients.
        Coefficients are automatically normalized to sum to 1.

    ``name`` : ``str``, optional
        Custom name for logging.
        Auto-generated from field and metric if not provided.
        Must be unique across all metrics.

    ``per_type`` : ``bool``, default=False
        Compute separate metrics for each atom type, then average them.
        Only valid for node fields (like ``forces``).
        Requires ``type_names`` parameter.

    ``ignore_nan`` : ``bool``, default=False
        Handle NaN values in target data by masking them out.
        Useful for datasets with partial labels (e.g., stress data available only for some structures).

    Key Behaviors
    -------------
    **Coefficient Normalization**: Coefficients are automatically normalized to sum to 1.
    For example, ``{energy: 3.0, forces: 1.0}`` becomes ``{energy: 0.75, forces: 0.25}``.

    **Per-Type Requirements**: If ANY metric uses ``per_type=True``, the ``type_names``
    parameter is mandatory for the entire MetricsManager instance.

    **Per-Type Averaging**: During batch steps, per-type metrics may be NaN if that atom
    type is absent from the batch.
    The effective metric averages only over types present in that batch.
    During epoch computation, all configured types contribute to the average.

    **Special Field=None Mode**: When ``field=None``, the metric receives the complete
    ``AtomicDataDict`` objects ``(preds, target)`` directly.
    This enables custom metrics that need access to multiple fields or geometric information.
    However, ``per_type`` and ``ignore_nan`` features are disabled—the custom metric must handle type filtering
    and NaN processing itself if needed.

    **Weighted Sum**: A ``weighted_sum`` metric is automatically computed when any metric
    has a coefficient.
    This serves as the effective loss (for training) or monitoring metric (for validation).
    Metrics without coefficients are computed but excluded from the weighted sum.

    E.g., custom ``MetricsManager`` equivalent to ``EnergyForceLoss``:

    .. code-block:: yaml

        _target_: nequip.train.MetricsManager
        metrics:
            - name: per_atom_energy_mse
            field:
                _target_: nequip.data.PerAtomModifier
                field: total_energy
            coeff: 1
            metric:
                _target_: nequip.train.MeanSquaredError
            - name: forces_mse
            field: forces
            coeff: 1
            metric:
                _target_: nequip.train.MeanSquaredError
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
                assert key in _METRICS_MANAGER_INPUT_KEYS, (
                    f"unrecognized key `{key}` found as input in `MetricsManager`"
                )

            # == check that mandatory keys are present ==
            for mandatory_key in _METRICS_MANAGER_MANDATORY_INPUT_KEYS:
                assert mandatory_key in metric_dict.keys(), (
                    f"Each dictionary in `MetricManager`'s `metrics` argument must possess at least the following mandatory keys: {_METRICS_MANAGER_MANDATORY_INPUT_KEYS}"
                )

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
            assert name != "weighted_sum", (
                "`weighted_sum` is a specially reserved metric name that should not be configured."
            )
            assert name not in self.metrics.keys(), (
                f"Repeated names found ({name}) -- names must be unique. It is recommended to give custom names instead of relying on the automatic naming."
            )

            # === sanity check `field=None` case ===
            if field is None:
                for key in ("ignore_nan", "per_type"):
                    assert key not in metric_dict, (
                        f"When field is not provided or `field: None`, `{key}` should not be provided."
                    )
                # NOTE that we still go through the `ignore_nan` and `per_type` handling below so they get the default values
                # so special metrics will have to handle NaNs on their own, and cannot be used with the automatic `per_type` system

            # == ignore Nan ==
            ignore_nan = metric_dict.get("ignore_nan", False)
            assert isinstance(ignore_nan, bool), (
                f"`ignore_nan` should be a bool, but found {ignore_nan} of type {type(ignore_nan)}"
            )

            # == per_type metrics ==
            per_type = metric_dict.get("per_type", False)
            # sanity check that `type_names` is provided
            # to only do the assert once
            if per_type and not per_type_encountered:
                per_type_encountered = True
                assert type_names is not None, (
                    "`type_names` must be provided if any `per_type=True`"
                )
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
    """Simplified :class:`MetricsManager` wrapper for a **loss** term containing energy and forces mean squared errors (MSEs).

    The loss component names are ``per_atom_energy_mse`` OR ``total_energy_mse`` (depending on whether ``per_atom_energy`` is ``True`` or ``False``), and ``forces_mse``, which are the names to refer to when neeeded, e.g. when scheduling loss component coefficients.

    Example usage in config:

    .. code-block:: yaml

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
    """Simplified :class:`MetricsManager` wrapper for a **metric** term containing energy and force mean absolute errors (MAEs) and root mean squared errors (RMSEs).

    Example usage in config:

    .. code-block:: yaml

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
    assert all([k in _EF_METRICS_COEFFS_KEYS for k in coeffs.keys()]), (
        f"Unrecognized key found in `coeffs`, only the following are recognized: {_EF_METRICS_COEFFS_KEYS}"
    )
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
    """Simplified :class:`MetricsManager` wrapper for a **loss** term containing energy, forces and stress mean squared errors (MSEs).

    The loss component names are ``per_atom_energy_mse`` OR ``total_energy_mse`` (depending on whether ``per_atom_energy`` is ``True`` or ``False``), ``forces_mse``, and ``stress_mse``, which are the names to refer to when neeeded, e.g. when scheduling loss component coefficients.

    Example usage in config:

    .. code-block:: yaml

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
    """Simplified :class:`MetricsManager` wrapper for a **metric** term containing energy, force and stress mean absolute errors (MAEs) and root mean squared errors (RMSEs).

    Example usage in config:

    .. code-block:: yaml

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
    assert all([k in _EFS_METRICS_COEFFS_KEYS for k in coeffs.keys()]), (
        f"Unrecognized key found in `coeffs`, only the following are recognized: {_EFS_METRICS_COEFFS_KEYS}"
    )
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


def EnergyOnlyLoss(
    per_atom_energy: bool = True,
    type_names=None,
):
    """Simplified :class:`MetricsManager` wrapper for a **loss** term containing only energy mean squared error (MSE).

    The loss component name is ``per_atom_energy_mse`` OR ``total_energy_mse`` (depending on whether ``per_atom_energy`` is ``True`` or ``False``).

    Example usage in config:

    .. code-block:: yaml

        training_module:
          _target_: nequip.train.NequIPLightningModule

          loss:
            _target_: nequip.train.EnergyOnlyLoss
            per_atom_energy: true

    Args:
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
            "coeff": 1.0,  # single term, coefficient is always 1
            "metric": MeanSquaredError(),
        },
    ]
    return MetricsManager(metrics, type_names=type_names)


_ENERGY_ONLY_METRICS_COEFFS_KEYS: Final[List[str]] = [
    "total_energy_rmse",
    "per_atom_energy_rmse",
    "total_energy_mae",
    "per_atom_energy_mae",
]


def EnergyOnlyMetrics(
    coeffs: Dict[str, float] = {
        "total_energy_rmse": 1.0,
        "per_atom_energy_rmse": None,
        "total_energy_mae": None,
        "per_atom_energy_mae": None,
    },
    type_names=None,
):
    """Simplified :class:`MetricsManager` wrapper for a **metric** term containing only energy mean absolute errors (MAEs) and root mean squared errors (RMSEs).

    Example usage in config:

    .. code-block:: yaml

        training_module:
          _target_: nequip.train.NequIPLightningModule

          val_metrics:
            _target_: nequip.train.EnergyOnlyMetrics
            coeffs:
              total_energy_rmse: 1.0
              per_atom_energy_rmse: null
              total_energy_mae: null
              per_atom_energy_mae: null

    Args:
        coeffs (Dict[str, float]): ``dict`` that stores the relative contribution of the different energy metrics to the ``weighted_sum`` version of the metric as in ``nequip.train.MetricsManager`` (default ``{'total_energy_rmse': 1.0, 'per_atom_energy_rmse': None, 'total_energy_mae': None, 'per_atom_energy_mae': None}``)
    """
    assert all([k in _ENERGY_ONLY_METRICS_COEFFS_KEYS for k in coeffs.keys()]), (
        f"Unrecognized key found in `coeffs`, only the following are recognized: {_ENERGY_ONLY_METRICS_COEFFS_KEYS}"
    )
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
    ]
    return MetricsManager(metrics, type_names=type_names)
