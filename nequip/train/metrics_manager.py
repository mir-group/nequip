# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import dataclasses
import torch
from torchmetrics import Metric
from nequip.data import AtomicDataDict, BaseModifier, PerAtomModifier
from .metrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    RootMeanSquaredError,
    MaximumAbsoluteError,
)

from collections.abc import Mapping
from typing import Any, List, Dict, Union, Callable, Final, Optional


@dataclasses.dataclass
class MetricEntry:
    field: Optional[BaseModifier]
    coeff: Optional[float]
    ignore_nan: bool
    per_type: bool
    per_type_coeffs: Optional[List[float]]


# valid input keys for a metric dict: MetricEntry fields plus 'metric' and 'name'
_METRICS_MANAGER_INPUT_KEYS: Final[frozenset] = frozenset(
    {f.name for f in dataclasses.fields(MetricEntry)} | {"metric", "name"}
)


def _with_extra_metrics(
    metrics: List[Dict[str, Any]], extra_metrics: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    if extra_metrics is None:
        return metrics
    return [*metrics, *extra_metrics]


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

    ``per_type_coeffs`` : ``Dict[str, float]``, optional
        Per-type coefficients for combining per-type metrics into a single scalar.
        Note: these are loss-aggregation coefficients
        Requires ``per_type=True``. Must contain a strictly positive coefficient
        for every type in ``type_names`` (no zeros, no negatives, no missing keys).
        The aggregate becomes the weighted mean
        ``sum(c_i * metric_i) / sum(c_i)`` instead of the equal-mean default;
        equal coefficients reproduce the default exactly.
        Example: ``{H: 5.0, O: 1.0, Cs: 0.5}`` emphasizes H force errors relative to O and Cs.

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
        type_names: Optional[List[str]] = None,
    ):
        super().__init__()

        # pre-scan: ensure type_names is provided if any metric uses per_type
        if any(m.get("per_type", False) for m in metrics):
            assert type_names is not None, (
                "`type_names` must be provided if any `per_type=True`"
            )
            self.type_names = type_names

        self.entries: Dict[str, MetricEntry] = {}
        for metric_dict in metrics:
            name, entry, metric_module = self.parse_entry(metric_dict, type_names)
            assert name not in self.entries, (
                f"Repeated names found ({name}) -- names must be unique. It is recommended to give custom names instead of relying on the automatic naming."
            )
            self.entries[name] = entry
            self.update({name: metric_module})

        # normalize coefficients if not all None
        self.do_weighted_sum = False
        self.set_coeffs({k: v.coeff for k, v in self.entries.items()})

        # convenient to cache metrics computed last for callbacks
        self.metrics_values_step = {k: None for k in self.entries}
        self.metrics_values_epoch = {k: None for k in self.entries}

    @staticmethod
    def parse_entry(
        metric_dict: Dict[str, Any],
        type_names: Optional[List[str]],
    ):
        """Validates and parses a single metric dict into a ``(name, MetricEntry, metric_module)`` tuple."""
        # validate keys
        for key in metric_dict:
            assert key in _METRICS_MANAGER_INPUT_KEYS, (
                f"unrecognized key `{key}` found as input in `MetricsManager`"
            )
        assert "metric" in metric_dict, (
            "each dictionary in `MetricsManager`'s `metrics` argument must contain a `metric` key"
        )

        # field can be:
        # - str: wrap with BaseModifier
        # - Callable: must be a subclass of BaseModifier
        # - None: special metric receiving full AtomicDataDict objects
        field = metric_dict.get("field", None)
        field = BaseModifier(field) if isinstance(field, str) else field
        if field is not None:
            assert isinstance(field, BaseModifier)

        metric = metric_dict["metric"]

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

        # when field=None, per_type and ignore_nan are not supported
        if field is None:
            for key in ("ignore_nan", "per_type"):
                assert key not in metric_dict, (
                    f"When field is not provided or `field: None`, `{key}` should not be provided."
                )

        ignore_nan = metric_dict.get("ignore_nan", False)
        assert isinstance(ignore_nan, bool), (
            f"`ignore_nan` should be a bool, but found {ignore_nan} of type {type(ignore_nan)}"
        )

        per_type = metric_dict.get("per_type", False)
        if per_type:
            assert type_names is not None, (
                "`type_names` must be provided if any `per_type=True`"
            )
            if field.type != "node":
                raise RuntimeError(
                    f"`per_type` metrics only supported for node fields, but {field.type} field found for {name}."
                )
            # one copy of the base Metric per type, for use in forward() and compute()
            metric = torch.nn.ModuleList([metric.clone() for _ in type_names])

        # parse per_type_coeffs: align user-provided {type_name: coeff} to type_names ordering
        raw_coeffs = metric_dict.get("per_type_coeffs", None)
        if raw_coeffs is not None:
            if not per_type:
                raise ValueError(
                    f"`per_type_coeffs` provided for `{name}` but `per_type` is not True; per-type coefficients require `per_type: true`."
                )
            if not isinstance(raw_coeffs, Mapping):
                raise TypeError(
                    f"`per_type_coeffs` must be a dict mapping type name to positive float, got {type(raw_coeffs).__name__}."
                )
            type_names_set = set(type_names)
            provided_set = set(raw_coeffs.keys())
            unknown = provided_set - type_names_set
            if unknown:
                raise ValueError(
                    f"`per_type_coeffs` for `{name}` contains type names {sorted(unknown)} not in `type_names` {type_names}."
                )
            missing = type_names_set - provided_set
            if missing:
                raise ValueError(
                    f"`per_type_coeffs` for `{name}` must specify a positive coefficient for every type in `type_names`; missing: {sorted(missing)}."
                )
            per_type_coeffs = []
            for tn in type_names:
                c = float(raw_coeffs[tn])
                if c <= 0:
                    raise ValueError(
                        f"`per_type_coeffs` entry for `{tn}` must be positive (got {c})."
                    )
                per_type_coeffs.append(c)
        else:
            per_type_coeffs = None

        entry = MetricEntry(
            field=field,
            coeff=metric_dict.get("coeff", None),
            ignore_nan=ignore_nan,
            per_type=per_type,
            per_type_coeffs=per_type_coeffs,
        )
        return name, entry, metric

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
        self.metrics_values_step = {k: None for k in self.entries}
        if self.do_weighted_sum:
            weighted_sum = 0.0
        metric_dict = {}

        for metric_name, entry in self.entries.items():
            field: Optional[Callable] = entry.field

            if field is not None:
                per_type: bool = entry.per_type
                ignore_nan: bool = entry.ignore_nan

                preds_field, target_field = field(preds, target)
                if per_type:
                    metric = 0
                    coeffs: Optional[List[float]] = entry.per_type_coeffs
                    num_contributing_types = 0
                    coeff_sum = 0.0
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
                        metric_dict[f"{prefix}{metric_name}_{type_name}{suffix}"] = (
                            pt_metric
                        )
                        # account for batches without atom type
                        assert pt_metric.numel() == 1
                        if not torch.isnan(pt_metric):
                            if coeffs is None:
                                metric = metric + pt_metric
                                num_contributing_types += 1
                            else:
                                c = coeffs[type_idx]
                                metric = metric + c * pt_metric
                                coeff_sum += c
                    if coeffs is None:
                        assert num_contributing_types <= len(self.type_names)
                        metric = metric / num_contributing_types
                    else:
                        # weighted mean over contributing types
                        metric = metric / coeff_sum
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

            metric_dict[f"{prefix}{metric_name}{suffix}"] = metric
            self.metrics_values_step[metric_name] = metric.item()

            if self.do_weighted_sum:
                coeff: Optional[float] = entry.coeff
                if coeff is not None:
                    weighted_sum = weighted_sum + metric * coeff

        if self.do_weighted_sum:
            metric_dict[f"{prefix}weighted_sum{suffix}"] = weighted_sum

        return metric_dict

    def compute(self, prefix: str = "", suffix: str = ""):
        """
        Aggregates accumulated metrics (intended for use at the end of an epoch).
        """
        self.metrics_values_epoch = {k: None for k in self.entries}
        if self.do_weighted_sum:
            weighted_sum = 0.0
        metric_dict = {}
        for metric_name, entry in self.entries.items():
            if entry.per_type:
                metric = 0
                coeffs: Optional[List[float]] = entry.per_type_coeffs
                for type_idx, type_name in enumerate(self.type_names):
                    ps_metric = self[metric_name][type_idx].compute()
                    metric_dict[f"{prefix}{metric_name}_{type_name}{suffix}"] = (
                        ps_metric
                    )
                    if coeffs is None:
                        metric = metric + ps_metric
                    else:
                        metric = metric + coeffs[type_idx] * ps_metric
                if coeffs is None:
                    metric = metric / len(self.type_names)
                else:
                    metric = metric / sum(coeffs)
            else:
                metric = self[metric_name].compute()
            metric_dict[f"{prefix}{metric_name}{suffix}"] = metric
            self.metrics_values_epoch[metric_name] = metric.item()
            if self.do_weighted_sum:
                coeff = entry.coeff
                if coeff is not None:
                    weighted_sum = weighted_sum + metric * coeff
        if self.do_weighted_sum:
            metric_dict[f"{prefix}weighted_sum{suffix}"] = weighted_sum

        return metric_dict

    def reset(self):
        for metric_name, entry in self.entries.items():
            if entry.per_type:
                for m in self[metric_name]:
                    m.reset()
            else:
                self[metric_name].reset()

    def set_coeffs(self, coeff_dict: Dict[str, Optional[float]]) -> None:
        """
        Sanity checks and normalizes coefficients to one before setting the new coefficients.
        If some metrics are unspecified, the ``coeff`` will be assumed to be ``None``.
        """
        # fill missing keys with None
        coeffs = {k: coeff_dict.get(k) for k in self.entries}
        tot = sum(v for v in coeffs.values() if v is not None)
        self.do_weighted_sum = tot > 0
        for name, entry in self.entries.items():
            c = coeffs[name]
            entry.coeff = (
                (c / tot) if (c is not None and self.do_weighted_sum) else None
            )

    def get_extra_state(self) -> Dict[str, Any]:
        """"""
        return {
            "coeff_dict": {k: v.coeff for k, v in self.entries.items()},
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
    per_type_forces_coeffs: Optional[Dict[str, float]] = None,
    type_names: Optional[List[str]] = None,
    extra_metrics: Optional[List[Dict[str, Any]]] = None,
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
            # Optional: per-species coefficients for the forces loss.
            per_type_forces_coeffs:
              H:  5.0
              O:  1.0
              Cs: 0.5

    Args:
        coeffs (Dict[str, float]): ``dict`` that stores the relative weight of energy and forces to the overall loss (default ``{'total_energy': 1.0, 'forces': 1.0}``)
        per_atom_energy (bool, optional): whether to normalize the total energy by the number of atoms (default ``True``)
        per_type_forces_coeffs (Dict[str, float], optional): if provided, the forces MSE becomes a per-type weighted mean. See the ``per_type_coeffs`` key on :class:`MetricsManager` for the dict format and validation rules. (default ``None``)
        extra_metrics (list of dict, optional): additional metric entries appended to the wrapper's defaults
    """

    forces_entry: Dict[str, Any] = {
        "name": "forces_mse",
        "field": AtomicDataDict.FORCE_KEY,
        "coeff": coeffs[AtomicDataDict.FORCE_KEY],
        "metric": MeanSquaredError(),
    }
    if per_type_forces_coeffs is not None:
        forces_entry["per_type"] = True
        forces_entry["per_type_coeffs"] = per_type_forces_coeffs

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
        forces_entry,
    ]
    return MetricsManager(
        _with_extra_metrics(metrics, extra_metrics), type_names=type_names
    )


_EF_METRICS_COEFFS_KEYS: Final[List[str]] = [
    "total_energy_rmse",
    "per_atom_energy_rmse",
    "forces_rmse",
    "total_energy_mae",
    "per_atom_energy_mae",
    "forces_mae",
    "total_energy_maxabserr",
    "per_atom_energy_maxabserr",
    "forces_maxabserr",
]

_EFS_METRICS_COEFFS_KEYS: Final[List[str]] = _EF_METRICS_COEFFS_KEYS + [
    "stress_rmse",
    "stress_mae",
    "stress_maxabserr",
]


def EnergyForceMetrics(
    coeffs: Dict[str, Optional[float]] = {
        "total_energy_rmse": 1.0,
        "per_atom_energy_rmse": None,
        "forces_rmse": 1.0,
        "total_energy_mae": None,
        "per_atom_energy_mae": None,
        "forces_mae": None,
        "total_energy_maxabserr": None,
        "per_atom_energy_maxabserr": None,
        "forces_maxabserr": None,
    },
    type_names: Optional[List[str]] = None,
    extra_metrics: Optional[List[Dict[str, Any]]] = None,
):
    """Simplified :class:`MetricsManager` wrapper for a **metric** term containing energy and force mean absolute errors (MAEs), root mean squared errors (RMSEs), and maximum absolute errors (MaxAbsErrs).

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
              total_energy_maxabserr: null
              per_atom_energy_maxabserr: null
              forces_maxabserr: null

    Args:
        coeffs (Dict[str, float]): ``dict`` that stores the relative contribution of the different energy and forces metrics to the ``weighted_sum`` version of the metric as in ``nequip.train.MetricsManager`` (default ``{'total_energy_rmse': 1.0, 'per_atom_energy_rmse': None, 'forces_rmse': 1.0, 'total_energy_mae': None, 'per_atom_energy_mae': None, 'forces_mae': None, 'total_energy_maxabserr': None, 'per_atom_energy_maxabserr': None, 'forces_maxabserr': None}``)
        extra_metrics (list of dict, optional): additional metric entries appended to the wrapper's defaults
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
        {
            "name": "total_energy_maxabserr",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": MaximumAbsoluteError(),
            "coeff": coeffs.get("total_energy_maxabserr", None),
        },
        {
            "name": "per_atom_energy_maxabserr",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": MaximumAbsoluteError(),
            "coeff": coeffs.get("per_atom_energy_maxabserr", None),
        },
        {
            "name": "forces_maxabserr",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": MaximumAbsoluteError(),
            "coeff": coeffs.get("forces_maxabserr", None),
        },
    ]
    return MetricsManager(
        _with_extra_metrics(metrics, extra_metrics), type_names=type_names
    )


def EnergyForceStressLoss(
    coeffs: Dict[str, float] = {
        AtomicDataDict.TOTAL_ENERGY_KEY: 1.0,
        AtomicDataDict.FORCE_KEY: 1.0,
        AtomicDataDict.STRESS_KEY: 1.0,
    },
    per_atom_energy: bool = True,
    per_type_forces_coeffs: Optional[Dict[str, float]] = None,
    type_names: Optional[List[str]] = None,
    ignore_nan: Optional[Dict[str, bool]] = None,
    extra_metrics: Optional[List[Dict[str, Any]]] = None,
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
            # Optional: per-species coefficients for the forces loss.
            per_type_forces_coeffs:
              H:  5.0
              O:  1.0
              Cs: 0.5
            # if not all frames have stresses, one can populate the stress labels with NaN and set ignore_nan here:
            # ignore_nan:
            #   stress: true

    Args:
        coeffs (Dict[str, float]): ``dict`` that stores the relative weight of energy and forces to the overall loss (default ``{'total_energy': 1.0, 'forces': 1.0, 'stress': 1.0}``)
        per_atom_energy (bool, optional): whether to normalize the total energy by the number of atoms (default ``True``)
        per_type_forces_coeffs (Dict[str, float], optional): if provided, the forces MSE becomes a per-type weighted mean. Applies to forces only; energy and stress are unaffected. See the ``per_type_coeffs`` key on :class:`MetricsManager` for the dict format and validation rules. (default ``None``)
        ignore_nan (Dict[str, bool], optional): ``dict`` that specifies whether to ignore NaN values for each field (default: all ``False``)
        extra_metrics (list of dict, optional): additional metric entries appended to the wrapper's defaults
    """

    ignore_nan = {} if ignore_nan is None else ignore_nan
    forces_entry: Dict[str, Any] = {
        "name": "forces_mse",
        "field": AtomicDataDict.FORCE_KEY,
        "coeff": coeffs[AtomicDataDict.FORCE_KEY],
        "metric": MeanSquaredError(),
        "ignore_nan": ignore_nan.get(AtomicDataDict.FORCE_KEY, False),
    }
    if per_type_forces_coeffs is not None:
        forces_entry["per_type"] = True
        forces_entry["per_type_coeffs"] = per_type_forces_coeffs

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
            "ignore_nan": ignore_nan.get(AtomicDataDict.TOTAL_ENERGY_KEY, False),
        },
        forces_entry,
        {
            "name": "stress_mse",
            "field": AtomicDataDict.STRESS_KEY,
            "coeff": coeffs[AtomicDataDict.STRESS_KEY],
            "metric": MeanSquaredError(),
            "ignore_nan": ignore_nan.get(AtomicDataDict.STRESS_KEY, False),
        },
    ]
    return MetricsManager(
        _with_extra_metrics(metrics, extra_metrics), type_names=type_names
    )


def EnergyForceStressMetrics(
    coeffs: Dict[str, Optional[float]] = {
        "total_energy_rmse": 1.0,
        "per_atom_energy_rmse": None,
        "forces_rmse": 1.0,
        "stress_rmse": 1.0,
        "total_energy_mae": None,
        "per_atom_energy_mae": None,
        "forces_mae": None,
        "stress_mae": None,
        "total_energy_maxabserr": None,
        "per_atom_energy_maxabserr": None,
        "forces_maxabserr": None,
        "stress_maxabserr": None,
    },
    type_names: Optional[List[str]] = None,
    ignore_nan: Optional[Dict[str, bool]] = None,
    extra_metrics: Optional[List[Dict[str, Any]]] = None,
):
    """Simplified :class:`MetricsManager` wrapper for a **metric** term containing energy, force and stress mean absolute errors (MAEs), root mean squared errors (RMSEs), and maximum absolute errors (MaxAbsErrs).

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
              total_energy_maxabserr: null
              per_atom_energy_maxabserr: null
              forces_maxabserr: null
              stress_maxabserr: null
            # if not all frames have stresses, one can populate the stress labels with NaN and set ignore_nan here:
            # ignore_nan:
            #   stress: true

    Args:
        coeffs (Dict[str, float]): ``dict`` that stores the relative contribution of the different energy and forces metrics to the ``weighted_sum`` version of the metric as in ``nequip.train.MetricsManager`` (default ``{'total_energy_rmse': 1.0, 'per_atom_energy_rmse': None, 'forces_rmse': 1.0, 'stress_rmse': 1.0, 'total_energy_mae': None, 'per_atom_energy_mae': None, 'forces_mae': None, 'stress_mae': None, 'total_energy_maxabserr': None, 'per_atom_energy_maxabserr': None, 'forces_maxabserr': None, 'stress_maxabserr': None}``)
        ignore_nan (Dict[str, bool], optional): ``dict`` that specifies whether to ignore NaN values for each field (default: all ``False``)
        extra_metrics (list of dict, optional): additional metric entries appended to the wrapper's defaults
    """
    assert all([k in _EFS_METRICS_COEFFS_KEYS for k in coeffs.keys()]), (
        f"Unrecognized key found in `coeffs`, only the following are recognized: {_EFS_METRICS_COEFFS_KEYS}"
    )
    ignore_nan = {} if ignore_nan is None else ignore_nan
    metrics = [
        {
            "name": "total_energy_rmse",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("total_energy_rmse", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.TOTAL_ENERGY_KEY, False),
        },
        {
            "name": "total_energy_mae",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("total_energy_mae", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.TOTAL_ENERGY_KEY, False),
        },
        {
            "name": "per_atom_energy_rmse",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("per_atom_energy_rmse", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.TOTAL_ENERGY_KEY, False),
        },
        {
            "name": "per_atom_energy_mae",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("per_atom_energy_mae", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.TOTAL_ENERGY_KEY, False),
        },
        {
            "name": "forces_rmse",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("forces_rmse", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.FORCE_KEY, False),
        },
        {
            "name": "forces_mae",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("forces_mae", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.FORCE_KEY, False),
        },
        {
            "name": "stress_rmse",
            "field": AtomicDataDict.STRESS_KEY,
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("stress_rmse", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.STRESS_KEY, False),
        },
        {
            "name": "stress_mae",
            "field": AtomicDataDict.STRESS_KEY,
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("stress_mae", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.STRESS_KEY, False),
        },
        {
            "name": "total_energy_maxabserr",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": MaximumAbsoluteError(),
            "coeff": coeffs.get("total_energy_maxabserr", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.TOTAL_ENERGY_KEY, False),
        },
        {
            "name": "per_atom_energy_maxabserr",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": MaximumAbsoluteError(),
            "coeff": coeffs.get("per_atom_energy_maxabserr", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.TOTAL_ENERGY_KEY, False),
        },
        {
            "name": "forces_maxabserr",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": MaximumAbsoluteError(),
            "coeff": coeffs.get("forces_maxabserr", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.FORCE_KEY, False),
        },
        {
            "name": "stress_maxabserr",
            "field": AtomicDataDict.STRESS_KEY,
            "metric": MaximumAbsoluteError(),
            "coeff": coeffs.get("stress_maxabserr", None),
            "ignore_nan": ignore_nan.get(AtomicDataDict.STRESS_KEY, False),
        },
    ]
    return MetricsManager(
        _with_extra_metrics(metrics, extra_metrics), type_names=type_names
    )


def EnergyOnlyLoss(
    per_atom_energy: bool = True,
    type_names: Optional[List[str]] = None,
    extra_metrics: Optional[List[Dict[str, Any]]] = None,
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
        extra_metrics (list of dict, optional): additional metric entries appended to the wrapper's defaults
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
    return MetricsManager(
        _with_extra_metrics(metrics, extra_metrics), type_names=type_names
    )


_ENERGY_ONLY_METRICS_COEFFS_KEYS: Final[List[str]] = [
    "total_energy_rmse",
    "per_atom_energy_rmse",
    "total_energy_mae",
    "per_atom_energy_mae",
    "total_energy_maxabserr",
    "per_atom_energy_maxabserr",
]


def EnergyOnlyMetrics(
    coeffs: Dict[str, Optional[float]] = {
        "total_energy_rmse": 1.0,
        "per_atom_energy_rmse": None,
        "total_energy_mae": None,
        "per_atom_energy_mae": None,
        "total_energy_maxabserr": None,
        "per_atom_energy_maxabserr": None,
    },
    type_names: Optional[List[str]] = None,
    extra_metrics: Optional[List[Dict[str, Any]]] = None,
):
    """Simplified :class:`MetricsManager` wrapper for a **metric** term containing only energy mean absolute errors (MAEs), root mean squared errors (RMSEs), and maximum absolute errors (MaxAbsErrs).

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
              total_energy_maxabserr: null
              per_atom_energy_maxabserr: null

    Args:
        coeffs (Dict[str, float]): ``dict`` that stores the relative contribution of the different energy metrics to the ``weighted_sum`` version of the metric as in ``nequip.train.MetricsManager`` (default ``{'total_energy_rmse': 1.0, 'per_atom_energy_rmse': None, 'total_energy_mae': None, 'per_atom_energy_mae': None, 'total_energy_maxabserr': None, 'per_atom_energy_maxabserr': None}``)
        extra_metrics (list of dict, optional): additional metric entries appended to the wrapper's defaults
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
        {
            "name": "total_energy_maxabserr",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": MaximumAbsoluteError(),
            "coeff": coeffs.get("total_energy_maxabserr", None),
        },
        {
            "name": "per_atom_energy_maxabserr",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": MaximumAbsoluteError(),
            "coeff": coeffs.get("per_atom_energy_maxabserr", None),
        },
    ]
    return MetricsManager(
        _with_extra_metrics(metrics, extra_metrics), type_names=type_names
    )
