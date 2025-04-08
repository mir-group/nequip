# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from torchmetrics import Metric
from . import AtomicDataDict

from .modifier import BaseModifier, PerAtomModifier, NumNeighbors
from .stats import Mean, RootMeanSquare
from typing import List, Dict, Union, Callable, Iterable

from nequip.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class DataStatisticsManager(torch.nn.ModuleList):
    """Manages ``nequip`` metrics that can be applied to ``AtomicDataDict`` s to compute dataset statistics.

    The main input argument ``metrics`` is a list of dictionaries, where each dictionary contains the following keys.

    There are two mandatory keys.

      - ``field`` refers to the quantity of interest for metric computation. It has two formats.

         - a ``str`` for a ``nequip`` defined field (e.g. ``total_energy``, ``forces``, ``stress``), or
         - a ``Callable`` that performs some additional operations before returning a ``torch.Tensor``
           for metric computation (e.g. ``nequip.data.PerAtomModifier``).
      - ``metric`` is a ``nequip`` data metric object (a subclass of ``torchmetrics.Metric``).

    The remaining keys are optional.

      - ``per_type`` is a ``bool`` (defaults to ``False`` if not provided). If ``True``, node fields (such as ``forces``) will have their metrics computed separately for each node type based on the ``type_names`` argument.

      - ``ignore_nan`` is a ``bool`` (defaults to ``False`` if not provided). This should be set to true if one expects the underlying ``target`` data to contain ``NaN`` entries.

      - ``name`` is the name that the metric is logged as. Default names are used if not provided, but it is recommended for users to set custom names for clarity and control.

    Args:
        metrics (list): list of dictionaries with keys ``field``, ``metric``, ``per_type``, ``ignore_nan``, and ``name``
        dataloader_kwargs (dict): arguments of `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ for dataset statitstics computation (ideally, the ``batch_size`` should be as large as possible without triggering OOM)
        type_names (list): required for ``per_type`` metrics (this must match the ``type_names`` argument of the model, it is advisable to use variable interpolation in the config file to make sure they are consistent)
    """

    def __init__(
        self,
        metrics: List[
            Dict[str, Union[float, str, Dict[str, Union[str, Callable]], Metric]]
        ],
        dataloader_kwargs: Dict = {},
        type_names: List[str] = None,
    ):
        super().__init__()
        assert len(metrics) != 0

        assert all(
            key not in dataloader_kwargs
            for key in ["dataset", "generator", "collate_fn"]
        )
        self.dataloader_kwargs = dataloader_kwargs

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
                assert field_type in [
                    "node",
                    "edge",
                ], f"`per_type` metrics only apply to node or edge fields, but {field_type} field found for {self.names[idx]}."
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
        self.stats_dict = {}

    def forward(
        self,
        data: AtomicDataDict.Type,
    ):
        """"""
        for idx in range(self.num_metrics):
            data_tensor = self.fields[idx](data)

            if self.per_type[idx]:
                field_type = self.fields[idx].type
                if field_type == "node":
                    for type_idx in range(len(self.type_names)):
                        # index out each type
                        selector = torch.eq(
                            data[AtomicDataDict.ATOM_TYPE_KEY], type_idx
                        )
                        per_type_data_tensor = data_tensor[selector]
                        if self.ignore_nans[idx]:
                            notnan_mask = ~torch.isnan(per_type_data_tensor)
                            per_type_data_tensor = torch.masked_select(
                                per_type_data_tensor, notnan_mask
                            )
                        _ = self[idx][type_idx](per_type_data_tensor)
                elif field_type == "edge":
                    # index out each type pair
                    edge_type = torch.index_select(
                        data[AtomicDataDict.ATOM_TYPE_KEY].reshape(-1),
                        0,
                        data[AtomicDataDict.EDGE_INDEX_KEY].reshape(-1),
                    ).view(2, -1)
                    edge_type = edge_type[0] * len(self.type_names) + edge_type[1]
                    for type_idx in range(len(self.type_names) * len(self.type_names)):
                        selector = torch.eq(edge_type, type_idx)
                        per_type_data_tensor = data_tensor[selector]
                        if self.ignore_nans[idx]:
                            notnan_mask = ~torch.isnan(per_type_data_tensor)
                            per_type_data_tensor = torch.masked_select(
                                per_type_data_tensor, notnan_mask
                            )
                        _ = self[idx][type_idx](per_type_data_tensor)
            else:
                if self.ignore_nans[idx]:
                    notnan_mask = ~torch.isnan(data_tensor)
                    data_tensor = torch.masked_select(data_tensor, notnan_mask)
                _ = self[idx](data_tensor)

    def compute(self):
        logger.info("Computed data statistics:")
        self.stats_dict = {}
        for idx in range(self.num_metrics):
            if self.per_type[idx]:
                field_type = self.fields[idx].type
                pt_stats = []
                if field_type == "node":
                    for type_idx, type_name in enumerate(self.type_names):
                        pt_stat = self[idx][type_idx].compute()
                        pt_stats.append(pt_stat.item())
                        pt_stat_name = "_".join([self.names[idx], type_name])
                        self.stats_dict.update({pt_stat_name: pt_stat})
                        logger.info(f"{pt_stat_name}: {pt_stat}")
                elif field_type == "edge":
                    for center_idx, center_type in enumerate(self.type_names):
                        for neigh_idx, neigh_type in enumerate(self.type_names):
                            type_pair_idx = (
                                center_idx + len(self.type_names) * neigh_idx
                            )
                            pt_stat = self[idx][type_pair_idx].compute()
                            pt_stats.append(pt_stat.item())
                            pt_stat_name = "_".join(
                                [self.names[idx], center_type + neigh_type]
                            )
                            self.stats_dict.update({pt_stat_name: pt_stat})
                            logger.info(f"{pt_stat_name}: {pt_stat}")
                self.stats_dict.update({self.names[idx]: pt_stats})
            else:
                stat = self[idx].compute()
                self.stats_dict.update({self.names[idx]: stat.item()})
                logger.info(f"{self.names[idx]}: {stat}")
        return self.stats_dict

    def reset(self):
        """Resets accumulated statistics."""
        for idx in range(self.num_metrics):
            if self.per_type[idx]:
                field_type = self.fields[idx].type
                if field_type == "node":
                    num_types = len(self.type_names)
                elif field_type == "edge":
                    num_types = len(self.type_names) * len(self.type_names)
                for type_idx in range(num_types):
                    self[idx][type_idx].reset()
            else:
                self[idx].reset()

    def get_statistics(self, data_source: Iterable[AtomicDataDict.Type]):
        """
        Remember to call reset before this is needed.

        Args:
            data_source (Iterable[AtomicDataDict]): iterable data source
        """
        for data in data_source:
            self(data)
        return self.compute()


def CommonDataStatisticsManager(
    dataloader_kwargs: Dict = {},
    type_names: List[str] = None,
):
    """``DataStatisticsManager`` wrapper that implements common dataset statistics.

    The dataset statistics computed by using this wrapper include ``num_neighbors_mean``, ``per_atom_energy_mean``, ``forces_rms``, and ``per_type_forces_rms``, which are variables that can be interpolated for in the ``model`` section of the config file.

    For example::

        training_module:
        _target_: nequip.train.EMALightningModule

        # other `EMALightningModule` arguments

        model:
          _target_: nequip.model.NequIPGNNModel

          # other model hyperparameters
          avg_num_neighbors: ${training_data_stats:num_neighbors_mean}
          per_type_energy_shifts: ${training_data_stats:per_atom_energy_mean}
          per_type_energy_scales: ${training_data_stats:forces_rms}
          # or alternatively the per-type forces RMS
          # per_type_energy_scales: ${training_data_stats:per_type_forces_rms}

    """
    metrics = [
        {
            "name": "num_neighbors_mean",
            "field": NumNeighbors(),
            "metric": Mean(),
        },
        {
            "name": "per_type_num_neighbors_mean",
            "field": NumNeighbors(),
            "metric": Mean(),
            "per_type": True,
        },
        {
            "name": "per_atom_energy_mean",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": Mean(),
        },
        {
            "name": "forces_rms",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": RootMeanSquare(),
        },
        {
            "name": "per_type_forces_rms",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": RootMeanSquare(),
            "per_type": True,
        },
    ]
    return DataStatisticsManager(metrics, dataloader_kwargs, type_names)
