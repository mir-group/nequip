import numpy as np
import torch
from nequip.data import AtomicDataDict
from nequip.data import (
    DataStatisticsManager,
    EnergyOnlyDataStatisticsManager,
    PerAtomModifier,
    EdgeLengths,
    NumNeighbors,
)
from nequip.data import (
    Count,
    Mean,
    RootMeanSquare,
    StandardDeviation,
    Max,
    Min,
)
from nequip.nn import with_edge_vectors_
import pytest


class TestStatistics:
    # implicitly testing automatic naming convention, i.e.
    # energy (E), forces (F), etc

    @pytest.mark.parametrize("num_trainval_test", [(200, 100), (500, 200)])
    @pytest.mark.parametrize("batch_size", [1, 2, 5, 50])
    def test_statistics(self, TolueneDataModule):
        _, dloader, batch = TolueneDataModule

        metrics = [
            {
                "field": "total_energy",
                "metric": Mean(),
            },
            {
                "field": "total_energy",
                "metric": StandardDeviation(),
            },
            {
                "field": PerAtomModifier("total_energy"),
                "metric": Mean(),
            },
            {
                "field": PerAtomModifier("total_energy"),
                "metric": StandardDeviation(),
            },
            {
                "field": "forces",
                "metric": Mean(),
            },
            {
                "field": "forces",
                "metric": StandardDeviation(),
            },
            {
                "field": "forces",
                "metric": RootMeanSquare(),
            },
            {
                "field": "forces",
                "metric": Max(abs=True),
            },
            {
                "field": "num_atoms",
                "metric": Mean(),
            },
            {
                "field": NumNeighbors(),
                "metric": Mean(),
            },
            {
                "field": EdgeLengths(),
                "metric": Mean(),
            },
            {
                "field": EdgeLengths(),
                "metric": StandardDeviation(),
            },
            {
                "field": EdgeLengths(),
                "metric": Min(),
            },
        ]
        stats_manager = DataStatisticsManager(metrics)
        stats_dict = stats_manager.get_statistics(dloader)

        # test energy
        eng = batch[AtomicDataDict.TOTAL_ENERGY_KEY]
        assert np.allclose(stats_dict["E_mean"], torch.mean(eng).item())
        assert np.allclose(stats_dict["E_std"], torch.std(eng).item())
        eng_per_atom = eng.reshape(-1) / batch[AtomicDataDict.NUM_NODES_KEY]
        assert np.allclose(
            stats_dict["per_atom_E_mean"], torch.mean(eng_per_atom).item()
        )
        assert np.allclose(stats_dict["per_atom_E_std"], torch.std(eng_per_atom).item())

        # test force
        f_raveled = batch[AtomicDataDict.FORCE_KEY].flatten()
        assert np.allclose(stats_dict["F_mean"], torch.mean(f_raveled).item())
        assert np.allclose(stats_dict["F_std"], torch.std(f_raveled).item())
        assert np.allclose(
            stats_dict["F_rms"], torch.sqrt(torch.mean(f_raveled.square())).item()
        )
        assert np.allclose(stats_dict["F_absmax"], f_raveled.abs().max().item())

        # test num atoms, num neighbors
        expected_avg_num_atoms = torch.mean(
            batch[AtomicDataDict.NUM_NODES_KEY].to(torch.get_default_dtype())
        )
        assert np.allclose(stats_dict["num_atoms_mean"], expected_avg_num_atoms.item())

        expected_ave_num_neighbors = torch.mean(
            torch.unique(batch[AtomicDataDict.EDGE_INDEX_KEY][0], return_counts=True)[
                1
            ].to(torch.get_default_dtype())
        )
        assert np.allclose(
            stats_dict["num_neighbors_mean"], expected_ave_num_neighbors.item()
        )

        data = with_edge_vectors_(batch, with_lengths=True)
        lengths = data[AtomicDataDict.EDGE_LENGTH_KEY]
        assert np.allclose(stats_dict["edge_lengths_mean"], torch.mean(lengths).item())
        assert np.allclose(stats_dict["edge_lengths_std"], torch.std(lengths).item())
        assert np.allclose(stats_dict["edge_lengths_min"], lengths.min().item())

    @pytest.mark.parametrize("num_trainval_test", [(200, 100), (500, 200)])
    @pytest.mark.parametrize("batch_size", [1, 2, 5, 50])
    def test_per_type(self, TolueneDataModule):
        _, dloader, batch = TolueneDataModule

        metrics = [
            {
                "field": "forces",
                "per_type": True,
                "metric": Mean(),
            },
            {
                "field": "forces",
                "per_type": True,
                "metric": StandardDeviation(),
            },
            {
                "field": "forces",
                "per_type": True,
                "metric": RootMeanSquare(),
            },
            {
                "field": "atom_types",
                "per_type": True,
                "metric": Count(),
            },
            {
                "field": EdgeLengths(),
                "per_type": True,
                "metric": StandardDeviation(),
            },
        ]
        stats_manager = DataStatisticsManager(metrics, type_names=["C", "H"])
        stats_dict = stats_manager.get_statistics(dloader)

        atom_types = batch[AtomicDataDict.ATOM_TYPE_KEY]
        C_forces = batch[AtomicDataDict.FORCE_KEY][atom_types == 0]
        H_forces = batch[AtomicDataDict.FORCE_KEY][atom_types == 1]
        assert np.allclose(stats_dict["F_mean_C"], torch.mean(C_forces).item())
        assert np.allclose(stats_dict["F_mean_H"], torch.mean(H_forces).item())
        assert np.allclose(stats_dict["F_std_C"], torch.std(C_forces).item())
        assert np.allclose(stats_dict["F_std_H"], torch.std(H_forces).item())
        assert np.allclose(
            stats_dict["F_rms_C"], torch.sqrt(torch.mean(C_forces.square())).item()
        )
        assert np.allclose(
            stats_dict["F_rms_H"], torch.sqrt(torch.mean(H_forces.square())).item()
        )

        _, count = torch.unique(batch[AtomicDataDict.ATOM_TYPE_KEY], return_counts=True)
        assert (
            stats_dict["atom_types_count_C"] == count[0]
            and stats_dict["atom_types_count_H"] == count[1]
        )
        data = with_edge_vectors_(batch, with_lengths=True)
        edge_type = torch.index_select(
            data[AtomicDataDict.ATOM_TYPE_KEY].reshape(-1),
            0,
            data[AtomicDataDict.EDGE_INDEX_KEY].reshape(-1),
        ).view(2, -1)
        lengths = data[AtomicDataDict.EDGE_LENGTH_KEY]
        CC_type = torch.logical_and(edge_type[0] == 0, edge_type[1] == 0)
        CH_type = torch.logical_and(edge_type[0] == 0, edge_type[1] == 1)
        HC_type = torch.logical_and(edge_type[0] == 1, edge_type[1] == 0)
        HH_type = torch.logical_and(edge_type[0] == 1, edge_type[1] == 1)
        assert np.allclose(
            stats_dict["edge_lengths_std_CC"], torch.std(lengths[CC_type]).item()
        )
        assert np.allclose(
            stats_dict["edge_lengths_std_CH"], torch.std(lengths[CH_type]).item()
        )
        assert np.allclose(
            stats_dict["edge_lengths_std_HC"], torch.std(lengths[HC_type]).item()
        )
        assert np.allclose(
            stats_dict["edge_lengths_std_HH"], torch.std(lengths[HH_type]).item()
        )

    @pytest.mark.parametrize("num_trainval_test", [(200, 100), (500, 200)])
    @pytest.mark.parametrize("batch_size", [1, 2, 5, 50])
    def test_energy_only_statistics_manager(self, TolueneDataModule):
        """test EnergyOnlyDataStatisticsManager for energy-only datasets."""
        _, dloader, batch = TolueneDataModule

        stats_manager = EnergyOnlyDataStatisticsManager(type_names=["C", "H"])
        stats_dict = stats_manager.get_statistics(dloader)

        # test that all expected energy-only statistics are computed
        expected_stats = [
            "num_neighbors_mean",
            "per_type_num_neighbors_mean",
            "per_atom_energy_mean",
            "per_atom_energy_std",
            "total_energy_std",
        ]
        for stat_name in expected_stats:
            assert stat_name in stats_dict, f"Missing stat: {stat_name}"

        # verify num_neighbors_mean matches manual calculation
        expected_ave_num_neighbors = torch.mean(
            torch.unique(batch[AtomicDataDict.EDGE_INDEX_KEY][0], return_counts=True)[
                1
            ].to(torch.get_default_dtype())
        )
        assert np.allclose(
            stats_dict["num_neighbors_mean"], expected_ave_num_neighbors.item()
        )

        # verify per_atom_energy_mean and per_atom_energy_std
        eng = batch[AtomicDataDict.TOTAL_ENERGY_KEY]
        eng_per_atom = eng.reshape(-1) / batch[AtomicDataDict.NUM_NODES_KEY]
        assert np.allclose(
            stats_dict["per_atom_energy_mean"], torch.mean(eng_per_atom).item()
        )
        assert np.allclose(
            stats_dict["per_atom_energy_std"], torch.std(eng_per_atom).item()
        )

        # verify total_energy_std
        assert np.allclose(stats_dict["total_energy_std"], torch.std(eng).item())

        # verify per-type neighbor statistics exist
        assert "per_type_num_neighbors_mean" in stats_dict
        per_type_stats = stats_dict["per_type_num_neighbors_mean"]
        assert "C" in per_type_stats and "H" in per_type_stats
