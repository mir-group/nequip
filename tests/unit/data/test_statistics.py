import torch
from nequip.data import AtomicDataDict
from nequip.data import (
    DataStatisticsManager,
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
        assert torch.allclose(stats_dict["E_mean"], torch.mean(eng))
        assert torch.allclose(stats_dict["E_std"], torch.std(eng))
        eng_per_atom = eng.reshape(-1) / batch[AtomicDataDict.NUM_NODES_KEY]
        assert torch.allclose(stats_dict["per_atom_E_mean"], torch.mean(eng_per_atom))
        assert torch.allclose(stats_dict["per_atom_E_std"], torch.std(eng_per_atom))

        # test force
        f_raveled = batch[AtomicDataDict.FORCE_KEY].flatten()
        assert torch.allclose(stats_dict["F_mean"], torch.mean(f_raveled))
        assert torch.allclose(stats_dict["F_std"], torch.std(f_raveled))
        assert torch.allclose(
            stats_dict["F_rms"], torch.sqrt(torch.mean(f_raveled.square()))
        )
        assert torch.allclose(stats_dict["F_absmax"], f_raveled.abs().max())

        # test num atoms, num neighbors
        expected_avg_num_atoms = torch.mean(
            batch[AtomicDataDict.NUM_NODES_KEY].to(torch.get_default_dtype())
        )
        assert torch.allclose(stats_dict["num_atoms_mean"], expected_avg_num_atoms)

        expected_ave_num_neighbors = torch.mean(
            torch.unique(batch[AtomicDataDict.EDGE_INDEX_KEY][0], return_counts=True)[
                1
            ].to(torch.get_default_dtype())
        )
        assert torch.allclose(
            stats_dict["num_neighbors_mean"], expected_ave_num_neighbors
        )

        data = AtomicDataDict.with_edge_vectors(batch, with_lengths=True)
        lengths = data[AtomicDataDict.EDGE_LENGTH_KEY]
        assert torch.allclose(stats_dict["edge_lengths_mean"], torch.mean(lengths))
        assert torch.allclose(stats_dict["edge_lengths_std"], torch.std(lengths))
        assert torch.allclose(stats_dict["edge_lengths_min"], lengths.min())

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
        assert torch.allclose(stats_dict["F_mean_C"], torch.mean(C_forces))
        assert torch.allclose(stats_dict["F_mean_H"], torch.mean(H_forces))
        assert torch.allclose(stats_dict["F_std_C"], torch.std(C_forces))
        assert torch.allclose(stats_dict["F_std_H"], torch.std(H_forces))
        assert torch.allclose(
            stats_dict["F_rms_C"], torch.sqrt(torch.mean(C_forces.square()))
        )
        assert torch.allclose(
            stats_dict["F_rms_H"], torch.sqrt(torch.mean(H_forces.square()))
        )

        _, count = torch.unique(batch[AtomicDataDict.ATOM_TYPE_KEY], return_counts=True)
        assert (
            stats_dict["atom_types_count_C"] == count[0]
            and stats_dict["atom_types_count_H"] == count[1]
        )
        data = AtomicDataDict.with_edge_vectors(batch, with_lengths=True)
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
        assert torch.allclose(
            stats_dict["edge_lengths_std_CC"], torch.std(lengths[CC_type])
        )
        assert torch.allclose(
            stats_dict["edge_lengths_std_CH"], torch.std(lengths[CH_type])
        )
        assert torch.allclose(
            stats_dict["edge_lengths_std_HC"], torch.std(lengths[HC_type])
        )
        assert torch.allclose(
            stats_dict["edge_lengths_std_HH"], torch.std(lengths[HH_type])
        )
