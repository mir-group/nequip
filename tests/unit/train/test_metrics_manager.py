import pytest
import torch
from nequip.data import AtomicDataDict, PerAtomModifier
from nequip.train import (
    MetricsManager,
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
)


class TestMetricsManager:

    @pytest.mark.parametrize(
        "ratio", [(1, 1), (1, 2), (1, 3), (10, 10), (1, 100), (1, 1e5)]
    )
    def test_basic_and_weighted_sum(self, data, ratio):
        pred1, ref1, pred2, ref2 = data
        mm = MetricsManager(
            [
                {
                    "field": AtomicDataDict.TOTAL_ENERGY_KEY,
                    "coeff": ratio[0],
                    "metric": MeanAbsoluteError(),
                },
                {
                    "field": AtomicDataDict.FORCE_KEY,
                    "coeff": ratio[1],
                    "metric": MeanAbsoluteError(),
                },
            ]
        )

        for pred, ref in [(pred1, ref1), (pred2, ref2)]:
            metrics_dict = mm(pred, ref)
            assert len(metrics_dict) > 0
            assert isinstance(metrics_dict, dict)
            for key, value in metrics_dict.items():
                assert isinstance(value, torch.Tensor)
            E_MAE = torch.mean(
                torch.abs(
                    pred[AtomicDataDict.TOTAL_ENERGY_KEY]
                    - ref[AtomicDataDict.TOTAL_ENERGY_KEY]
                )
            )
            F_MAE = torch.mean(
                torch.abs(
                    pred[AtomicDataDict.FORCE_KEY] - ref[AtomicDataDict.FORCE_KEY]
                )
            )
            weighted_sum = (ratio[0] * E_MAE + ratio[1] * F_MAE) / sum(ratio)
            assert torch.allclose(metrics_dict["E_mae"], E_MAE)
            assert torch.allclose(metrics_dict["F_mae"], F_MAE)
            assert torch.allclose(metrics_dict["weighted_sum"], weighted_sum)

        E_MAE = torch.mean(
            torch.abs(
                torch.cat(
                    [
                        pred1[AtomicDataDict.TOTAL_ENERGY_KEY],
                        pred2[AtomicDataDict.TOTAL_ENERGY_KEY],
                    ]
                )
                - torch.cat(
                    [
                        ref1[AtomicDataDict.TOTAL_ENERGY_KEY],
                        ref2[AtomicDataDict.TOTAL_ENERGY_KEY],
                    ]
                )
            )
        )
        F_MAE = torch.mean(
            torch.abs(
                torch.cat(
                    [pred1[AtomicDataDict.FORCE_KEY], pred2[AtomicDataDict.FORCE_KEY]]
                )
                - torch.cat(
                    [ref1[AtomicDataDict.FORCE_KEY], ref2[AtomicDataDict.FORCE_KEY]]
                )
            )
        )
        weighted_sum = (ratio[0] * E_MAE + ratio[1] * F_MAE) / sum(ratio)
        metrics_dict = mm.compute()
        assert torch.allclose(metrics_dict["E_mae"], E_MAE)
        assert torch.allclose(metrics_dict["F_mae"], F_MAE)
        assert torch.allclose(metrics_dict["weighted_sum"], weighted_sum)

    def test_per_type(self, data):
        pred, ref, pred2, ref2 = data
        mm = MetricsManager(
            [
                {
                    "field": AtomicDataDict.FORCE_KEY,
                    "per_type": True,
                    "coeff": 1.0,
                    "metric": MeanSquaredError(),
                    "name": "per_type_force_MSE",
                }
            ],
            type_names=["0", "1"],
        )
        metrics_dict = mm(pred, ref)

        # first half data are type 1
        loss_ref_1 = torch.square(
            pred[AtomicDataDict.FORCE_KEY][:5] - ref[AtomicDataDict.FORCE_KEY][:5]
        ).mean()
        loss_ref_0 = torch.square(
            pred[AtomicDataDict.FORCE_KEY][5:] - ref[AtomicDataDict.FORCE_KEY][5:]
        ).mean()

        assert torch.isclose(
            metrics_dict["per_type_force_MSE"], (loss_ref_0 + loss_ref_1) / 2.0
        )

    def test_per_atom(self, data):
        pred, ref, pred2, ref2 = data
        mm = MetricsManager(
            [
                {
                    "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
                    "coeff": 1.0,
                    "metric": RootMeanSquaredError(),
                    "name": "per_atom_energy_MSE",
                }
            ],
        )
        metrics_dict = mm(pred, ref)

        # first graph
        loss_ref_1 = torch.square(
            (
                pred[AtomicDataDict.TOTAL_ENERGY_KEY][0]
                - ref[AtomicDataDict.TOTAL_ENERGY_KEY][0]
            )
            / 3.0
        )
        # second graph
        loss_ref_2 = torch.square(
            (
                pred[AtomicDataDict.TOTAL_ENERGY_KEY][1]
                - ref[AtomicDataDict.TOTAL_ENERGY_KEY][1]
            )
            / 7.0
        )
        loss_ref = torch.sqrt((loss_ref_1 + loss_ref_2) / 2.0)
        assert torch.isclose(metrics_dict["per_atom_energy_MSE"], loss_ref)


class TestNaN:
    def test_basic(self, data_w_NaN):
        pred, ref, wo_nan_pred, wo_nan_ref = data_w_NaN
        mm = MetricsManager(
            [
                {
                    "field": AtomicDataDict.TOTAL_ENERGY_KEY,
                    "coeff": 3.0,
                    "metric": MeanAbsoluteError(),
                    "ignore_nan": True,
                    "name": "energy_MAE",
                },
                {
                    "field": AtomicDataDict.FORCE_KEY,
                    "coeff": 1.0,
                    "metric": MeanSquaredError(),
                    "ignore_nan": True,
                    "name": "force_MSE",
                },
            ],
        )
        metrics_dict = mm(pred, ref)
        metrics_dict_wo_nan = mm(wo_nan_pred, wo_nan_ref)
        for k in metrics_dict.keys():
            assert torch.isclose(metrics_dict[k], metrics_dict_wo_nan[k])

    def test_per_type(self, data_w_NaN):
        pred, ref, wo_nan_pred, wo_nan_ref = data_w_NaN
        mm = MetricsManager(
            [
                {
                    "field": AtomicDataDict.FORCE_KEY,
                    "per_type": True,
                    "metric": MeanSquaredError(),
                    "ignore_nan": True,
                    "name": "per_type_force_MSE",
                },
            ],
            type_names=["a", "b"],
        )
        metrics_dict = mm(pred, ref)
        metrics_dict_wo_nan = mm(wo_nan_pred, wo_nan_ref)
        assert torch.isclose(
            metrics_dict["per_type_force_MSE"],
            metrics_dict_wo_nan["per_type_force_MSE"],
        )

    def test_per_atom(self, data_w_NaN):
        pred, ref, wo_nan_pred, wo_nan_ref = data_w_NaN
        mm = MetricsManager(
            [
                {
                    "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
                    "metric": MeanSquaredError(),
                    "ignore_nan": True,
                    "name": "per_atom_E_MSE",
                },
            ],
        )
        metrics_dict = mm(pred, ref)
        metrics_dict_wo_nan = mm(wo_nan_pred, wo_nan_ref)
        assert torch.isclose(
            metrics_dict["per_atom_E_MSE"], metrics_dict_wo_nan["per_atom_E_MSE"]
        )

        # first half data are species 1
        loss_ref = torch.square(
            (
                pred[AtomicDataDict.TOTAL_ENERGY_KEY][0]
                - ref[AtomicDataDict.TOTAL_ENERGY_KEY][0]
            )
            / 3.0
        )

        assert torch.isclose(metrics_dict["per_atom_E_MSE"], loss_ref)


@pytest.fixture(scope="function")
def data():
    """"""
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int64)
    num_atoms = torch.tensor([3, 7], dtype=torch.int64)
    atom_types = torch.as_tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    pred1 = {
        AtomicDataDict.BATCH_KEY: batch,
        AtomicDataDict.NUM_NODES_KEY: num_atoms,
        AtomicDataDict.FORCE_KEY: torch.rand(10, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((2, 1)),
        AtomicDataDict.ATOM_TYPE_KEY: atom_types,
    }
    ref1 = {
        AtomicDataDict.BATCH_KEY: batch,
        AtomicDataDict.NUM_NODES_KEY: num_atoms,
        AtomicDataDict.FORCE_KEY: torch.rand(10, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((2, 1)),
        AtomicDataDict.ATOM_TYPE_KEY: atom_types,
    }
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=torch.int64)
    num_atoms = torch.tensor([4, 6, 3], dtype=torch.int64)
    atom_types = torch.as_tensor([1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
    pred2 = {
        AtomicDataDict.BATCH_KEY: batch,
        AtomicDataDict.NUM_NODES_KEY: num_atoms,
        AtomicDataDict.FORCE_KEY: torch.rand(13, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((3, 1)),
        AtomicDataDict.ATOM_TYPE_KEY: atom_types,
    }
    ref2 = {
        AtomicDataDict.BATCH_KEY: batch,
        AtomicDataDict.NUM_NODES_KEY: num_atoms,
        AtomicDataDict.FORCE_KEY: torch.rand(13, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((3, 1)),
        AtomicDataDict.ATOM_TYPE_KEY: atom_types,
    }
    yield pred1, ref1, pred2, ref2


@pytest.fixture(scope="function")
def data_w_NaN(data):
    """"""
    _pred, _ref, _, _ = data

    pred = {k: torch.clone(v) for k, v in _pred.items()}
    ref = {k: torch.clone(v) for k, v in _ref.items()}
    ref[AtomicDataDict.FORCE_KEY][-1] = float("nan")
    ref[AtomicDataDict.FORCE_KEY][0] = float("nan")
    ref[AtomicDataDict.TOTAL_ENERGY_KEY][1] = float("nan")

    wo_nan_pred = {k: torch.clone(v) for k, v in _pred.items()}
    wo_nan_ref = {k: torch.clone(v) for k, v in _ref.items()}
    wo_nan_ref[AtomicDataDict.TOTAL_ENERGY_KEY] = wo_nan_ref[
        AtomicDataDict.TOTAL_ENERGY_KEY
    ][:1]
    wo_nan_ref[AtomicDataDict.NUM_NODES_KEY] = wo_nan_ref[AtomicDataDict.NUM_NODES_KEY][
        :1
    ]
    wo_nan_ref[AtomicDataDict.FORCE_KEY] = wo_nan_ref[AtomicDataDict.FORCE_KEY][1:-1]
    wo_nan_ref[AtomicDataDict.ATOM_TYPE_KEY] = wo_nan_ref[AtomicDataDict.ATOM_TYPE_KEY][
        1:-1
    ]
    wo_nan_pred[AtomicDataDict.TOTAL_ENERGY_KEY] = wo_nan_pred[
        AtomicDataDict.TOTAL_ENERGY_KEY
    ][:1]
    wo_nan_pred[AtomicDataDict.NUM_NODES_KEY] = wo_nan_pred[
        AtomicDataDict.NUM_NODES_KEY
    ][:1]
    wo_nan_pred[AtomicDataDict.FORCE_KEY] = wo_nan_pred[AtomicDataDict.FORCE_KEY][1:-1]
    wo_nan_pred[AtomicDataDict.ATOM_TYPE_KEY] = wo_nan_pred[
        AtomicDataDict.ATOM_TYPE_KEY
    ][1:-1]

    yield pred, ref, wo_nan_pred, wo_nan_ref
