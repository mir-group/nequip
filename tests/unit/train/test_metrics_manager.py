import pytest
import torch
from nequip.data import AtomicDataDict, PerAtomModifier
from nequip.data.transforms import AddNaNStressTransform
from nequip.train import (
    MetricsManager,
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
    EnergyForceLoss,
    EnergyForceMetrics,
    EnergyForceStressLoss,
    EnergyForceStressMetrics,
    EnergyOnlyLoss,
    EnergyOnlyMetrics,
)
from nequip.train.metrics import MaximumAbsoluteError


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
            torch.testing.assert_close(metrics_dict["E_mae"], E_MAE)
            torch.testing.assert_close(metrics_dict["F_mae"], F_MAE)
            torch.testing.assert_close(metrics_dict["weighted_sum"], weighted_sum)

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
        torch.testing.assert_close(metrics_dict["E_mae"], E_MAE)
        torch.testing.assert_close(metrics_dict["F_mae"], F_MAE)
        torch.testing.assert_close(metrics_dict["weighted_sum"], weighted_sum)

    @pytest.mark.parametrize("ratio", [(1, 1, 1), (1, 1, 2), (1, 2, 3), (1, 10, 100)])
    def test_stress_and_weighted_sum(self, data, ratio):
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
                {
                    "field": AtomicDataDict.STRESS_KEY,
                    "coeff": ratio[2],
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
            S_MAE = torch.mean(
                torch.abs(
                    pred[AtomicDataDict.STRESS_KEY] - ref[AtomicDataDict.STRESS_KEY]
                )
            )
            weighted_sum = (
                ratio[0] * E_MAE + ratio[1] * F_MAE + ratio[2] * S_MAE
            ) / sum(ratio)
            torch.testing.assert_close(metrics_dict["E_mae"], E_MAE)
            torch.testing.assert_close(metrics_dict["F_mae"], F_MAE)
            torch.testing.assert_close(metrics_dict["stress_mae"], S_MAE)
            torch.testing.assert_close(metrics_dict["weighted_sum"], weighted_sum)

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
        S_MAE = torch.mean(
            torch.abs(
                torch.cat(
                    [pred1[AtomicDataDict.STRESS_KEY], pred2[AtomicDataDict.STRESS_KEY]]
                )
                - torch.cat(
                    [ref1[AtomicDataDict.STRESS_KEY], ref2[AtomicDataDict.STRESS_KEY]]
                )
            )
        )
        weighted_sum = (ratio[0] * E_MAE + ratio[1] * F_MAE + ratio[2] * S_MAE) / sum(
            ratio
        )
        metrics_dict = mm.compute()
        torch.testing.assert_close(metrics_dict["E_mae"], E_MAE)
        torch.testing.assert_close(metrics_dict["F_mae"], F_MAE)
        torch.testing.assert_close(metrics_dict["stress_mae"], S_MAE)
        torch.testing.assert_close(metrics_dict["weighted_sum"], weighted_sum)

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

    def test_per_type_coeffs(self, data):
        pred, ref, _, _ = data
        # first half data are type 1
        mse_type1 = torch.square(
            pred[AtomicDataDict.FORCE_KEY][:5] - ref[AtomicDataDict.FORCE_KEY][:5]
        ).mean()
        mse_type0 = torch.square(
            pred[AtomicDataDict.FORCE_KEY][5:] - ref[AtomicDataDict.FORCE_KEY][5:]
        ).mean()

        def mm_with(weights):
            return MetricsManager(
                [
                    {
                        "field": AtomicDataDict.FORCE_KEY,
                        "per_type": True,
                        "per_type_coeffs": weights,
                        "metric": MeanSquaredError(),
                        "name": "per_type_force_MSE",
                    }
                ],
                type_names=["0", "1"],
            )

        # equal weights reproduce the equal-mean per_type behavior
        out = mm_with({"0": 1.0, "1": 1.0})(pred, ref)
        torch.testing.assert_close(
            out["per_type_force_MSE"], (mse_type0 + mse_type1) / 2.0
        )

        # non-equal weights produce a weighted mean
        out = mm_with({"0": 10.0, "1": 1.0})(pred, ref)
        torch.testing.assert_close(
            out["per_type_force_MSE"],
            (10.0 * mse_type0 + 1.0 * mse_type1) / 11.0,
        )

    @pytest.mark.parametrize(
        "weights,per_type,err_type,err_match",
        [
            # `per_type_coeffs` set but `per_type` flag missing
            ({"0": 1.0, "1": 1.0}, False, ValueError, "require `per_type: true`"),
            # extra key not in type_names
            ({"0": 1.0, "1": 1.0, "Xx": 2.0}, True, ValueError, "not in `type_names`"),
            # missing a type
            ({"0": 1.0}, True, ValueError, "missing"),
            # zero is rejected (typo guard)
            ({"0": 2.0, "1": 0.0}, True, ValueError, "must be positive"),
            # negative is rejected
            ({"0": -1.0, "1": 1.0}, True, ValueError, "must be positive"),
            # not a dict
            ([1.0, 2.0], True, TypeError, "must be a dict"),
        ],
    )
    def test_per_type_coeffs_validation(self, weights, per_type, err_type, err_match):
        spec = {
            "field": AtomicDataDict.FORCE_KEY,
            "per_type_coeffs": weights,
            "metric": MeanSquaredError(),
            "name": "per_type_force_MSE",
        }
        if per_type:
            spec["per_type"] = True
        with pytest.raises(err_type, match=err_match):
            MetricsManager([spec], type_names=["0", "1"])

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

    def test_stress_with_nan_transform(self, data):
        """Test EnergyForceStressLoss/Metrics with AddNaNStressTransform for partial stress data."""
        # get original fixture with stress
        pred1, ref1, _, _ = data

        # create a duplicate without stress key to simulate missing stress data
        pred2 = {k: torch.clone(v) for k, v in pred1.items()}
        ref2 = {k: torch.clone(v) for k, v in ref1.items()}
        del ref2[AtomicDataDict.STRESS_KEY]

        # apply AddNaNStressTransform to add NaN stress tensors
        transform = AddNaNStressTransform()
        ref2 = transform(ref2)

        # verify transform added NaN stress
        assert AtomicDataDict.STRESS_KEY in pred2
        assert AtomicDataDict.STRESS_KEY in ref2
        assert torch.all(torch.isnan(ref2[AtomicDataDict.STRESS_KEY]))

        # batch the two samples together into a single batch
        # so we have mixed stress tensors (some frames with real stress, some with NaN)
        pred_batched = AtomicDataDict.batched_from_list([pred1, pred2])
        ref_batched = AtomicDataDict.batched_from_list([ref1, ref2])

        # verify we have mixed stress (some real, some NaN)
        assert not torch.all(torch.isnan(ref_batched[AtomicDataDict.STRESS_KEY]))
        assert torch.any(torch.isnan(ref_batched[AtomicDataDict.STRESS_KEY]))

        # test with loss function - ignore_nan should handle mixed stress
        mm_loss = EnergyForceStressLoss(
            coeffs={
                AtomicDataDict.TOTAL_ENERGY_KEY: 1.0,
                AtomicDataDict.FORCE_KEY: 1.0,
                AtomicDataDict.STRESS_KEY: 1.0,
            },
            ignore_nan={AtomicDataDict.STRESS_KEY: True},
        )

        # pass batched data with mixed stress
        metrics = mm_loss(pred_batched, ref_batched)
        assert "per_atom_energy_mse" in metrics
        assert "forces_mse" in metrics
        assert "stress_mse" in metrics

        # test with metrics function
        mm_metrics = EnergyForceStressMetrics(
            coeffs={
                "total_energy_rmse": 1.0,
                "forces_rmse": 1.0,
                "stress_rmse": 1.0,
            },
            ignore_nan={AtomicDataDict.STRESS_KEY: True},
        )

        # pass batched data
        metrics = mm_metrics(pred_batched, ref_batched)
        assert "stress_rmse" in metrics
        assert "stress_mae" in metrics
        assert "stress_maxabserr" in metrics
        assert "total_energy_rmse" in metrics
        assert "forces_rmse" in metrics


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
        AtomicDataDict.STRESS_KEY: torch.rand((2, 3, 3)),
    }
    ref1 = {
        AtomicDataDict.BATCH_KEY: batch,
        AtomicDataDict.NUM_NODES_KEY: num_atoms,
        AtomicDataDict.FORCE_KEY: torch.rand(10, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((2, 1)),
        AtomicDataDict.ATOM_TYPE_KEY: atom_types,
        AtomicDataDict.STRESS_KEY: torch.rand((2, 3, 3)),
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
        AtomicDataDict.STRESS_KEY: torch.rand((3, 3, 3)),
    }
    ref2 = {
        AtomicDataDict.BATCH_KEY: batch,
        AtomicDataDict.NUM_NODES_KEY: num_atoms,
        AtomicDataDict.FORCE_KEY: torch.rand(13, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((3, 1)),
        AtomicDataDict.ATOM_TYPE_KEY: atom_types,
        AtomicDataDict.STRESS_KEY: torch.rand((3, 3, 3)),
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


class TestMetricsManagerBuilders:
    @pytest.mark.parametrize(
        "ratio", [(1, 1), (1, 2), (1, 3), (10, 10), (1, 100), (1, 1e5)]
    )
    @pytest.mark.parametrize("per_atom_energy", [True, False])
    def test_EnergyForceLoss(self, data, ratio, per_atom_energy):
        pred1, ref1, pred2, ref2 = data
        mm = EnergyForceLoss(
            coeffs={
                AtomicDataDict.TOTAL_ENERGY_KEY: ratio[0],
                AtomicDataDict.FORCE_KEY: ratio[1],
            },
            per_atom_energy=per_atom_energy,
        )

        for pred, ref in [(pred1, ref1), (pred2, ref2)]:
            metrics_dict = mm(pred, ref)
            assert len(metrics_dict) > 0
            assert isinstance(metrics_dict, dict)
            for key, value in metrics_dict.items():
                assert isinstance(value, torch.Tensor)
            E_MSE = self.compute_MSE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy
            )
            F_MSE = self.compute_MSE(pred, ref, AtomicDataDict.FORCE_KEY)
            weighted_sum = (ratio[0] * E_MSE + ratio[1] * F_MSE) / sum(ratio)
            torch.testing.assert_close(
                metrics_dict[
                    "per_atom_energy_mse" if per_atom_energy else "total_energy_mse"
                ],
                E_MSE,
            )
            torch.testing.assert_close(metrics_dict["forces_mse"], F_MSE)
            torch.testing.assert_close(metrics_dict["weighted_sum"], weighted_sum)

    @pytest.mark.parametrize("ratio1", [1])
    @pytest.mark.parametrize("ratio2", [1, 10])
    @pytest.mark.parametrize("ratio3", [1, 10, 100])
    @pytest.mark.parametrize("ratio4", [1, 10, 100, None])
    @pytest.mark.parametrize("ratio5", [1, 10])
    @pytest.mark.parametrize("ratio6", [1, 10])
    def test_EnergyForceMetrics(
        self, data, ratio1, ratio2, ratio3, ratio4, ratio5, ratio6
    ):
        pred1, ref1, pred2, ref2 = data
        mm = EnergyForceMetrics(
            coeffs={
                "total_energy_rmse": ratio1,
                "forces_rmse": ratio2,
                "total_energy_mae": ratio3,
                "forces_mae": ratio4,
                "per_atom_energy_rmse": ratio5,
                "per_atom_energy_mae": ratio6,
            },
        )

        for pred, ref in [(pred1, ref1), (pred2, ref2)]:
            metrics_dict = mm(pred, ref)
            assert len(metrics_dict) > 0
            assert isinstance(metrics_dict, dict)
            for key, value in metrics_dict.items():
                assert isinstance(value, torch.Tensor)
            E_RMSE = self.compute_RMSE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=False
            )
            F_RMSE = self.compute_RMSE(pred, ref, AtomicDataDict.FORCE_KEY)
            per_atom_E_RMSE = self.compute_RMSE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=True
            )
            E_MAE = self.compute_MAE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=False
            )
            F_MAE = self.compute_MAE(pred, ref, AtomicDataDict.FORCE_KEY)
            per_atom_E_MAE = self.compute_MAE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=True
            )
            if not ratio4:
                ratio4 = 0
            weighted_sum = (
                ratio1 * E_RMSE
                + ratio2 * F_RMSE
                + ratio3 * E_MAE
                + ratio4 * F_MAE
                + ratio5 * per_atom_E_RMSE
                + ratio6 * per_atom_E_MAE
            ) / (ratio1 + ratio2 + ratio3 + ratio4 + ratio5 + ratio6)
            torch.testing.assert_close(metrics_dict["total_energy_rmse"], E_RMSE)
            torch.testing.assert_close(metrics_dict["forces_rmse"], F_RMSE)
            torch.testing.assert_close(metrics_dict["total_energy_mae"], E_MAE)
            torch.testing.assert_close(metrics_dict["forces_mae"], F_MAE)
            torch.testing.assert_close(
                metrics_dict["per_atom_energy_rmse"], per_atom_E_RMSE
            )
            torch.testing.assert_close(
                metrics_dict["per_atom_energy_mae"], per_atom_E_MAE
            )
            torch.testing.assert_close(metrics_dict["weighted_sum"], weighted_sum)

    @pytest.mark.parametrize(
        "ratio", [(1, 1, 1), (1, 2, 3), (10, 10, 10), (1, 100, 1000), (1, 1e3, 1e5)]
    )
    @pytest.mark.parametrize("per_atom_energy", [True, False])
    def test_EnergyForceStressLoss(self, data, ratio, per_atom_energy):
        pred1, ref1, pred2, ref2 = data
        mm = EnergyForceStressLoss(
            coeffs={
                AtomicDataDict.TOTAL_ENERGY_KEY: ratio[0],
                AtomicDataDict.FORCE_KEY: ratio[1],
                AtomicDataDict.STRESS_KEY: ratio[2],
            },
            per_atom_energy=per_atom_energy,
        )

        for pred, ref in [(pred1, ref1), (pred2, ref2)]:
            metrics_dict = mm(pred, ref)
            assert len(metrics_dict) > 0
            assert isinstance(metrics_dict, dict)
            for key, value in metrics_dict.items():
                assert isinstance(value, torch.Tensor)

            E_MSE = self.compute_MSE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy
            )
            F_MSE = self.compute_MSE(pred, ref, AtomicDataDict.FORCE_KEY)
            S_MSE = self.compute_MSE(pred, ref, AtomicDataDict.STRESS_KEY)

            weighted_sum = (
                ratio[0] * E_MSE + ratio[1] * F_MSE + ratio[2] * S_MSE
            ) / sum(ratio)

            torch.testing.assert_close(
                metrics_dict[
                    "per_atom_energy_mse" if per_atom_energy else "total_energy_mse"
                ],
                E_MSE,
            )
            torch.testing.assert_close(metrics_dict["forces_mse"], F_MSE)
            torch.testing.assert_close(metrics_dict["stress_mse"], S_MSE)
            torch.testing.assert_close(metrics_dict["weighted_sum"], weighted_sum)

    @pytest.mark.parametrize("ratio1", [1])
    @pytest.mark.parametrize("ratio2", [1, 10])
    @pytest.mark.parametrize("ratio3", [1, 10, 100])
    @pytest.mark.parametrize("ratio4", [1, 10])
    @pytest.mark.parametrize("ratio5", [1, 10])
    @pytest.mark.parametrize("ratio6", [1, 10, 100])
    @pytest.mark.parametrize("ratio7", [1, 10])
    @pytest.mark.parametrize("ratio8", [1, 10])
    def test_EnergyForceStressMetrics(
        self, data, ratio1, ratio2, ratio3, ratio4, ratio5, ratio6, ratio7, ratio8
    ):
        pred1, ref1, pred2, ref2 = data
        mm = EnergyForceStressMetrics(
            coeffs={
                "total_energy_rmse": ratio1,
                "forces_rmse": ratio2,
                "stress_rmse": ratio3,
                "total_energy_mae": ratio4,
                "forces_mae": ratio5,
                "stress_mae": ratio6,
                "per_atom_energy_rmse": ratio7,
                "per_atom_energy_mae": ratio8,
            },
        )

        for pred, ref in [(pred1, ref1), (pred2, ref2)]:
            metrics_dict = mm(pred, ref)
            assert len(metrics_dict) > 0
            assert isinstance(metrics_dict, dict)
            for key, value in metrics_dict.items():
                assert isinstance(value, torch.Tensor)
            E_RMSE = self.compute_RMSE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=False
            )
            F_RMSE = self.compute_RMSE(pred, ref, AtomicDataDict.FORCE_KEY)
            S_RMSE = self.compute_RMSE(pred, ref, AtomicDataDict.STRESS_KEY)
            E_MAE = self.compute_MAE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=False
            )
            F_MAE = self.compute_MAE(pred, ref, AtomicDataDict.FORCE_KEY)
            S_MAE = self.compute_MAE(pred, ref, AtomicDataDict.STRESS_KEY)
            per_atom_E_RMSE = self.compute_RMSE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=True
            )
            per_atom_E_MAE = self.compute_MAE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=True
            )
            weighted_sum = (
                ratio1 * E_RMSE
                + ratio2 * F_RMSE
                + ratio3 * S_RMSE
                + ratio4 * E_MAE
                + ratio5 * F_MAE
                + ratio6 * S_MAE
                + ratio7 * per_atom_E_RMSE
                + ratio8 * per_atom_E_MAE
            ) / (ratio1 + ratio2 + ratio3 + ratio4 + ratio5 + ratio6 + ratio7 + ratio8)
            torch.testing.assert_close(metrics_dict["total_energy_rmse"], E_RMSE)
            torch.testing.assert_close(metrics_dict["forces_rmse"], F_RMSE)
            torch.testing.assert_close(metrics_dict["stress_rmse"], S_RMSE)
            torch.testing.assert_close(metrics_dict["total_energy_mae"], E_MAE)
            torch.testing.assert_close(metrics_dict["forces_mae"], F_MAE)
            torch.testing.assert_close(metrics_dict["stress_mae"], S_MAE)
            torch.testing.assert_close(
                metrics_dict["per_atom_energy_rmse"], per_atom_E_RMSE
            )
            torch.testing.assert_close(
                metrics_dict["per_atom_energy_mae"], per_atom_E_MAE
            )
            torch.testing.assert_close(metrics_dict["weighted_sum"], weighted_sum)

    def normalize_energy(self, atomic_dict):
        new_atomic_dict = atomic_dict.copy()
        new_atomic_dict[AtomicDataDict.TOTAL_ENERGY_KEY] = new_atomic_dict[
            AtomicDataDict.TOTAL_ENERGY_KEY
        ] / new_atomic_dict[AtomicDataDict.NUM_NODES_KEY].reshape(
            new_atomic_dict[AtomicDataDict.TOTAL_ENERGY_KEY].shape
        )
        return new_atomic_dict

    def compute_MSE(self, pred, ref, property_key, per_atom_energy=False):
        if per_atom_energy:
            pred = self.normalize_energy(pred)
            ref = self.normalize_energy(ref)
        MSE = torch.mean((pred[property_key] - ref[property_key]) ** 2)
        return MSE

    def compute_MAE(self, pred, ref, property_key, per_atom_energy=False):
        if per_atom_energy:
            pred = self.normalize_energy(pred)
            ref = self.normalize_energy(ref)
        MAE = torch.mean(torch.abs(pred[property_key] - ref[property_key]))
        return MAE

    def compute_RMSE(self, pred, ref, property_key, per_atom_energy=False):
        if per_atom_energy:
            pred = self.normalize_energy(pred)
            ref = self.normalize_energy(ref)
        MSE = torch.mean((pred[property_key] - ref[property_key]) ** 2)
        RMSE = torch.sqrt(MSE)
        return RMSE

    def test_EnergyForceLoss_per_type_forces_coeffs(self, data):
        """Verify the `per_type_forces_coeffs` kwarg is routed to the forces metric only."""
        pred, ref, _, _ = data
        # first half data are type 1
        mse_type1 = torch.square(
            pred[AtomicDataDict.FORCE_KEY][:5] - ref[AtomicDataDict.FORCE_KEY][:5]
        ).mean()
        mse_type0 = torch.square(
            pred[AtomicDataDict.FORCE_KEY][5:] - ref[AtomicDataDict.FORCE_KEY][5:]
        ).mean()
        mm = EnergyForceLoss(
            coeffs={
                AtomicDataDict.TOTAL_ENERGY_KEY: 1.0,
                AtomicDataDict.FORCE_KEY: 1.0,
            },
            per_atom_energy=False,
            per_type_forces_coeffs={"0": 4.0, "1": 1.0},
            type_names=["0", "1"],
        )
        out = mm(pred, ref)
        torch.testing.assert_close(
            out["forces_mse"], (4.0 * mse_type0 + 1.0 * mse_type1) / 5.0
        )
        # per-type breakdowns are also logged
        assert "forces_mse_0" in out and "forces_mse_1" in out

    def test_invalid_EF_coeff_key_triggers_assert(self):
        with pytest.raises(AssertionError):
            EnergyForceMetrics(coeffs={"bad_key": 0.5})

    def test_invalid_EFS_coeff_key_triggers_assert(self):
        with pytest.raises(AssertionError):
            EnergyForceStressMetrics(coeffs={"bad_key": 0.5})

    @pytest.mark.parametrize("per_atom_energy", [True, False])
    def test_EnergyOnlyLoss(self, data, per_atom_energy):
        """Test EnergyOnlyLoss for energy-only training."""
        pred1, ref1, pred2, ref2 = data
        mm = EnergyOnlyLoss(per_atom_energy=per_atom_energy)

        for pred, ref in [(pred1, ref1), (pred2, ref2)]:
            metrics_dict = mm(pred, ref)
            assert len(metrics_dict) > 0
            assert isinstance(metrics_dict, dict)
            for key, value in metrics_dict.items():
                assert isinstance(value, torch.Tensor)

            E_MSE = self.compute_MSE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy
            )
            # coefficient is fixed at 1.0, so weighted_sum should equal the energy MSE
            expected_name = (
                "per_atom_energy_mse" if per_atom_energy else "total_energy_mse"
            )
            torch.testing.assert_close(metrics_dict[expected_name], E_MSE)
            torch.testing.assert_close(metrics_dict["weighted_sum"], E_MSE)

            # should only have energy metric (no forces)
            assert "forces_mse" not in metrics_dict
            assert (
                len([k for k in metrics_dict.keys() if not k.endswith("weighted_sum")])
                == 1
            )

    @pytest.mark.parametrize("ratio1", [1])
    @pytest.mark.parametrize("ratio2", [1, 10])
    @pytest.mark.parametrize("ratio3", [1, 10, 100])
    @pytest.mark.parametrize("ratio4", [1, 10, 100, None])
    def test_EnergyOnlyMetrics(self, data, ratio1, ratio2, ratio3, ratio4):
        """Test EnergyOnlyMetrics for energy-only datasets."""
        pred1, ref1, pred2, ref2 = data
        mm = EnergyOnlyMetrics(
            coeffs={
                "total_energy_rmse": ratio1,
                "per_atom_energy_rmse": ratio2,
                "total_energy_mae": ratio3,
                "per_atom_energy_mae": ratio4,
            },
        )

        for pred, ref in [(pred1, ref1), (pred2, ref2)]:
            metrics_dict = mm(pred, ref)
            assert len(metrics_dict) > 0
            assert isinstance(metrics_dict, dict)
            for key, value in metrics_dict.items():
                assert isinstance(value, torch.Tensor)

            E_RMSE = self.compute_RMSE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=False
            )
            per_atom_E_RMSE = self.compute_RMSE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=True
            )
            E_MAE = self.compute_MAE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=False
            )
            per_atom_E_MAE = self.compute_MAE(
                pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, per_atom_energy=True
            )

            if not ratio4:
                ratio4 = 0
            weighted_sum = (
                ratio1 * E_RMSE
                + ratio2 * per_atom_E_RMSE
                + ratio3 * E_MAE
                + ratio4 * per_atom_E_MAE
            ) / (ratio1 + ratio2 + ratio3 + ratio4)

            torch.testing.assert_close(metrics_dict["total_energy_rmse"], E_RMSE)
            torch.testing.assert_close(
                metrics_dict["per_atom_energy_rmse"], per_atom_E_RMSE
            )
            torch.testing.assert_close(metrics_dict["total_energy_mae"], E_MAE)
            torch.testing.assert_close(
                metrics_dict["per_atom_energy_mae"], per_atom_E_MAE
            )
            torch.testing.assert_close(metrics_dict["weighted_sum"], weighted_sum)

            # should only have energy metrics (no forces)
            assert "forces_rmse" not in metrics_dict
            assert "forces_mae" not in metrics_dict

    def test_invalid_energy_only_coeff_key_triggers_assert(self):
        """Test that invalid coefficient keys trigger assertion errors."""
        with pytest.raises(AssertionError):
            EnergyOnlyMetrics(coeffs={"bad_key": 0.5})

    def test_maximum_absolute_error_with_metrics_manager(self, data):
        """Test MaximumAbsoluteError integration with MetricsManager."""
        pred1, ref1, pred2, ref2 = data
        mm = MetricsManager(
            [
                {
                    "field": AtomicDataDict.TOTAL_ENERGY_KEY,
                    "metric": MaximumAbsoluteError(),
                    "name": "energy_max_ae",
                },
                {
                    "field": AtomicDataDict.FORCE_KEY,
                    "metric": MaximumAbsoluteError(),
                    "name": "force_max_ae",
                },
            ]
        )

        # test step metrics
        metrics_dict = mm(pred1, ref1)
        assert "energy_max_ae" in metrics_dict
        assert "force_max_ae" in metrics_dict
        assert isinstance(metrics_dict["energy_max_ae"], torch.Tensor)
        assert isinstance(metrics_dict["force_max_ae"], torch.Tensor)

        # test second batch
        metrics_dict = mm(pred2, ref2)
        assert "energy_max_ae" in metrics_dict
        assert "force_max_ae" in metrics_dict

        # test epoch metrics (compute)
        epoch_metrics = mm.compute()
        assert "energy_max_ae" in epoch_metrics
        assert "force_max_ae" in epoch_metrics

        # max should be >= individual batch maxes
        energy_max_batch1 = torch.max(
            torch.abs(
                pred1[AtomicDataDict.TOTAL_ENERGY_KEY]
                - ref1[AtomicDataDict.TOTAL_ENERGY_KEY]
            )
        )
        energy_max_batch2 = torch.max(
            torch.abs(
                pred2[AtomicDataDict.TOTAL_ENERGY_KEY]
                - ref2[AtomicDataDict.TOTAL_ENERGY_KEY]
            )
        )
        overall_energy_max = torch.max(
            torch.stack([energy_max_batch1, energy_max_batch2])
        )

        torch.testing.assert_close(epoch_metrics["energy_max_ae"], overall_energy_max)
