import inspect
import pytest
import torch
from nequip.data import AtomicDataDict
from nequip.train import Metrics

from .test_loss import data

# all the config to test init
# only the last one will be used to test the loss and mae
metrics_tests = [
    (
        (AtomicDataDict.TOTAL_ENERGY_KEY, "rmse"),
        (
            AtomicDataDict.FORCE_KEY,
            "rmse",
            {"PerSpecies": True, "functional": "L1Loss", "dim": 3},
        ),
        (AtomicDataDict.FORCE_KEY, "mae", {"dim": 3}),
    )
]


class TestInit:
    def test_init(self, metrics):

        for key, value in metrics.funcs.items():
            assert callable(value)


class TestMetrics:
    def test_run(self, metrics, data):

        pred, ref = data

        metrics_dict = metrics(pred, ref)

        for key, value in metrics_dict.items():
            assert isinstance(value, torch.Tensor)


class TestWeight:
    @pytest.mark.parametrize("reduction", [True, False])
    def test_per_specie(self, data, reduction):

        pred, ref = data

        if reduction:
            dim = {"dim": 3, "reduce_dims": 0}
        else:
            dim = {"dim": 3}

        loss = Metrics(
            components=[
                (AtomicDataDict.FORCE_KEY, "rmse", dim),
                (AtomicDataDict.FORCE_KEY, "mae", dim),
            ]
        )

        dim["PerSpecies"] = True
        w_loss = Metrics(
            components=[
                (AtomicDataDict.FORCE_KEY, "rmse", dim),
                (AtomicDataDict.FORCE_KEY, "mae", dim),
            ]
        )

        w_contb = w_loss(pred, ref)
        contb = loss(pred, ref)

        # first half data are specie 1
        loss_ref_0 = torch.square(pred["forces"][5:] - ref["forces"][5:])
        loss_ref_1 = torch.square(pred["forces"][:5] - ref["forces"][:5])
        if reduction:
            loss_ref_1 = torch.sqrt(loss_ref_1.mean())
            loss_ref_0 = torch.sqrt(loss_ref_0.mean())
        else:
            loss_ref_1 = torch.sqrt(loss_ref_1.mean(dim=0))
            loss_ref_0 = torch.sqrt(loss_ref_0.mean(dim=0))

        for c in [w_contb, contb]:
            for key, value in c.items():
                k, reduction = key
                assert k in ["forces"]

        # mae should be the same cause # of type 0 == # of type 1
        assert torch.allclose(
            w_contb[("forces", "mae")].mean(dim=0), contb[("forces", "mae")]
        )
        assert torch.allclose(w_contb[("forces", "rmse")][0], loss_ref_0)
        assert torch.allclose(w_contb[("forces", "rmse")][1], loss_ref_1)


@pytest.fixture(scope="class", params=metrics_tests)
def metrics(request):
    """"""
    coeffs = request.param
    instance = Metrics(components=request.param)
    yield instance


# @pytest.fixture(scope="class")
# def w_loss():
#     """"""
#     instance = Metrics(coeffs=metrics_tests[-1], atomic_weight_on=True)
#     yield instance
