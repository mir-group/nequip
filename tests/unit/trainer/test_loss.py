import pytest
import torch
from nequip.data import AtomicDataDict
from nequip.train import Loss

# all the config to test init
# only the last one will be used to test the loss and mae
dicts = (
    {AtomicDataDict.TOTAL_ENERGY_KEY: (3.0, "MSELoss")},
    {AtomicDataDict.TOTAL_ENERGY_KEY: "MSELoss"},
    [AtomicDataDict.FORCE_KEY, AtomicDataDict.TOTAL_ENERGY_KEY],
    {AtomicDataDict.FORCE_KEY: (1.0, "PerSpeciesMSELoss")},
    {AtomicDataDict.FORCE_KEY: (1.0), "k": (1.0, torch.nn.L1Loss())},
    AtomicDataDict.TOTAL_ENERGY_KEY,
    {
        AtomicDataDict.TOTAL_ENERGY_KEY: (3.0, "L1Loss"),
        AtomicDataDict.FORCE_KEY: (1.0),
        "k": 1.0,
    },
)
nan_dict = {
    AtomicDataDict.TOTAL_ENERGY_KEY: (3.0, "L1Loss", {"ignore_nan": True}),
    AtomicDataDict.FORCE_KEY: (1.0, "MSELoss", {"ignore_nan": True}),
    "k": 1.0,
}


class TestInit:
    @pytest.mark.parametrize("loss", dicts, indirect=True)
    def test_init(self, loss):

        assert len(loss.funcs) == len(loss.coeffs)
        for key, value in loss.coeffs.items():
            assert isinstance(value, torch.Tensor)


class TestLoss:
    @pytest.mark.parametrize("loss", dicts[-2:], indirect=True)
    def test_loss(self, loss, data):

        pred, ref = data

        loss_value = loss(pred, ref)

        loss_value, contrib = loss_value
        assert len(contrib) > 0
        assert isinstance(contrib, dict)
        for key, value in contrib.items():
            assert isinstance(value, torch.Tensor)

        assert isinstance(loss_value, torch.Tensor)

    def test_per_species(self, data):

        pred, ref = data

        config = {AtomicDataDict.FORCE_KEY: (1.0, "PerSpeciesMSELoss")}
        loss = Loss(coeffs=config)

        l, contb = loss(pred, ref)

        # first half data are specie 1
        loss_ref_1 = torch.square(
            pred[AtomicDataDict.FORCE_KEY][:5] - ref[AtomicDataDict.FORCE_KEY][:5]
        ).mean()
        loss_ref_0 = torch.square(
            pred[AtomicDataDict.FORCE_KEY][5:] - ref[AtomicDataDict.FORCE_KEY][5:]
        ).mean()

        assert torch.isclose(
            contb[AtomicDataDict.FORCE_KEY], (loss_ref_0 + loss_ref_1) / 2.0
        )


class TestNaN:
    def test_loss(self, data_w_NaN):

        pred, ref, wo_nan_pred, wo_nan_ref = data_w_NaN

        loss = Loss(coeffs=nan_dict)
        l, contb = loss(pred, ref)
        l_wo_nan, contb_wo_nan = loss(wo_nan_pred, wo_nan_ref)

        assert torch.isclose(l_wo_nan, l)
        for k in contb:
            assert torch.isclose(contb_wo_nan[k], contb[k])

    def test_per_species(self, data_w_NaN):

        pred, ref, wo_nan_pred, wo_nan_ref = data_w_NaN

        config = {
            AtomicDataDict.FORCE_KEY: (1.0, "PerSpeciesMSELoss", {"ignore_nan": True})
        }
        loss = Loss(coeffs=config)

        l, contb = loss(pred, ref)
        l_wo_nan, contb_wo_nan = loss(wo_nan_pred, wo_nan_ref)

        assert torch.isclose(l_wo_nan, l)
        for k in contb:
            assert torch.isclose(contb_wo_nan[k], contb[k])


@pytest.fixture(scope="class")
def loss(request):
    """"""
    d = request.param
    instance = Loss(coeffs=d)
    yield instance


@pytest.fixture(scope="module")
def data(float_tolerance):
    """"""
    pred = {
        AtomicDataDict.FORCE_KEY: torch.rand(10, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((2, 1)),
        "k": torch.rand((2, 1)),
        AtomicDataDict.ATOM_TYPE_KEY: torch.as_tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    }
    ref = {
        AtomicDataDict.FORCE_KEY: torch.rand(10, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((2, 1)),
        "k": torch.rand((2, 1)),
        AtomicDataDict.ATOM_TYPE_KEY: torch.as_tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    }
    yield pred, ref


@pytest.fixture(scope="module")
def data_w_NaN(float_tolerance, data):
    """"""
    _pred, _ref = data

    pred = {k: torch.clone(v) for k, v in _pred.items()}
    ref = {k: torch.clone(v) for k, v in _ref.items()}
    ref[AtomicDataDict.FORCE_KEY][-1] = float("nan")
    ref[AtomicDataDict.FORCE_KEY][0] = float("nan")

    wo_nan_pred = {k: torch.clone(v) for k, v in _pred.items()}
    wo_nan_ref = {k: torch.clone(v) for k, v in _ref.items()}
    wo_nan_ref[AtomicDataDict.FORCE_KEY] = wo_nan_ref[AtomicDataDict.FORCE_KEY][1:-1]
    wo_nan_ref[AtomicDataDict.ATOM_TYPE_KEY] = wo_nan_ref[AtomicDataDict.ATOM_TYPE_KEY][
        1:-1
    ]
    wo_nan_pred[AtomicDataDict.FORCE_KEY] = wo_nan_pred[AtomicDataDict.FORCE_KEY][1:-1]
    wo_nan_pred[AtomicDataDict.ATOM_TYPE_KEY] = wo_nan_pred[
        AtomicDataDict.ATOM_TYPE_KEY
    ][1:-1]

    yield pred, ref, wo_nan_pred, wo_nan_ref
