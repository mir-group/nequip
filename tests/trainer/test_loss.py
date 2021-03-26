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


class TestWeight:
    def test_loss(self, data):

        pred, ref = data

        loss = Loss(coeffs=dicts[-1], atomic_weight_on=False)
        w_loss = Loss(coeffs=dicts[-1], atomic_weight_on=True)

        w_l, w_contb = w_loss(pred, ref)
        l, contb = loss(pred, ref)

        assert isinstance(w_l, torch.Tensor)
        assert not torch.isclose(w_l, l)
        assert torch.isclose(
            w_contb[AtomicDataDict.FORCE_KEY], contb[AtomicDataDict.FORCE_KEY]
        )

    def test_per_specie(self, data):

        pred, ref = data

        config = {AtomicDataDict.FORCE_KEY: (1.0, "PerSpeciesMSELoss")}
        loss = Loss(coeffs=config, atomic_weight_on=False)
        w_loss = Loss(coeffs=config, atomic_weight_on=True)

        w_l, w_contb = w_loss(pred, ref)
        l, contb = loss(pred, ref)

        # first half data are specie 1
        # loss_ref_1 = torch.square(pred[AtomicDataDict.FORCE_KEY][:5] - ref[AtomicDataDict.FORCE_KEY][:5]).mean()
        # loss_ref_0 = torch.square(pred[AtomicDataDict.FORCE_KEY][5:] - ref[AtomicDataDict.FORCE_KEY][5:]).mean()

        # since atomic weights are all the same value,
        # the two loss should have the same result
        assert isinstance(w_l, torch.Tensor)
        print(w_l)
        print(l)
        assert torch.isclose(w_l, l)

        for c in [w_contb, contb]:
            for key, value in c.items():
                assert key in [AtomicDataDict.FORCE_KEY]

        assert torch.allclose(
            w_contb[AtomicDataDict.FORCE_KEY], contb[AtomicDataDict.FORCE_KEY]
        )
        # assert torch.isclose(w_contb[1][AtomicDataDict.FORCE_KEY], loss_ref_1)
        # assert torch.isclose(w_contb[0][AtomicDataDict.FORCE_KEY], loss_ref_0)


@pytest.fixture(scope="class")
def loss(request):
    """"""
    d = request.param
    instance = Loss(coeffs=d, atomic_weight_on=False)
    yield instance


@pytest.fixture(scope="class")
def w_loss():
    """"""
    instance = Loss(coeffs=dicts[-1], atomic_weight_on=True)
    yield instance


@pytest.fixture(scope="module")
def data(float_tolerance):
    """"""
    pred = {
        AtomicDataDict.FORCE_KEY: torch.rand(10, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((2, 1)),
        "k": torch.rand((2, 1)),
    }
    ref = {
        AtomicDataDict.FORCE_KEY: torch.rand(10, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((2, 1)),
        "k": torch.rand((2, 1)),
        AtomicDataDict.SPECIES_INDEX_KEY: torch.as_tensor(
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        ),
    }
    ref[AtomicDataDict.WEIGHTS_KEY + AtomicDataDict.FORCE_KEY] = 2 * torch.ones((10, 1))
    ref[AtomicDataDict.WEIGHTS_KEY + AtomicDataDict.TOTAL_ENERGY_KEY] = torch.rand(
        (2, 1)
    )
    yield pred, ref
