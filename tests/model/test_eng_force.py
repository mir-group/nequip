import pytest

import logging
import tempfile
import torch
from os.path import isfile

import numpy as np

from e3nn import o3
from e3nn.util.jit import script

from nequip.data import AtomicDataDict, AtomicData
from nequip.models import EnergyModel, ForceModel
from nequip.nn import GraphModuleMixin, AtomwiseLinear
from nequip.utils.test import assert_AtomicData_equivariant


logging.basicConfig(level=logging.DEBUG)

ALLOWED_SPECIES = [1, 6, 8]
r_max = 3
minimal_config1 = dict(
    allowed_species=ALLOWED_SPECIES,
    irreps_edge_sh="0e + 1o",
    r_max=4,
    feature_irreps_hidden="4x0e + 4x1o",
    resnet=True,
    num_layers=2,
    num_basis=8,
    PolynomialCutoff_p=6,
    nonlinearity_type="norm",
)
minimal_config2 = dict(
    allowed_species=ALLOWED_SPECIES,
    irreps_edge_sh="0e + 1o",
    r_max=4,
    feature_embedding_irreps_out="8x0e + 8x0o + 8x1e + 8x1o",
    irreps_mid_output_block="2x0e",
    feature_irreps_hidden="4x0e + 4x1o",
)
minimal_config3 = dict(
    allowed_species=ALLOWED_SPECIES,
    irreps_edge_sh="0e + 1o",
    r_max=4,
    feature_irreps_hidden="4x0e + 4x1o",
    resnet=True,
    num_layers=2,
    num_basis=8,
    PolynomialCutoff_p=6,
    nonlinearity_type="gate",
)


@pytest.fixture(
    scope="module", params=[minimal_config1, minimal_config2, minimal_config3]
)
def config(request):
    return request.param


@pytest.fixture(
    params=[
        (ForceModel, AtomicDataDict.FORCE_KEY),
        (EnergyModel, AtomicDataDict.TOTAL_ENERGY_KEY),
    ]
)
def model(request, config):
    torch.manual_seed(0)
    np.random.seed(0)
    builder, out_field = request.param
    return builder(**config), out_field


@pytest.fixture(
    scope="module",
    params=(
        [torch.device("cuda"), torch.device("cpu")]
        if torch.cuda.is_available()
        else [torch.device("cpu")]
    ),
)
def device(request):
    return request.param


class TestWorkflow:
    """
    test class methods
    """

    def test_init(self, model):
        instance, _ = model
        assert isinstance(instance, GraphModuleMixin)

    def test_jit(self, model, atomic_batch, device):
        instance, out_field = model
        data = AtomicData.to_AtomicDataDict(atomic_batch.to(device=device))
        instance = instance.to(device=device)
        model_script = script(instance)

        assert torch.allclose(
            instance(data)[out_field], model_script(data)[out_field], atol=1e-7
        )

        # - Try saving, loading in another process, and running -
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save stuff
            model_script.save(tmpdir + "/model.pt")
            torch.save(data, tmpdir + "/dat.pt")
            # Ideally, this would be tested in a subprocess where nequip isn't imported.
            # But CUDA + torch plays very badly with subprocesses here and causes a race condition.
            # So instead we do a slightly less complete test, loading the saved model here in the original process:
            load_model = torch.jit.load(tmpdir + "/model.pt")
            load_dat = torch.load(tmpdir + "/dat.pt")

            atol = {
                # tight, but not that tight, since GPU nondet has to pass
                torch.float32: 1e-6,
                torch.float64: 1e-10,
            }[torch.get_default_dtype()]

            assert torch.allclose(
                model_script(data)[out_field],
                load_model(load_dat)[out_field],
                atol=atol,
            )

    def test_submods(self):
        model = EnergyModel(**minimal_config2)
        assert isinstance(model.feature_embedding, AtomwiseLinear)
        true_irreps = o3.Irreps(minimal_config2["feature_embedding_irreps_out"])
        assert (
            model.feature_embedding.irreps_out[model.feature_embedding.out_field]
            == true_irreps
        )
        # Make sure it propagates
        assert (
            model.layer0_convnet.irreps_in[model.feature_embedding.out_field]
            == true_irreps
        )

    def test_forward(self, model, atomic_batch, device):
        instance, out_field = model
        instance.to(device)
        data = atomic_batch.to(device)
        output = instance(AtomicData.to_AtomicDataDict(data))
        assert out_field in output

    def test_saveload(self, model):
        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
            instance, _ = model
            torch.save(instance, tmp.name)
            assert isfile(tmp.name)

            new_model = torch.load(tmp.name)
            assert isinstance(new_model, type(instance))


class TestGradient:
    def test_numeric_gradient(self, config, atomic_batch, device, float_tolerance):
        model = ForceModel(**config)
        model.to(device)
        data = atomic_batch.to(device)
        output = model(AtomicData.to_AtomicDataDict(data))

        forces = output[AtomicDataDict.FORCE_KEY]

        epsilon = torch.as_tensor(1e-3)
        epsilon2 = torch.as_tensor(2e-3)
        iatom = 1
        for idir in range(3):
            pos = data[AtomicDataDict.POSITIONS_KEY][iatom, idir]
            data[AtomicDataDict.POSITIONS_KEY][iatom, idir] = pos + epsilon
            output = model(AtomicData.to_AtomicDataDict(data.to(device)))
            e_plus = output[AtomicDataDict.TOTAL_ENERGY_KEY].sum()

            data[AtomicDataDict.POSITIONS_KEY][iatom, idir] -= epsilon2
            output = model(AtomicData.to_AtomicDataDict(data.to(device)))
            e_minus = output[AtomicDataDict.TOTAL_ENERGY_KEY].sum()

            numeric = -(e_plus - e_minus) / epsilon2
            analytical = forces[iatom, idir]
            print(numeric.item(), analytical.item())
            assert torch.isclose(numeric, analytical, atol=2e-2) or torch.isclose(
                numeric, analytical, rtol=5e-2
            )


class TestAutoGradient:
    def test_cross_frame_grad(self, config, nequip_dataset):
        batch = nequip_dataset.data
        device = "cpu"
        energy_model = EnergyModel(**config)
        energy_model.to(device)
        data = AtomicData.to_AtomicDataDict(batch.to(device))
        data[AtomicDataDict.POSITIONS_KEY].requires_grad = True

        output = energy_model(data)
        grads = torch.autograd.grad(
            outputs=output[AtomicDataDict.TOTAL_ENERGY_KEY][-1],
            inputs=data[AtomicDataDict.POSITIONS_KEY],
            allow_unused=True,
        )[0]

        last_frame_n_atom = batch.ptr[-1] - batch.ptr[-2]

        in_frame_grad = grads[-last_frame_n_atom:]
        cross_frame_grad = grads[:-last_frame_n_atom]

        assert cross_frame_grad.abs().max().item() == 0
        assert in_frame_grad.abs().max().item() > 0


class TestEquivariance:
    def test_forward(self, model, atomic_batch, device):
        instance, out_field = model
        instance = instance.to(device=device)
        atomic_batch = atomic_batch.to(device=device)
        assert_AtomicData_equivariant(func=instance, data_in=atomic_batch)


class TestCutoff:
    def test_large_separation(self, model, config, molecules):
        atol = {torch.float32: 1e-6, torch.float64: 1e-10}[torch.get_default_dtype()]
        instance, _ = model
        r_max = config["r_max"]
        atoms1 = molecules[0].copy()
        atoms2 = molecules[1].copy()
        # translate atoms2 far away
        atoms2.positions += 40.0 + np.random.randn(3)
        atoms_both = atoms1.copy()
        atoms_both.extend(atoms2)
        data1 = AtomicData.from_ase(atoms1, r_max=r_max)
        data2 = AtomicData.from_ase(atoms2, r_max=r_max)
        data_both = AtomicData.from_ase(atoms_both, r_max=r_max)
        assert (
            data_both[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
            == data1[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
            + data2[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        )

        out1 = instance(AtomicData.to_AtomicDataDict(data1))
        out2 = instance(AtomicData.to_AtomicDataDict(data2))
        out_both = instance(AtomicData.to_AtomicDataDict(data_both))

        assert torch.allclose(
            out1[AtomicDataDict.TOTAL_ENERGY_KEY]
            + out2[AtomicDataDict.TOTAL_ENERGY_KEY],
            out_both[AtomicDataDict.TOTAL_ENERGY_KEY],
            atol=atol,
        )

        atoms_both2 = atoms1.copy()
        atoms3 = atoms2.copy()
        atoms3.positions += np.random.randn(3)
        atoms_both2.extend(atoms3)
        data_both2 = AtomicData.from_ase(atoms_both2, r_max=r_max)
        out_both2 = instance(AtomicData.to_AtomicDataDict(data_both2))
        assert torch.allclose(
            out_both2[AtomicDataDict.TOTAL_ENERGY_KEY],
            out_both[AtomicDataDict.TOTAL_ENERGY_KEY],
            atol=atol,
        )
        assert torch.allclose(
            out_both2[AtomicDataDict.PER_ATOM_ENERGY_KEY],
            out_both[AtomicDataDict.PER_ATOM_ENERGY_KEY],
            atol=atol,
        )

    def test_embedding_cutoff(self, config):
        instance = EnergyModel(**config)
        r_max = config["r_max"]

        # make a synthetic three atom example
        data = AtomicData(
            atomic_numbers=np.random.choice(ALLOWED_SPECIES, size=3),
            pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            edge_index=np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
        )
        edge_embed = instance(AtomicData.to_AtomicDataDict(data))[
            AtomicDataDict.EDGE_EMBEDDING_KEY
        ]
        data.pos[2, 1] = r_max  # put it past the cutoff
        edge_embed2 = instance(AtomicData.to_AtomicDataDict(data))[
            AtomicDataDict.EDGE_EMBEDDING_KEY
        ]

        assert torch.allclose(edge_embed[:2], edge_embed2[:2])
        assert edge_embed[2:].abs().sum() > 1e-6  # some nonzero terms
        assert torch.allclose(edge_embed2[2:], torch.zeros(1))

        # test gradients
        in_dict = AtomicData.to_AtomicDataDict(data)
        in_dict[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)

        with torch.autograd.set_detect_anomaly(True):
            out = instance(in_dict)

            # is the edge embedding of the cutoff length edge unchanged at the cutoff?
            grads = torch.autograd.grad(
                outputs=out[AtomicDataDict.EDGE_EMBEDDING_KEY][2:].sum(),
                inputs=in_dict[AtomicDataDict.POSITIONS_KEY],
                retain_graph=True,
            )[0]
            assert torch.allclose(grads, torch.zeros(1))

            # are the first two atom's energies unaffected by atom at the cutoff?
            grads = torch.autograd.grad(
                outputs=out[AtomicDataDict.PER_ATOM_ENERGY_KEY][:2].sum(),
                inputs=in_dict[AtomicDataDict.POSITIONS_KEY],
            )[0]
            print(grads)
            # only care about gradient wrt moved atom
            assert grads.shape == (3, 3)
            assert torch.allclose(grads[2], torch.zeros(1))
