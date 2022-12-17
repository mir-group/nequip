import pytest

import tempfile
import functools
import torch

import numpy as np

from e3nn.util.jit import script

from nequip.data import (
    AtomicDataDict,
    AtomicData,
    Collater,
    _GRAPH_FIELDS,
    _NODE_FIELDS,
    _EDGE_FIELDS,
)
from nequip.data.transforms import TypeMapper
from nequip.model import model_from_config
from nequip.nn import GraphModuleMixin
from nequip.utils.test import assert_AtomicData_equivariant


# see https://github.com/pytest-dev/pytest/issues/421#issuecomment-943386533
# to allow external packages to import tests through subclassing
class BaseModelTests:
    @pytest.fixture(scope="class")
    def config(self):
        """Implemented by subclasses.

        Return a tuple of config, out_field
        """
        raise NotImplementedError

    @pytest.fixture(
        scope="class",
        params=(
            [torch.device("cuda"), torch.device("cpu")]
            if torch.cuda.is_available()
            else [torch.device("cpu")]
        ),
    )
    def device(self, request):
        return request.param

    @staticmethod
    def make_model(config, device, initialize: bool = True, deploy: bool = False):
        torch.manual_seed(127)
        np.random.seed(193)
        config = config.copy()
        config.update(
            {
                "num_types": 3,
                "types_names": ["H", "C", "O"],
            }
        )
        model = model_from_config(config, initialize=initialize, deploy=deploy)
        model = model.to(device)
        return model

    @pytest.fixture(scope="class")
    def model(self, config, device):
        config, out_fields = config
        model = self.make_model(config, device=device)
        return model, out_fields

    # == common tests for all models ==
    def test_init(self, model):
        instance, _ = model
        assert isinstance(instance, GraphModuleMixin)

    def test_jit(self, model, atomic_batch, device):
        instance, out_fields = model
        data = AtomicData.to_AtomicDataDict(atomic_batch.to(device=device))
        instance = instance.to(device=device)
        model_script = script(instance)

        for out_field in out_fields:
            assert torch.allclose(
                instance(data)[out_field],
                model_script(data)[out_field],
                atol=1e-6,
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

            for out_field in out_fields:
                assert torch.allclose(
                    model_script(data)[out_field],
                    load_model(load_dat)[out_field],
                    atol=atol,
                )

    def test_forward(self, model, atomic_batch, device):
        instance, out_fields = model
        instance.to(device)
        data = atomic_batch.to(device)
        output = instance(AtomicData.to_AtomicDataDict(data))
        for out_field in out_fields:
            assert out_field in output

    def test_batch(self, model, atomic_batch, device, float_tolerance):
        """Confirm that the results for individual examples are the same regardless of whether they are batched."""
        allclose = functools.partial(torch.allclose, atol=float_tolerance)
        instance, out_fields = model
        instance.to(device)
        data = atomic_batch.to(device)
        data1 = data.get_example(0)
        data2 = data.get_example(1)
        output1 = instance(AtomicData.to_AtomicDataDict(data1))
        output2 = instance(AtomicData.to_AtomicDataDict(data2))
        output = instance(AtomicData.to_AtomicDataDict(data))
        for out_field in out_fields:
            if out_field in _GRAPH_FIELDS:
                assert allclose(
                    output1[out_field],
                    output[out_field][0],
                )
                assert allclose(
                    output2[out_field],
                    output[out_field][1],
                )
            elif out_field in _NODE_FIELDS:
                assert allclose(
                    output1[out_field],
                    output[out_field][output[AtomicDataDict.BATCH_KEY] == 0],
                )
                assert allclose(
                    output2[out_field],
                    output[out_field][output[AtomicDataDict.BATCH_KEY] == 1],
                )
            elif out_field in _EDGE_FIELDS:
                assert allclose(
                    output1[out_field],
                    output[out_field][
                        output[AtomicDataDict.BATCH_KEY][
                            output[AtomicDataDict.EDGE_INDEX_KEY][0]
                        ]
                        == 0
                    ],
                )
                assert allclose(
                    output2[out_field],
                    output[out_field][
                        output[AtomicDataDict.BATCH_KEY][
                            output[AtomicDataDict.EDGE_INDEX_KEY][0]
                        ]
                        == 1
                    ],
                )
            else:
                raise NotImplementedError

    def test_equivariance(self, model, atomic_batch, device):
        instance, out_fields = model
        instance = instance.to(device=device)
        atomic_batch = atomic_batch.to(device=device)
        assert_AtomicData_equivariant(func=instance, data_in=atomic_batch)

    def test_embedding_cutoff(self, model, config, device):
        instance, out_fields = model
        config, out_fields = config
        r_max = config["r_max"]

        # make a synthetic three atom example
        data = AtomicData(
            atom_types=np.random.choice([0, 1, 2], size=3),
            pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            edge_index=np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
        )
        data = data.to(device)
        edge_embed = instance(AtomicData.to_AtomicDataDict(data))
        if AtomicDataDict.EDGE_FEATURES_KEY in edge_embed:
            key = AtomicDataDict.EDGE_FEATURES_KEY
        else:
            key = AtomicDataDict.EDGE_EMBEDDING_KEY
        edge_embed = edge_embed[key]
        data.pos[2, 1] = r_max  # put it past the cutoff
        edge_embed2 = instance(AtomicData.to_AtomicDataDict(data))[key]

        if key == AtomicDataDict.EDGE_EMBEDDING_KEY:
            # we can only check that other edges are unaffected if we know it's an embedding
            # For example, an Allegro edge feature is many body so will be affected
            assert torch.allclose(edge_embed[:2], edge_embed2[:2])
        assert edge_embed[2:].abs().sum() > 1e-6  # some nonzero terms
        assert torch.allclose(edge_embed2[2:], torch.zeros(1, device=device))

        # test gradients
        in_dict = AtomicData.to_AtomicDataDict(data)
        in_dict[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)

        with torch.autograd.set_detect_anomaly(True):
            out = instance(in_dict)

            # is the edge embedding of the cutoff length edge unchanged at the cutoff?
            grads = torch.autograd.grad(
                outputs=out[key][2:].sum(),
                inputs=in_dict[AtomicDataDict.POSITIONS_KEY],
                retain_graph=True,
            )[0]
            assert torch.allclose(grads, torch.zeros(1, device=device))

            if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
                # are the first two atom's energies unaffected by atom at the cutoff?
                grads = torch.autograd.grad(
                    outputs=out[AtomicDataDict.PER_ATOM_ENERGY_KEY][:2].sum(),
                    inputs=in_dict[AtomicDataDict.POSITIONS_KEY],
                )[0]
                print(grads)
                # only care about gradient wrt moved atom
                assert grads.shape == (3, 3)
                assert torch.allclose(grads[2], torch.zeros(1, device=device))


class BaseEnergyModelTests(BaseModelTests):
    def test_large_separation(self, model, config, molecules, device):
        atol = {torch.float32: 1e-4, torch.float64: 1e-10}[torch.get_default_dtype()]
        instance, _ = model
        instance.to(device)
        config, out_fields = config
        r_max = config["r_max"]
        atoms1 = molecules[0].copy()
        atoms2 = molecules[1].copy()
        # translate atoms2 far away
        atoms2.positions += 40.0 + np.random.randn(3)
        atoms_both = atoms1.copy()
        atoms_both.extend(atoms2)
        tm = TypeMapper(chemical_symbols=["H", "C", "O"])
        data1 = tm(AtomicData.from_ase(atoms1, r_max=r_max).to(device=device))
        data2 = tm(AtomicData.from_ase(atoms2, r_max=r_max).to(device=device))
        data_both = tm(AtomicData.from_ase(atoms_both, r_max=r_max).to(device=device))
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
        data_both2 = tm(AtomicData.from_ase(atoms_both2, r_max=r_max).to(device=device))
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

    def test_cross_frame_grad(self, model, device, nequip_dataset):
        c = Collater.for_dataset(nequip_dataset)
        batch = c([nequip_dataset[i] for i in range(len(nequip_dataset))])
        energy_model, out_fields = model
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

    def test_numeric_gradient(self, model, atomic_batch, device):
        model, out_fields = model
        if AtomicDataDict.FORCE_KEY not in out_fields:
            pytest.skip()
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

    def test_partial_forces(self, config, atomic_batch, device, strict_locality):
        config, out_fields = config
        if "ForceOutput" not in config["model_builders"]:
            pytest.skip()
        config = config.copy()
        partial_config = config.copy()
        partial_config["model_builders"] = [
            "PartialForceOutput" if b == "ForceOutput" else b
            for b in partial_config["model_builders"]
        ]
        model = model_from_config(config=config, initialize=True)
        partial_model = model_from_config(config=partial_config, initialize=True)
        model.to(device)
        partial_model.to(device)
        partial_model.load_state_dict(model.state_dict())
        data = atomic_batch.to(device)
        output = model(AtomicData.to_AtomicDataDict(data))
        output_partial = partial_model(AtomicData.to_AtomicDataDict(data))
        # everything should be the same
        # including the
        for k in output:
            assert k != AtomicDataDict.PARTIAL_FORCE_KEY
            assert k in output_partial
            if output[k].is_floating_point():
                assert torch.allclose(
                    output[k],
                    output_partial[k],
                    atol=1e-8 if k == AtomicDataDict.TOTAL_ENERGY_KEY else 1e-6,
                )
            else:
                assert torch.equal(output[k], output_partial[k])
        n_at = data[AtomicDataDict.POSITIONS_KEY].shape[0]
        partial_forces = output_partial[AtomicDataDict.PARTIAL_FORCE_KEY]
        assert partial_forces.shape == (n_at, n_at, 3)
        # confirm that sparsity matches graph topology:
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        adjacency = torch.zeros(
            n_at, n_at, dtype=torch.bool, device=partial_forces.device
        )
        if strict_locality:
            # only adjacent for nonzero deriv to neighbors
            adjacency[edge_index[0], edge_index[1]] = True
            arange = torch.arange(n_at, device=partial_forces.device)
            adjacency[arange, arange] = True  # diagonal is ofc True
        else:
            # technically only adjacent to n-th degree neighbor, but in this tiny test system that is same as all-to-all and easier to program
            adjacency = data[AtomicDataDict.BATCH_KEY].view(-1, 1) == data[
                AtomicDataDict.BATCH_KEY
            ].view(1, -1)
        assert torch.equal(adjacency, torch.any(partial_forces != 0, dim=-1))
