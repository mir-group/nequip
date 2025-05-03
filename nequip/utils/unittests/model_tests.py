# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import pytest

import tempfile
import functools
import torch

import numpy as np

from e3nn.util.jit import script

from nequip.data import (
    from_dict,
    from_ase,
    compute_neighborlist_,
    AtomicDataDict,
    _GRAPH_FIELDS,
    _NODE_FIELDS,
    _EDGE_FIELDS,
)
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper
from nequip.nn import GraphModuleMixin, ForceStressOutput, PartialForceOutput
from nequip.utils import dtype_to_name, find_first_of_type
from nequip.utils.versions import _TORCH_GE_2_6
from nequip.utils.test import (
    assert_AtomicData_equivariant,
    FLOAT_TOLERANCE,
    override_irreps_debug,
)

from hydra.utils import get_method


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
        params=(["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]),
    )
    def device(self, request):
        return request.param

    @staticmethod
    def make_model(model_config, device):
        config = model_config.copy()
        model_builder = get_method(config.pop("_target_"))
        model = model_builder(**config)
        model = model.to(device)
        # test if possible to print model
        print(model)
        return model

    @pytest.fixture(scope="class")
    def model(self, config, device, model_dtype):
        # === sanity check contracts with subclasses ===
        assert (
            "model_dtype" not in config
        ), "model test subclasses should not include `model_dtype` in the configs -- the test class will handle it (looping over `float32` and `float64`)"
        assert (
            "compile_mode" not in config
        ), "model test subclasses should not include `compile_mode` in the configs -- the test class will handle it (looping over TorchScript and `torch.compile` modes)"
        config = config.copy()
        config.update({"model_dtype": model_dtype})
        model = self.make_model(config, device=device)
        out_fields = model.irreps_out.keys()
        # we return the config with the correct `model_dtype`
        return model, config, out_fields

    @pytest.fixture(scope="class")
    def partial_model(self, model, device):
        _, config, _ = model
        aux_model = self.make_model(config.copy(), device=device)
        module = find_first_of_type(aux_model, ForceStressOutput)
        # skip test if force/stress module not found
        if module is None:
            pytest.skip("force/stress module not found")
        # replace force/stress module with partial force module
        aux_model.model = PartialForceOutput(module.func)
        out_fields = aux_model.model.irreps_out.keys()

        # partial_model does not output config. Please use `model` fixture for config
        return aux_model, out_fields

    @pytest.fixture(scope="class")
    def model_test_data(self, config, atomic_batch, device):
        # clone the data to avoid mutating the original batch
        # cpu because we need to reconstruct the neighborlist
        atomic_batch = {
            k: v.clone().detach().to("cpu") for k, v in atomic_batch.items()
        }
        # reset neighborlist
        atomic_batch = compute_neighborlist_(atomic_batch, r_max=config["r_max"])
        # return the data in the device for testing
        return AtomicDataDict.to_(atomic_batch, device)

    # == common tests for all models ==
    def test_init(self, model):
        instance, _, _ = model
        assert isinstance(instance, GraphModuleMixin)

    def test_jit(self, model, atomic_batch, device):
        instance, _, out_fields = model
        data = AtomicDataDict.to_(atomic_batch, device)
        model_script = script(instance)

        atol = {
            # tight, but not that tight, since GPU nondet has to pass
            # plus model insides are still float32 with global dtype float64 in the tests
            torch.float32: 5e-5,
            torch.float64: 5e-7,
        }[instance.model_dtype]

        out_instance = instance(data.copy())
        out_script = model_script(data.copy())

        for out_field in out_fields:
            assert torch.allclose(
                out_instance[out_field],
                out_script[out_field],
                atol=atol,
            ), f"JIT didn't repro non-JIT on field {out_field} with max error {(out_instance[out_field] - out_script[out_field]).abs().max().item()}"

        # - Try saving, loading in another process, and running -
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save stuff
            model_script.save(tmpdir + "/model.pt")
            torch.save(data, tmpdir + "/dat.pt")
            # Ideally, this would be tested in a subprocess where nequip isn't imported.
            # But CUDA + torch plays very badly with subprocesses here and causes a race condition.
            # So instead we do a slightly less complete test, loading the saved model here in the original process:
            load_model = torch.jit.load(tmpdir + "/model.pt")
            load_dat = torch.load(tmpdir + "/dat.pt", weights_only=False)

            out_script = model_script(data.copy())
            out_load = load_model(load_dat.copy())

            for out_field in out_fields:
                assert torch.allclose(
                    out_script[out_field],
                    out_load[out_field],
                    atol=atol,
                ), f"JIT didn't repro save-and-loaded JIT on field {out_field} with max error {(out_script[out_field] - out_load[out_field]).abs().max().item()}"

    @override_irreps_debug(False)
    def test_compile(self, model, model_test_data, device):
        """
        Test train-time compilation, i.e. `make_fx` -> `export` -> `AOTAutograd` correctness.
        """
        if not _TORCH_GE_2_6:
            pytest.skip("PT2 compile tests skipped for torch < 2.6")

        # TODO: sort out the CPU compilation issues
        if device == "cpu":
            pytest.skip(
                "compile tests are skipped for CPU as there are known compilation bugs for both NequIP and Allegro models on CPU"
            )

        instance, config, _ = model
        # get tolerance based on model_dtype
        tol = {
            torch.float32: 5e-5,
            torch.float64: 1e-12,
        }[instance.model_dtype]

        # make compiled model
        config = config.copy()
        config["compile_mode"] = "compile"
        compile_model = self.make_model(config, device=device)

        compile_out = compile_model(model_test_data.copy())  # shallow copy
        assert compile_model._compiled_model, "compilation unsuccessful"
        # if compilation was successful, internal checks would have ensured consistency of base and compiled model predictions
        # here, we check that backward pass of the model works for training

        # test backwards pass if there are trainable weights
        if any([p.requires_grad for p in compile_model.parameters()]):
            compile_loss = compile_out[AtomicDataDict.TOTAL_ENERGY_KEY].square().sum()
            compile_loss.backward()

            # compute base model predictions
            out = instance(model_test_data.copy())  # shallow copy
            loss = out[AtomicDataDict.TOTAL_ENERGY_KEY].square().sum()
            loss.backward()
            compile_params = dict(compile_model.named_parameters())
            for k, v in instance.named_parameters():
                err = torch.max(torch.abs(v.grad - compile_params[k].grad))
                assert torch.allclose(
                    v.grad, compile_params[k].grad, atol=tol, rtol=tol
                ), err

    @pytest.mark.skipif(
        not _TORCH_GE_2_6, reason="PT2 compile tests skipped for torch < 2.6"
    )
    @override_irreps_debug(False)
    def test_aot_export(self, model, model_test_data, device):
        """
        Tests inference-time compilation, i.e. that `make_fx` -> `export` -> `AOTExport` correctness.

        For now, we only test the unbatched case, i.e. a single frame, relevant for ase, pair-nequip, and pair-allegro.
        """
        from nequip.model.utils import (
            override_model_compile_mode,
            _COMPILE_TIME_AOTINDUCTOR_KEY,
        )
        from nequip.utils.compile import prepare_model_for_compile
        from nequip.utils.aot import aot_export_model
        from nequip.scripts._compile_utils import PAIR_NEQUIP_INPUTS, LMP_OUTPUTS

        # get a single frame, and drop batch fields to take optimized path
        export_data = AtomicDataDict.frame_from_batched(model_test_data.copy(), 0)
        export_data.pop(AtomicDataDict.BATCH_KEY)
        export_data.pop(AtomicDataDict.NUM_NODES_KEY)

        _, config, _ = model
        with override_model_compile_mode(compile_mode=_COMPILE_TIME_AOTINDUCTOR_KEY):
            model = self.make_model(config, device=device)
        model = prepare_model_for_compile(model, device)
        # export model
        batch_map = {
            "graph": torch.export.Dim.STATIC,
            "node": torch.export.dynamic_shapes.Dim("num_nodes", min=2, max=torch.inf),
            "edge": torch.export.dynamic_shapes.Dim("num_edges", min=2, max=torch.inf),
        }
        # sanity checking done when exporting
        with tempfile.TemporaryDirectory() as tmpdir:
            _ = aot_export_model(
                model=model,
                device=device,
                input_fields=PAIR_NEQUIP_INPUTS,
                output_fields=LMP_OUTPUTS,
                data=export_data,
                batch_map=batch_map,
                output_path=tmpdir + "/export_test.pt2",
                inductor_configs={},
                seed=0,
            )

    def test_forward(self, model, model_test_data):
        instance, _, out_fields = model
        output = instance(model_test_data)
        for out_field in out_fields:
            assert out_field in output

    def test_wrapped_unwrapped(self, model, device, Cu_bulk):
        atoms, data_orig = Cu_bulk
        instance, _, out_fields = model
        data = from_ase(atoms)
        data = compute_neighborlist_(data, r_max=3.5)
        data[AtomicDataDict.ATOM_TYPE_KEY] = data_orig[AtomicDataDict.ATOM_TYPE_KEY]
        data = AtomicDataDict.to_(data, device)
        out_ref = instance(data)

        # now put things in other periodic images
        rng = torch.Generator(device=device).manual_seed(12345)
        # try a few different shifts
        for _ in range(3):
            cell_shifts = torch.randint(
                -5,
                5,
                (len(atoms), 3),
                device=device,
                dtype=data[AtomicDataDict.POSITIONS_KEY].dtype,
                generator=rng,
            )
            shifts = torch.einsum(
                "zi,ix->zx", cell_shifts, data[AtomicDataDict.CELL_KEY].reshape((3, 3))
            )
            atoms2 = atoms.copy()
            atoms2.positions += shifts.detach().cpu().numpy()
            # must recompute the neighborlist for this, since the edge_cell_shifts changed
            data2 = from_ase(atoms2)
            data2 = compute_neighborlist_(data2, r_max=3.5)
            data2[AtomicDataDict.ATOM_TYPE_KEY] = data[AtomicDataDict.ATOM_TYPE_KEY]
            data2 = AtomicDataDict.to_(data2, device)
            assert torch.equal(
                data[AtomicDataDict.EDGE_INDEX_KEY],
                data2[AtomicDataDict.EDGE_INDEX_KEY],
            )
            tmp = (
                data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
                + cell_shifts[data[AtomicDataDict.EDGE_INDEX_KEY][0]]
                - cell_shifts[data[AtomicDataDict.EDGE_INDEX_KEY][1]]
            )
            assert torch.equal(
                tmp,
                data2[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
            )
            out_unwrapped = instance(from_dict(data2))
            tolerance = FLOAT_TOLERANCE[dtype_to_name(instance.model_dtype)]
            for out_field in out_fields:
                # not important for the purposes of this test
                if out_field in [
                    AtomicDataDict.POSITIONS_KEY,
                    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
                ]:
                    continue
                assert torch.allclose(
                    out_ref[out_field], out_unwrapped[out_field], atol=tolerance
                ), f'failed for key "{out_field}" with max absolute diff {torch.abs(out_ref[out_field] - out_unwrapped[out_field]).max().item():.5g} (tol={tolerance:.5g})'

    def test_batch(self, model, model_test_data):
        """Confirm that the results for individual examples are the same regardless of whether they are batched."""
        instance, _, out_fields = model

        tolerance = FLOAT_TOLERANCE[dtype_to_name(instance.model_dtype)]
        allclose = functools.partial(torch.allclose, atol=tolerance)
        data1 = AtomicDataDict.frame_from_batched(model_test_data, 0)
        data2 = AtomicDataDict.frame_from_batched(model_test_data, 1)
        output1 = instance(data1)
        output2 = instance(data2)
        output = instance(model_test_data)
        for out_field in out_fields:
            # to ignore
            if out_field in [
                AtomicDataDict.EDGE_INDEX_KEY,
                AtomicDataDict.BATCH_KEY,
                AtomicDataDict.EDGE_TYPE_KEY,
            ]:
                continue
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
                ), f"failed for {out_field}"
                assert allclose(
                    output2[out_field],
                    output[out_field][output[AtomicDataDict.BATCH_KEY] == 1],
                ), f"failed for {out_field}"
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
                raise NotImplementedError(
                    f"Found unregistered `out_field` = {out_field}"
                )

    def test_equivariance(self, model, model_test_data, device):
        instance, _, _ = model
        instance = instance.to(device=device)

        assert_AtomicData_equivariant(
            func=instance,
            data_in=model_test_data,
            e3_tolerance={torch.float32: 1e-3, torch.float64: 1e-8}[
                instance.model_dtype
            ],
        )

    def test_embedding_cutoff(self, model, device):
        instance, config, _ = model
        r_max = config["r_max"]

        # make a synthetic three atom example
        data = {
            "atom_types": np.random.choice([0, 1, 2], size=3),
            "pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "edge_index": np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
        }
        data = AtomicDataDict.to_(from_dict(data), device)
        edge_embed = instance(data)
        if AtomicDataDict.EDGE_FEATURES_KEY in edge_embed:
            key = AtomicDataDict.EDGE_FEATURES_KEY
        elif AtomicDataDict.EDGE_EMBEDDING_KEY in edge_embed:
            key = AtomicDataDict.EDGE_EMBEDDING_KEY
        else:
            pytest.skip()
        edge_embed = edge_embed[key]
        data[AtomicDataDict.POSITIONS_KEY][2, 1] = r_max  # put it past the cutoff
        edge_embed2 = instance(from_dict(data))[key]

        if key == AtomicDataDict.EDGE_EMBEDDING_KEY:
            # we can only check that other edges are unaffected if we know it's an embedding
            # For example, an Allegro edge feature is many body so will be affected
            assert torch.allclose(edge_embed[:2], edge_embed2[:2])
        assert edge_embed[2:].abs().sum() > 1e-6  # some nonzero terms
        assert torch.allclose(
            edge_embed2[2:], torch.zeros(1, device=device, dtype=edge_embed2.dtype)
        )

        # test gradients
        in_dict = from_dict(data)
        in_dict[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)

        with torch.autograd.set_detect_anomaly(True):
            out = instance(in_dict)

            # is the edge embedding of the cutoff length edge unchanged at the cutoff?
            grads = torch.autograd.grad(
                outputs=out[key][2:].sum(),
                inputs=in_dict[AtomicDataDict.POSITIONS_KEY],
                retain_graph=True,
            )[0]
            assert torch.allclose(
                grads, torch.zeros(1, device=device, dtype=grads.dtype)
            )

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
    def test_large_separation(self, model, molecules, device):
        instance, config, _ = model
        atol = {torch.float32: 1e-4, torch.float64: 1e-10}[instance.model_dtype]
        r_max = config["r_max"]
        atoms1 = molecules[0].copy()
        atoms2 = molecules[1].copy()
        # translate atoms2 far away
        atoms2.positions += 40.0 + np.random.randn(3)
        atoms_both = atoms1.copy()
        atoms_both.extend(atoms2)
        tm = ChemicalSpeciesToAtomTypeMapper(
            chemical_symbols=["H", "C", "O"],
        )

        data1 = AtomicDataDict.to_(
            tm(compute_neighborlist_(from_ase(atoms1), r_max=r_max)),
            device,
        )

        data2 = AtomicDataDict.to_(
            tm(compute_neighborlist_(from_ase(atoms2), r_max=r_max)),
            device,
        )

        data_both = AtomicDataDict.to_(
            tm(compute_neighborlist_(from_ase(atoms_both), r_max=r_max)),
            device,
        )
        assert (
            data_both[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
            == data1[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
            + data2[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        )

        out1 = instance(from_dict(data1))
        out2 = instance(from_dict(data2))
        out_both = instance(from_dict(data_both))

        assert torch.allclose(
            out1[AtomicDataDict.TOTAL_ENERGY_KEY]
            + out2[AtomicDataDict.TOTAL_ENERGY_KEY],
            out_both[AtomicDataDict.TOTAL_ENERGY_KEY],
            atol=atol,
        )
        if AtomicDataDict.FORCE_KEY in out1:
            # check forces if it's a force model
            assert torch.allclose(
                torch.cat(
                    (out1[AtomicDataDict.FORCE_KEY], out2[AtomicDataDict.FORCE_KEY]),
                    dim=0,
                ),
                out_both[AtomicDataDict.FORCE_KEY],
                atol=atol,
            )

        atoms_both2 = atoms1.copy()
        atoms3 = atoms2.copy()
        atoms3.positions += np.random.randn(3)
        atoms_both2.extend(atoms3)

        data_both2 = AtomicDataDict.to_(
            tm(compute_neighborlist_(from_ase(atoms_both2), r_max=r_max)),
            device,
        )

        out_both2 = instance(data_both2)
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
        batch = AtomicDataDict.batched_from_list(
            [nequip_dataset[i] for i in range(len(nequip_dataset))]
        )
        energy_model, _, _ = model
        data = AtomicDataDict.to_(batch, device)
        data[AtomicDataDict.POSITIONS_KEY].requires_grad = True

        output = energy_model(data)
        grads = torch.autograd.grad(
            outputs=output[AtomicDataDict.TOTAL_ENERGY_KEY][-1],
            inputs=data[AtomicDataDict.POSITIONS_KEY],
            allow_unused=True,
        )[0]

        last_frame_n_atom = batch[AtomicDataDict.NUM_NODES_KEY][-1]

        in_frame_grad = grads[-last_frame_n_atom:]
        cross_frame_grad = grads[:-last_frame_n_atom]

        assert cross_frame_grad.abs().max().item() == 0
        assert in_frame_grad.abs().max().item() > 0

    def test_numeric_gradient(self, model, atomic_batch, device):
        """
        Tests the ForceStressOutput model by comparing numerical gradients of the forces to the analytical gradients.
        """
        model, _, out_fields = model
        # proceed with tests only if forces are available
        if AtomicDataDict.FORCE_KEY in out_fields:
            # physical predictions (energy, forces, etc) will be converted to default_dtype (float64) before comparing
            data = AtomicDataDict.to_(atomic_batch, device)
            output = model(data)
            forces = output[AtomicDataDict.FORCE_KEY]
            epsilon = 1e-3

            # Compute numerical gradients for each atom and direction and compare to analytical gradients
            for iatom in range(len(data[AtomicDataDict.POSITIONS_KEY])):
                for idir in range(3):
                    # Shift `iatom` an `epsilon` in the `idir` direction
                    pos = data[AtomicDataDict.POSITIONS_KEY][iatom, idir]
                    data[AtomicDataDict.POSITIONS_KEY][iatom, idir] = pos + epsilon
                    output = model(data)
                    e_plus = (
                        output[AtomicDataDict.TOTAL_ENERGY_KEY]
                        .sum()
                        .to(torch.get_default_dtype())
                    )

                    # Shift `iatom` an `epsilon` in the negative `idir` direction
                    data[AtomicDataDict.POSITIONS_KEY][iatom, idir] -= epsilon * 2
                    output = model(data)
                    e_minus = (
                        output[AtomicDataDict.TOTAL_ENERGY_KEY]
                        .sum()
                        .to(torch.get_default_dtype())
                    )

                    # Symmetric difference to get the partial forces to all the atoms
                    numeric = -(e_plus - e_minus) / (epsilon * 2)
                    analytical = forces[iatom, idir].to(torch.get_default_dtype())

                    assert torch.isclose(
                        numeric, analytical, atol=2e-2
                    ) or torch.isclose(
                        numeric, analytical, rtol=5e-3
                    ), f"numeric: {numeric.item()}, analytical: {analytical.item()}"

                    # Reset the position
                    data[AtomicDataDict.POSITIONS_KEY][iatom, idir] += epsilon

    def test_partial_forces(
        self, model, partial_model, atomic_batch, device, strict_locality
    ):
        model, _, _ = model
        partial_model, _ = partial_model

        data = AtomicDataDict.to_(atomic_batch, device)
        output = model(data)
        output_partial = partial_model(from_dict(data))
        # most data tensors should be the same
        for k in output:
            assert k != AtomicDataDict.PARTIAL_FORCE_KEY
            if k in [AtomicDataDict.STRESS_KEY, AtomicDataDict.VIRIAL_KEY]:
                continue
            assert k in output_partial, k
            if output[k].is_floating_point():
                assert torch.allclose(
                    output[k],
                    output_partial[k],
                    atol=(
                        1e-8
                        if k == AtomicDataDict.TOTAL_ENERGY_KEY
                        and model.model_dtype == torch.float64
                        else 1e-5
                    ),
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
        # for non-adjacent atoms, all partial forces must be zero
        assert torch.all(partial_forces[~adjacency] == 0)

    def test_numeric_gradient_partial(self, partial_model, atomic_batch, device):
        """
        Tests the PartialForceOutput model by comparing numerical gradients of the partial forces to the analytical gradients.
        """

        partial_model, out_fields = partial_model
        # proceed with tests only is partial forces are available
        if AtomicDataDict.PARTIAL_FORCE_KEY in out_fields:
            # physical predictions (energy, forces, etc) will be converted to default_dtype (float64) before comparing
            data = AtomicDataDict.to_(atomic_batch, device)
            output = partial_model(data)
            partial_forces = output[AtomicDataDict.PARTIAL_FORCE_KEY]
            epsilon = 1e-3

            # Compute numerical gradients for each atom and direction and compare to analytical gradients
            for iatom in range(len(data[AtomicDataDict.POSITIONS_KEY])):
                for idir in range(3):
                    # Shift `iatom` an `epsilon` in the `idir` direction
                    pos = data[AtomicDataDict.POSITIONS_KEY][iatom, idir]
                    data[AtomicDataDict.POSITIONS_KEY][iatom, idir] = pos + epsilon
                    output = partial_model(data)
                    e_plus = (
                        output[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                        .to(torch.get_default_dtype())
                        .flatten()
                    )

                    # Shift `iatom` an `epsilon` in the negative `idir` direction
                    data[AtomicDataDict.POSITIONS_KEY][iatom, idir] -= epsilon * 2
                    output = partial_model(data)
                    e_minus = (
                        output[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                        .to(torch.get_default_dtype())
                        .flatten()
                    )

                    # Symmetric difference
                    numeric = -(e_plus - e_minus) / (epsilon * 2)
                    analytical = partial_forces[:, iatom, idir].to(
                        torch.get_default_dtype()
                    )

                    assert torch.allclose(
                        numeric, analytical, atol=2e-2
                    ) or torch.allclose(
                        numeric, analytical, rtol=5e-2
                    ), f"numeric: {numeric.item()}, analytical: {analytical.item()}"

                    # Reset the position
                    data[AtomicDataDict.POSITIONS_KEY][iatom, idir] += epsilon

    @pytest.fixture(scope="class")
    def pair_force(self, model, partial_model, device):
        """
        Helper function to calculate forces/partial forces between a pair of atoms of specified types and separation distance.
        """
        # Initialize the models
        instance, _, out_fields = model
        partial_instance, out_fields_partial = partial_model

        # Skip the test if the models do not have the necessary output fields
        if AtomicDataDict.FORCE_KEY not in out_fields:
            pytest.skip()
        if AtomicDataDict.PARTIAL_FORCE_KEY not in out_fields_partial:
            pytest.skip()

        def wrapped(node_index, neighbor_index, dist, neighborlist_dist, partial=False):
            # Create a pair of atoms of the specified types, separated by the specified distance
            data = {
                "atom_types": np.array([node_index, neighbor_index]),
                "pos": np.array([[0.0, 0.0, 0.0], [dist, 0.0, 0.0]]),
            }

            # Create a neighborlist with the specified neighbor distance
            data = compute_neighborlist_(from_dict(data), r_max=neighborlist_dist)

            # Calculate forces/partial forces
            if partial:
                out = partial_instance(AtomicDataDict.to_(data, device))
                return out[AtomicDataDict.PARTIAL_FORCE_KEY]
            else:
                out = instance(AtomicDataDict.to_(data, device))
                return out[AtomicDataDict.FORCE_KEY]

        return wrapped

    def test_force_smoothness(self, model, device, pair_force):
        _, config, _ = model
        r_max = config["r_max"]
        type_names = config["type_names"]
        num_types = len(type_names)

        # don't test if model is using per-edge-type-cutoffs
        if "per_edge_type_cutoff" not in config:
            for node_idx in range(num_types):
                for nbor_idx in range(num_types):

                    # Control group: force is non-zero within the cutoff radius
                    forces = pair_force(node_idx, nbor_idx, 0.5 * r_max, 1.5 * r_max)
                    # expect some nonzero terms on the two connected atoms
                    # NOTE: sometimes it can be zero if the model has so little features such that the nonlinearity causes the activation to be ~0
                    assert forces.abs().sum() > 1e-4, f"{forces=}"

                    # For Test 1 and 2:
                    # No need to enforce `strictly_local`. Message passing models such as NequiIP should not receive information from beyond the cutoff radius.
                    # In fact, checking that message-passing models still have zero force at the cutoff radius is a good test of locality.

                    # Test 1: force is zero at the cutoff radius
                    forces = pair_force(node_idx, nbor_idx, r_max, 1.5 * r_max)
                    assert torch.allclose(
                        forces,
                        torch.zeros_like(forces, device=device, dtype=forces.dtype),
                    ), f"{forces=}"

                    # Test 2: force is zero outside of the cutoff radius
                    forces = pair_force(node_idx, nbor_idx, 1.1 * r_max, 1.5 * r_max)
                    assert torch.allclose(
                        forces,
                        torch.zeros_like(forces, device=device, dtype=forces.dtype),
                    ), f"{forces=}"

    def test_partial_force_smoothness(self, model, device, pair_force):
        # NOTE: This test is designed for models that have a variable cutoff radius, though it still applicable with
        # fixed cutoff models. This works on the assumption that the partial energies on the node should not be affected
        # by the presence of a neighbor outside the cutoff radius, thus making the corresponding forces zero.

        _, config, _ = model  # Just to get the config
        r_max = config["r_max"]
        type_names = config["type_names"]

        # Whether the cutoff radius is specified per edge type
        per_edge_type_cutoff = config.get("per_edge_type_cutoff")
        per_edge_type = not (per_edge_type_cutoff is None)

        # Check each edge type
        for node_idx, node_type in enumerate(type_names):
            for nbor_idx, nbor_type in enumerate(type_names):
                # Extract the cutoff radius for the edge type
                if per_edge_type:
                    r_max = per_edge_type_cutoff[node_type]
                    if not isinstance(r_max, float):
                        r_max = r_max[nbor_type]

                # Control group: force is non-zero within the cutoff radius
                partial_forces = pair_force(
                    node_idx, nbor_idx, 0.5 * r_max, 1.5 * r_max, partial=True
                )
                node_partial_forces = partial_forces[0]

                # NOTE: sometimes it can be zero if the model has so little features such that the nonlinearity causes the activation to be ~0
                assert (
                    node_partial_forces.abs().sum() > 1e-4
                ), f"partial forces: {node_partial_forces}"

                # For Test 1 and 2:
                # No need to enforce `strictly_local`. Message passing models such as NequiIP should not receive information from beyond the cutoff radius.
                # In fact, checking that message-passing models still have zero force at the cutoff radius is a good test of locality even with variable cutoffs.

                # Test 1: force is zero at the cutoff radius
                partial_forces = pair_force(
                    node_idx, nbor_idx, r_max, 1.5 * r_max, partial=True
                )
                # Take only the gradients for the energy relating to the node.
                # Only these forces are affected by the node's cutoff radius
                node_partial_forces = partial_forces[0]

                assert torch.allclose(
                    node_partial_forces,
                    torch.zeros_like(
                        node_partial_forces,
                        device=device,
                        dtype=node_partial_forces.dtype,
                    ),
                ), f"partial forces: {node_partial_forces}"

                # Test 2: force is zero outside of the cutoff radius
                partial_forces = pair_force(
                    node_idx, nbor_idx, 1.1 * r_max, 1.5 * r_max, partial=True
                )
                # Take only the gradients for the energy relating to the node.
                # Only these forces are affected by the node's cutoff radius
                node_partial_forces = partial_forces[0]

                assert torch.allclose(
                    node_partial_forces,
                    torch.zeros_like(
                        node_partial_forces,
                        device=device,
                        dtype=node_partial_forces.dtype,
                    ),
                ), f"partial forces: {node_partial_forces}"

    def test_isolated_atom_energies(self, model, device):
        """Checks that isolated atom energies provided for the per-atom shifts are restored for isolated atoms."""
        instance, config, _ = model

        # skip if no per-type energy shifts
        if "per_type_energy_shifts" in config:
            # get the isolated atom energies
            isolated_energies = torch.tensor(
                config["per_type_energy_shifts"], device=device
            )

            # make a synthetic data consisting of three isolated atom frames
            data_list = []
            for type_idx in range(3):
                data = {
                    "atom_types": np.array([type_idx]),
                    "pos": np.array([[0.0, 0.0, 0.0]]),
                }
                data_list.append(from_dict(data))
            data = AtomicDataDict.to_(
                compute_neighborlist_(
                    AtomicDataDict.batched_from_list(data_list), r_max=config["r_max"]
                ),
                device,
            )
            out = instance(data)
            assert torch.allclose(
                out[AtomicDataDict.TOTAL_ENERGY_KEY], isolated_energies.reshape(3, 1)
            )
