# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
Basic and energy-specific model test mixins.

These mixins provide core functionality tests and energy model-specific tests.
They can be composed with other test mixins (compilation, integrations) as needed.
"""

import pytest
import copy
import functools
import torch
import numpy as np

from nequip.data import (
    from_dict,
    from_ase,
    to_ase,
    compute_neighborlist_,
    AtomicDataDict,
    _GRAPH_FIELDS,
    _NODE_FIELDS,
    _EDGE_FIELDS,
)
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from nequip.nn import (
    ForceStressOutput,
    PartialForceOutput,
    PerTypeScaleShift,
)
from nequip.utils import dtype_to_name, find_first_of_type
from nequip.utils.test import (
    assert_AtomicData_equivariant,
    FLOAT_TOLERANCE,
)

from hydra.utils import get_method, instantiate
from .utils import _training_session


class BasicModelTestsMixin:
    """
    Core model tests: initialization, forward pass, equivariance, batching, cutoff behavior.

    These tests should pass for any graph model, regardless of output type.
    Subclasses must provide a `config` fixture that returns model configuration.
    """

    @pytest.fixture(scope="class")
    def config(self):
        """Implemented by subclasses.

        Return a model configuration dict.
        """
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def equivariance_tol(self, model_dtype):
        """May be overriden by subclasses.

        Returns tolerance based on ``model_dtype``.
        """
        return {"float32": 1e-3, "float64": 1e-8}[model_dtype]

    @pytest.fixture
    def strict_locality(self):
        """Override in subclasses.

        True: model is strictly local (forces only depend on neighbors within cutoff)
        False: model has message passing (forces may depend on broader neighborhood)

        This is a fundamental architectural property of the model.
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
        config = copy.deepcopy(model_config)
        model_builder = get_method(config.pop("_target_"))
        model = model_builder(**config)
        model = model.to(device)
        return model

    @pytest.fixture(scope="class")
    def model(self, config, device, model_dtype):
        # === sanity check contracts with subclasses ===
        assert "model_dtype" not in config, (
            "model test subclasses should not include `model_dtype` in the configs -- the test class will handle it (looping over `float32` and `float64`)"
        )
        assert "compile_mode" not in config, (
            "model test subclasses should not include `compile_mode` in the configs -- the test class will handle it"
        )
        config = copy.deepcopy(config)
        config.update({"model_dtype": model_dtype})
        model = self.make_model(config, device=device)
        out_fields = model.irreps_out.keys()
        # we return the config with the correct `model_dtype`
        return model, config, out_fields

    @pytest.fixture(scope="class")
    def partial_model(self, model, device):
        _, config, _ = model
        aux_model = self.make_model(copy.deepcopy(config), device=device)
        module = find_first_of_type(aux_model, ForceStressOutput)
        # skip test if force/stress module not found
        if module is None:
            pytest.skip("force/stress module not found")
        # replace force/stress module with partial force module
        aux_model.model = PartialForceOutput(module.func)
        out_fields = aux_model.model.irreps_out.keys()

        # partial_model does not output config. Please use `model` fixture for config
        return aux_model, out_fields

    @pytest.fixture(scope="class", params=["molecules", "bulk"])
    def model_test_data(self, config, atomic_batch, diamond_carbon, device, request):
        if request.param == "molecules":
            test_data = atomic_batch
        elif request.param == "bulk":
            test_data = diamond_carbon
        else:
            raise ValueError(f"Unknown test data parameter: {request.param}")

        # clone the data to avoid mutating the original batch
        # cpu because we need to reconstruct the neighborlist
        test_data = {k: v.clone().detach().to("cpu") for k, v in test_data.items()}
        # reset neighborlist
        test_data = NeighborListTransform(r_max=config["r_max"])(test_data)
        # return the data in the device for testing
        return AtomicDataDict.to_(test_data, device)

    @pytest.fixture(
        scope="class",
        params=["minimal_aspirin.yaml", "minimal_toy_emt.yaml"],
    )
    def conffile(self, request):
        """Training config files."""
        return request.param

    @pytest.fixture(scope="class")
    def train_fn(self):
        """Training session function - can be overridden by subclasses to use alternative training workflows.

        Default is `_training_session` which runs `nequip-train`.
        Subclasses can override to use different training commands.
        """
        return _training_session

    def model_parameter_updates(self, model_config):
        """
        Returns dict of parameter updates to inject into model config.

        Default behavior injects training_data_stats interpolations for NequIP framework models.
        Subclasses can override to customize injection behavior.

        Args:
            model_config: Model configuration dict to inspect

        Returns:
            dict: Parameter updates to apply
        """
        updates = {
            # use resolvers for data-dependent parameters
            "avg_num_neighbors": "${training_data_stats:num_neighbors_mean}",
            "per_type_energy_shifts": "${training_data_stats:per_atom_energy_mean}",
            "per_type_energy_scales": "${training_data_stats:per_type_forces_rms}",
            # use top-level config variable references
            "type_names": "${model_type_names}",
        }

        # handle nested chemical_species in pair_potential if present
        if (
            "pair_potential" in model_config
            and "chemical_species" in model_config["pair_potential"]
        ):
            updates["pair_potential.chemical_species"] = "${chemical_species}"

        # handle per_edge_type_cutoff - use the one from integration config if present
        if "per_edge_type_cutoff" in model_config:
            updates["per_edge_type_cutoff"] = "${per_edge_type_cutoff}"

        return updates

    def _update_config_recursively(self, config, updates):
        """
        Recursively update config with nested parameter updates.

        Args:
            config: Configuration dict to update
            updates: Dict with potentially nested keys using dots (e.g., "pair_potential.chemical_species")
        """
        for key, value in updates.items():
            if "." in key:
                # handle nested keys like "pair_potential.chemical_species"
                parts = key.split(".", 1)
                parent_key, child_key = parts[0], parts[1]
                if parent_key in config:
                    # recursively update the nested dict
                    self._update_config_recursively(
                        config[parent_key], {child_key: value}
                    )
            else:
                config[key] = value

    @pytest.fixture(scope="class", params=["checkpoint", "package"])
    def model_source(self, request):
        """model source for fake_model_training_session. subclasses can override params."""
        return request.param

    def load_validation_structures(self, training_config, tmpdir):
        """
        Load validation structures for testing.

        Default behavior loads from NequIP datamodule.
        Subclasses can override for different training workflows.

        Args:
            training_config: training configuration dict
            tmpdir: temporary directory path

        Returns:
            List of ASE Atoms objects for validation
        """
        datamodule = instantiate(training_config.data, _recursive_=False)
        datamodule.prepare_data()
        datamodule.setup("validate")
        dloader = datamodule.val_dataloader()[0]

        structures = []
        for data in dloader:
            structures += to_ase(data.copy())
        return structures

    # this means we also check if we can `nequip-package` a specific model
    # this could reveal e.g. missing externs, etc
    @pytest.fixture(scope="class")
    def fake_model_training_session(
        self, conffile, config, model_dtype, train_fn, model_source
    ):
        """Create a fake training session using integration test configs with injected model."""
        # make a deep copy and enforce integration config parameters
        model_config = copy.deepcopy(config)
        # get parameter updates (can be overridden by subclasses)
        updates = self.model_parameter_updates(model_config)
        # update model config with necessary updates
        self._update_config_recursively(model_config, updates)

        session = train_fn(
            conffile,
            model_dtype,
            extra_train_from_save=None
            if model_source == "checkpoint"
            else model_source,
            model_config=model_config,
        )
        training_config, tmpdir, env = next(session)

        # load validation structures (can be overridden by subclasses)
        structures = self.load_validation_structures(training_config, tmpdir)

        yield training_config, tmpdir, env, model_dtype, model_source, structures
        del session

    def test_init(self, model):
        instance, _, _ = model
        assert (
            hasattr(instance, "_is_graph_module_mixin")
            and instance._is_graph_module_mixin
        )

    def test_model_repr(self, model):
        """Test that the model can be represented as a string without errors."""
        instance, _, _ = model
        repr_str = repr(instance)
        assert isinstance(repr_str, str)

    def test_forward(self, model, model_test_data):
        """Tests that we can run a forward pass without errors."""
        instance, _, _ = model
        _ = instance(model_test_data)

    def test_wrapped_unwrapped(self, model, device, Cu_bulk):
        atoms, data_orig = Cu_bulk
        instance, _, _ = model
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
            for out_field in out_ref.keys():
                # not important for the purposes of this test
                if out_field in [
                    AtomicDataDict.POSITIONS_KEY,
                    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
                ]:
                    continue
                assert torch.allclose(
                    out_ref[out_field], out_unwrapped[out_field], atol=tolerance
                ), (
                    f'failed for key "{out_field}" with max absolute diff {torch.abs(out_ref[out_field] - out_unwrapped[out_field]).max().item():.5g} (tol={tolerance:.5g})'
                )

    def test_batch(self, model, model_test_data):
        """Confirm that the results for individual examples are the same regardless of whether they are batched."""
        instance, _, _ = model

        tolerance = FLOAT_TOLERANCE[dtype_to_name(instance.model_dtype)]
        allclose = functools.partial(torch.allclose, atol=tolerance)
        data1 = AtomicDataDict.frame_from_batched(model_test_data, 0)
        data2 = AtomicDataDict.frame_from_batched(model_test_data, 1)
        output1 = instance(data1)
        output2 = instance(data2)
        output = instance(model_test_data)
        for out_field in output.keys():
            # to ignore
            if out_field in [
                AtomicDataDict.EDGE_INDEX_KEY,
                AtomicDataDict.BATCH_KEY,
                AtomicDataDict.EDGE_TYPE_KEY,
            ]:
                continue
            if out_field in _GRAPH_FIELDS:
                # this condition is to account for models that may produce empty dummy tensors for certain fields
                # e.g. nonperiodic systems without a cell may produce empty stress tensors
                if output[out_field].numel() == 0:
                    continue
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

    def test_equivariance(self, model, model_test_data, device, equivariance_tol):
        instance, _, _ = model
        instance = instance.to(device=device)

        assert_AtomicData_equivariant(
            func=instance,
            data_in=model_test_data,
            e3_tolerance=equivariance_tol,
        )


class EnergyModelTestsMixin(BasicModelTestsMixin):
    """
    Tests specific to energy models (models that output energies and forces).

    Inherits from BasicModelTestsMixin, adding energy-specific tests:
    - Large separation additivity
    - Cross-frame gradient isolation
    - Numerical gradient correctness
    - Partial forces
    - Force smoothness at cutoff
    - Isolated atom energy shifts
    """

    def total_energy_keys(self):
        """Return list of total energy keys to test.

        Subclasses can override to test multiple energy keys (e.g., energy_0, energy_1).

        Returns:
            List of total energy key strings
        """
        return [AtomicDataDict.TOTAL_ENERGY_KEY]

    def per_atom_energy_keys(self):
        """Return list of per-atom energy keys to test.

        Subclasses can override to test multiple energy keys (e.g., per_atom_energy_0, per_atom_energy_1).

        Returns:
            List of per-atom energy key strings
        """
        return [AtomicDataDict.PER_ATOM_ENERGY_KEY]

    def force_keys(self):
        """Return list of force keys to test.

        Subclasses can override to test multiple force keys (e.g., force_0, force_1).

        Returns:
            List of force key strings
        """
        return [AtomicDataDict.FORCE_KEY]

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
            model_type_names=["H", "C", "O"],
            chemical_species_to_atom_type_map={"H": "H", "C": "C", "O": "O"},
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

        # test all total energy keys
        for energy_key in self.total_energy_keys():
            if energy_key in out1 and energy_key in out2 and energy_key in out_both:
                assert torch.allclose(
                    out1[energy_key] + out2[energy_key],
                    out_both[energy_key],
                    atol=atol,
                ), f"Large separation test failed for {energy_key}"

        # test all force keys
        for force_key in self.force_keys():
            if force_key in out1:
                # check forces if it's a force model
                assert torch.allclose(
                    torch.cat(
                        (out1[force_key], out2[force_key]),
                        dim=0,
                    ),
                    out_both[force_key],
                    atol=atol,
                ), f"Large separation test failed for {force_key}"

        atoms_both2 = atoms1.copy()
        atoms3 = atoms2.copy()
        atoms3.positions += np.random.randn(3)
        atoms_both2.extend(atoms3)

        data_both2 = AtomicDataDict.to_(
            tm(compute_neighborlist_(from_ase(atoms_both2), r_max=r_max)),
            device,
        )

        out_both2 = instance(data_both2)

        # test total energy invariance to rigid translation
        for energy_key in self.total_energy_keys():
            if energy_key in out_both and energy_key in out_both2:
                assert torch.allclose(
                    out_both2[energy_key],
                    out_both[energy_key],
                    atol=atol,
                ), f"Translation invariance test failed for {energy_key}"

        # test per-atom energy invariance to rigid translation
        for per_atom_energy_key in self.per_atom_energy_keys():
            if per_atom_energy_key in out_both and per_atom_energy_key in out_both2:
                assert torch.allclose(
                    out_both2[per_atom_energy_key],
                    out_both[per_atom_energy_key],
                    atol=atol,
                ), f"Translation invariance test failed for {per_atom_energy_key}"

    def test_cross_frame_grad(self, model, device, nequip_dataset):
        batch = AtomicDataDict.batched_from_list(
            [nequip_dataset[i] for i in range(len(nequip_dataset))]
        )
        energy_model, _, _ = model
        data = AtomicDataDict.to_(batch, device)
        data[AtomicDataDict.POSITIONS_KEY].requires_grad = True

        output = energy_model(data)

        # test cross-frame gradient isolation for all total energy keys
        for energy_key in self.total_energy_keys():
            if energy_key in output:
                grads = torch.autograd.grad(
                    outputs=output[energy_key][-1],
                    inputs=data[AtomicDataDict.POSITIONS_KEY],
                    allow_unused=True,
                    retain_graph=True,
                )[0]

                last_frame_n_atom = batch[AtomicDataDict.NUM_NODES_KEY][-1]

                in_frame_grad = grads[-last_frame_n_atom:]
                cross_frame_grad = grads[:-last_frame_n_atom]

                assert cross_frame_grad.abs().max().item() == 0, (
                    f"Cross-frame gradient test failed for {energy_key}"
                )
                assert in_frame_grad.abs().max().item() > 0, (
                    f"In-frame gradient test failed for {energy_key}"
                )

    def test_numeric_gradient(self, model, atomic_batch, device):
        """
        Tests the ForceStressOutput model by comparing numerical gradients of the forces to the analytical gradients.
        """
        model, _, out_fields = model

        # test numeric gradients for each pair of (total_energy_key, force_key)
        for energy_key, force_key in zip(self.total_energy_keys(), self.force_keys()):
            # proceed with tests only if forces are available
            if force_key in out_fields:
                # physical predictions (energy, forces, etc) will be converted to default_dtype (float64) before comparing
                data = AtomicDataDict.to_(atomic_batch, device)
                output = model(data)
                forces = output[force_key]
                epsilon = 1e-3

                # compute numerical gradients for each atom and direction and compare to analytical gradients
                for iatom in range(len(data[AtomicDataDict.POSITIONS_KEY])):
                    for idir in range(3):
                        # shift `iatom` an `epsilon` in the `idir` direction
                        pos = data[AtomicDataDict.POSITIONS_KEY][iatom, idir]
                        data[AtomicDataDict.POSITIONS_KEY][iatom, idir] = pos + epsilon
                        output = model(data)
                        e_plus = output[energy_key].sum().to(torch.get_default_dtype())

                        # shift `iatom` an `epsilon` in the negative `idir` direction
                        data[AtomicDataDict.POSITIONS_KEY][iatom, idir] -= epsilon * 2
                        output = model(data)
                        e_minus = output[energy_key].sum().to(torch.get_default_dtype())

                        # symmetric difference to get the partial forces to all the atoms
                        numeric = -(e_plus - e_minus) / (epsilon * 2)
                        analytical = forces[iatom, idir].to(torch.get_default_dtype())

                        assert torch.isclose(
                            numeric, analytical, atol=2e-2
                        ) or torch.isclose(numeric, analytical, rtol=5e-3), (
                            f"numeric: {numeric.item()}, analytical: {analytical.item()} for {energy_key}/{force_key}"
                        )

                        # reset the position
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
                    ) or torch.allclose(numeric, analytical, rtol=5e-2), (
                        f"numeric: {numeric.item()}, analytical: {analytical.item()}"
                    )

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
        per_edge_type = per_edge_type_cutoff is not None

        # Check each edge type
        for node_idx, node_type in enumerate(type_names):
            for nbor_idx, nbor_type in enumerate(type_names):
                # extract the cutoff radius for the edge type
                if per_edge_type:
                    if node_type in per_edge_type_cutoff:
                        r_max = per_edge_type_cutoff[node_type]
                        if not isinstance(r_max, float):
                            if nbor_type in r_max:
                                r_max = r_max[nbor_type]
                            else:
                                # default missing target types to global r_max
                                r_max = config["r_max"]
                    else:
                        # default missing source types to global r_max
                        r_max = config["r_max"]

                # Control group: force is non-zero within the cutoff radius
                partial_forces = pair_force(
                    node_idx, nbor_idx, 0.5 * r_max, 1.5 * r_max, partial=True
                )
                node_partial_forces = partial_forces[0]

                # NOTE: sometimes it can be zero if the model has so little features such that the nonlinearity causes the activation to be ~0
                assert node_partial_forces.abs().sum() > 1e-4, (
                    f"partial forces: {node_partial_forces}"
                )

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
        scale_shift_module = find_first_of_type(instance, PerTypeScaleShift)

        if scale_shift_module is not None:
            if scale_shift_module.has_shifts:
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
                        AtomicDataDict.batched_from_list(data_list),
                        r_max=config["r_max"],
                    ),
                    device,
                )
                output = instance(data)

                # test isolated atom energies for all total energy keys
                for energy_key in self.total_energy_keys():
                    if energy_key in output:
                        energies = output[energy_key]
                        assert torch.allclose(
                            energies, scale_shift_module.shifts.reshape(energies.shape)
                        ), f"Isolated atom energy test failed for {energy_key}"

    def test_embedding_cutoff(self, model, device):
        """Test that edge embeddings/features go to zero at cutoff and gradients are correct."""
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
            return
        edge_embed = edge_embed[key]
        data[AtomicDataDict.POSITIONS_KEY][2, 1] = r_max  # put it past the cutoff
        edge_embed2 = instance(from_dict(data))[key]

        if key == AtomicDataDict.EDGE_EMBEDDING_KEY:
            # we can only check that other edges are unaffected if we know it's an embedding
            # For example, an Allegro edge feature is many body so will be affected
            torch.testing.assert_close(edge_embed[:2], edge_embed2[:2])
        assert edge_embed[2:].abs().sum() > 1e-6, (
            f"Edge embeddings at cutoff should have some nonzero terms before moving atom to cutoff, "
            f"but got sum of absolute values: {edge_embed[2:].abs().sum()}"
        )
        torch.testing.assert_close(
            edge_embed2[2:],
            torch.zeros_like(edge_embed2[2:]),
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
            torch.testing.assert_close(
                grads,
                torch.zeros_like(grads),
            )

            # test that first two atoms' energies are unaffected by atom at cutoff
            for per_atom_energy_key in self.per_atom_energy_keys():
                if per_atom_energy_key in out:
                    # are the first two atom's energies unaffected by atom at the cutoff?
                    grads = torch.autograd.grad(
                        outputs=out[per_atom_energy_key][:2].sum(),
                        inputs=in_dict[AtomicDataDict.POSITIONS_KEY],
                        retain_graph=True,
                    )[0]
                    # only care about gradient wrt moved atom
                    assert grads.shape == (3, 3), (
                        f"Expected gradient shape (3, 3) for 3 atoms in 3D, got {grads.shape}"
                    )
                    torch.testing.assert_close(
                        grads[2],
                        torch.zeros_like(grads[2]),
                    )
