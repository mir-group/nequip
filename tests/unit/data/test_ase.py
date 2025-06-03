import pytest
import numpy as np
import torch
import ase.build
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

from nequip.data import AtomicDataDict, from_ase, to_ase


class TestASEDataParsing:
    """Test ASE data parsing functionality, especially tensor format handling."""

    def test_basic_ase_parsing(self):
        """Test basic ASE to AtomicDataDict conversion."""
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-10.5,
            forces=np.random.random((len(atoms), 3)),
        )

        data = from_ase(atoms)

        assert AtomicDataDict.POSITIONS_KEY in data
        assert AtomicDataDict.CELL_KEY in data
        assert AtomicDataDict.PBC_KEY in data
        assert AtomicDataDict.TOTAL_ENERGY_KEY in data
        assert AtomicDataDict.FORCE_KEY in data

        # Check basic shapes
        assert data[AtomicDataDict.POSITIONS_KEY].shape == (len(atoms), 3)
        assert data[AtomicDataDict.FORCE_KEY].shape == (len(atoms), 3)
        assert data[AtomicDataDict.TOTAL_ENERGY_KEY].shape == (1, 1)

    def test_stress_voigt_format_parsing(self):
        """Test parsing stress from Voigt (6,) format."""
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)

        # Voigt format: [xx, yy, zz, yz, xz, xy]
        stress_voigt = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        atoms.calc = SinglePointCalculator(atoms, stress=stress_voigt)

        data = from_ase(atoms)

        # Should be reshaped to (1, 3, 3) with batch dimension
        assert AtomicDataDict.STRESS_KEY in data
        assert data[AtomicDataDict.STRESS_KEY].shape == (1, 3, 3)

        # Verify conversion using ASE's function
        expected_3x3 = voigt_6_to_full_3x3_stress(stress_voigt)
        assert torch.allclose(
            data[AtomicDataDict.STRESS_KEY][0],
            torch.from_numpy(expected_3x3),
            atol=1e-6,
        )

    def test_stress_3x3_format_parsing(self):
        """Test parsing stress from full (3, 3) format."""
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)

        stress_3x3 = np.array([[1.0, 0.3, 0.2], [0.3, 2.0, 0.1], [0.2, 0.1, 3.0]])
        atoms.calc = SinglePointCalculator(atoms, stress=stress_3x3)

        data = from_ase(atoms)

        assert data[AtomicDataDict.STRESS_KEY].shape == (1, 3, 3)
        assert torch.allclose(
            data[AtomicDataDict.STRESS_KEY][0], torch.from_numpy(stress_3x3)
        )

    def test_stress_flat_9_format_parsing(self):
        """Test parsing stress from flat (9,) format."""
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)

        # Flat format: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
        stress_flat = np.array([1.0, 0.3, 0.2, 0.3, 2.0, 0.1, 0.2, 0.1, 3.0])
        atoms.info["stress"] = stress_flat  # Put in info rather than calc

        data = from_ase(atoms)

        assert data[AtomicDataDict.STRESS_KEY].shape == (1, 3, 3)
        expected_3x3 = stress_flat.reshape(3, 3)
        assert torch.allclose(
            data[AtomicDataDict.STRESS_KEY][0], torch.from_numpy(expected_3x3)
        )

    def test_stress_shape_validation_safety_asserts(self):
        """Test that the safety asserts in ase.py work correctly."""
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)

        # Test valid Voigt format
        stress_voigt = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        atoms.calc = SinglePointCalculator(atoms, stress=stress_voigt)
        data = from_ase(atoms)
        assert data[AtomicDataDict.STRESS_KEY].shape == (1, 3, 3)

        # Test valid 3x3 format
        stress_3x3 = np.random.random((3, 3))
        atoms.calc = SinglePointCalculator(atoms, stress=stress_3x3)
        data = from_ase(atoms)
        assert data[AtomicDataDict.STRESS_KEY].shape == (1, 3, 3)

    def test_born_charge_tensor_format_parsing(self):
        """Test parsing Born charge tensors in different formats."""
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)

        # Born charges as (N_atoms, 9) flat format
        n_atoms = len(atoms)
        born_charges_flat = np.random.random((n_atoms, 9))
        atoms.arrays["born_effective_charges"] = born_charges_flat

        data = from_ase(atoms)

        # Should be reshaped to (N_atoms, 3, 3)
        assert "born_effective_charges" in data
        assert data["born_effective_charges"].shape == (n_atoms, 3, 3)

        # Verify conversion
        for i in range(n_atoms):
            expected_3x3 = born_charges_flat[i].reshape(3, 3)
            assert torch.allclose(
                data["born_effective_charges"][i],
                torch.from_numpy(expected_3x3),
            )

    def test_ase_roundtrip_with_stress(self):
        """Test ASE -> AtomicDataDict -> ASE roundtrip preserves stress."""
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
        stress_3x3 = np.array([[1.0, 0.3, 0.2], [0.3, 2.0, 0.1], [0.2, 0.1, 3.0]])
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-10.5,
            forces=np.random.random((len(atoms), 3)),
            stress=stress_3x3,
        )

        # Round trip
        data = from_ase(atoms)
        atoms_reconstructed = to_ase(data)[0]

        # Check stress conversion back to Voigt format
        stress_reconstructed = atoms_reconstructed.calc.results["stress"]
        expected_voigt = full_3x3_to_voigt_6_stress(stress_3x3)

        assert np.allclose(expected_voigt, stress_reconstructed, atol=1e-6)

    def test_multiple_cartesian_tensor_fields(self):
        """Test parsing multiple cartesian tensor fields simultaneously."""
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)

        # Add multiple tensor fields - use registered keys
        stress_3x3 = np.random.random((3, 3))
        dielectric_flat = np.random.random(9)
        born_charges_flat = np.random.random((len(atoms), 9))

        atoms.calc = SinglePointCalculator(atoms, stress=stress_3x3)
        atoms.info["dielectric_tensor"] = dielectric_flat
        atoms.arrays["born_effective_charges"] = born_charges_flat

        data = from_ase(atoms)

        # Check all tensor fields are properly shaped
        assert data[AtomicDataDict.STRESS_KEY].shape == (1, 3, 3)
        assert data["dielectric_tensor"].shape == (1, 3, 3)
        assert data["born_effective_charges"].shape == (len(atoms), 3, 3)

    def test_ase_parsing_with_custom_keys(self):
        """Test ASE parsing with custom key mapping."""
        from nequip.data import _key_registry

        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
        stress_voigt = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        atoms.calc = SinglePointCalculator(atoms, stress=stress_voigt)

        # Register custom key as cartesian tensor temporarily
        _key_registry.register_fields(
            graph_fields=["custom_stress"],
            cartesian_tensor_fields={"custom_stress": "ij=ji"},
        )

        try:
            # Use custom key mapping
            data = from_ase(
                atoms,
                key_mapping={"stress": "custom_stress"},
                include_keys=["custom_stress"],
            )

            assert "custom_stress" in data
            assert data["custom_stress"].shape == (1, 3, 3)
        finally:
            # Clean up registration
            _key_registry.deregister_fields("custom_stress")

    def test_molecular_vs_periodic_systems(self):
        """Test ASE parsing for both molecular and periodic systems."""
        # Periodic system
        periodic_atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
        periodic_atoms.calc = SinglePointCalculator(
            periodic_atoms, stress=np.random.random((3, 3))
        )

        periodic_data = from_ase(periodic_atoms)
        assert periodic_data[AtomicDataDict.PBC_KEY].any()
        assert AtomicDataDict.CELL_KEY in periodic_data

        # Molecular system
        molecular_atoms = ase.build.molecule("H2O")
        molecular_atoms.calc = SinglePointCalculator(
            molecular_atoms,
            energy=-10.0,
            forces=np.random.random((len(molecular_atoms), 3)),
        )

        molecular_data = from_ase(molecular_atoms)
        assert not molecular_data[AtomicDataDict.PBC_KEY].any()


@pytest.fixture(scope="function")
def cu_bulk_with_stress():
    """Create Cu bulk system with stress for testing."""
    atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms = ase.build.make_supercell(atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])

    # Realistic stress tensor (symmetric)
    stress_3x3 = np.array(
        [[50.0, 5.0, 2.0], [5.0, 45.0, 3.0], [2.0, 3.0, 40.0]]
    )  # GPa scale

    atoms.calc = SinglePointCalculator(
        atoms,
        energy=np.random.random() * -100,
        forces=np.random.random((len(atoms), 3)) * 0.1,
        stress=stress_3x3,
    )

    return atoms, stress_3x3
