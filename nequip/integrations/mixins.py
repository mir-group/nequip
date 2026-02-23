# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.model.utils import _EAGER_MODEL_KEY
from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip.nn import graph_model
from nequip.utils.global_state import set_global_state

from .utils import handle_chemical_species_map, basic_transforms
from pathlib import Path
from typing import Union, Optional, Dict


class _IntegrationLoaderMixin:
    """Shared loader methods for NequIP integrations."""

    @classmethod
    def _get_aoti_compile_target(cls) -> Dict:
        """Return the AOTI compile target dictionary for this integration."""
        raise NotImplementedError(
            "subclasses must implement `_get_aoti_compile_target`"
        )

    @classmethod
    def from_compiled_model(
        cls,
        compile_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        chemical_species_to_atom_type_map: Optional[Union[Dict[str, str], bool]] = None,
        neighborlist_backend: str = "matscipy",
        **kwargs,
    ):
        """Build an integration calculator from a compiled model artifact.

        Args:
            compile_path: path to the compiled model artifact.
            device: device where the model is loaded and evaluated.
            chemical_species_to_atom_type_map: optional chemical species mapping override.
            neighborlist_backend: neighbor list backend used by neighbor transforms.
            **kwargs: forwarded to the integration class constructor.
        """
        from nequip.model.inference_models import load_compiled_model

        target = cls._get_aoti_compile_target()
        input_keys = list(target["input"])
        output_keys = list(target["output"])

        model, metadata = load_compiled_model(
            str(compile_path), device, input_keys, output_keys
        )

        r_max = metadata[graph_model.R_MAX_KEY]
        type_names = metadata[graph_model.TYPE_NAMES_KEY]

        chemical_species_to_atom_type_map = handle_chemical_species_map(
            chemical_species_to_atom_type_map, type_names
        )

        if "transforms" in kwargs:
            raise KeyError("`transforms` not allowed here")

        return cls(
            model=model,
            device=device,
            transforms=basic_transforms(
                metadata,
                r_max,
                type_names,
                chemical_species_to_atom_type_map,
                neighborlist_backend=neighborlist_backend,
            ),
            **kwargs,
        )

    @classmethod
    def _from_saved_model(
        cls,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        chemical_species_to_atom_type_map: Optional[Union[Dict[str, str], bool]] = None,
        allow_tf32: bool = False,
        model_name: str = _SOLE_MODEL_KEY,
        compile_mode: str = _EAGER_MODEL_KEY,
        neighborlist_backend: str = "matscipy",
        **kwargs,
    ):
        """Build an integration calculator from a saved NequIP model.

        Args:
            model_path: path to the saved model.
            device: device where the model is loaded and evaluated.
            chemical_species_to_atom_type_map: optional chemical species mapping override.
            allow_tf32: whether to allow TF32 for supported math ops.
            model_name: key of the model inside the saved model artifact.
            compile_mode: compile mode for loading the model; supported values
                are ``"eager"`` and ``"compile"`` (default: ``"eager"``).
            neighborlist_backend: neighbor list backend used by neighbor transforms.
            **kwargs: forwarded to the integration class constructor.
        """
        from nequip.model.saved_models.load_utils import load_saved_model

        set_global_state(allow_tf32=allow_tf32)

        model: graph_model.GraphModel = load_saved_model(
            str(model_path), compile_mode=compile_mode, model_key=model_name
        )
        model.eval()

        r_max = float(model.metadata[graph_model.R_MAX_KEY])
        type_names = model.metadata[graph_model.TYPE_NAMES_KEY].split(" ")

        chemical_species_to_atom_type_map = handle_chemical_species_map(
            chemical_species_to_atom_type_map, type_names
        )

        if "transforms" in kwargs:
            raise KeyError("`transforms` not allowed here")

        return cls(
            model=model,
            device=device,
            transforms=basic_transforms(
                model.metadata,
                r_max,
                type_names,
                chemical_species_to_atom_type_map,
                neighborlist_backend=neighborlist_backend,
            ),
            **kwargs,
        )
