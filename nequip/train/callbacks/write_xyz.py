# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import lightning
from lightning.pytorch.callbacks import Callback

import ase

from nequip.data import AtomicDataDict, to_ase
from nequip.data import (
    _register_field_prefix,
    register_fields,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
)
from nequip.train import NequIPLightningModule

from typing import List, Dict, Union, Optional


class TestTimeXYZFileWriter(Callback):
    """Writes model outputs to an ``xyz`` file.

    Users must provide an ``out_file`` that does not contain an extension. The actual output file will take
    the form ``{out_file}_dataset{idx}.xyz`` where ``idx`` is the dataset index (would be ``0`` for a single
    test set but varies depending on number of test sets).

    To incorporate original dataset fields in the ``xyz`` file to simplify analysis, users may provide
    ``output_fields_from_original_dataset``. Such fields will have the prefix ``original_dataset_`` in the ``xyz`` file.

    To obtain correct chemical species information, users must provide ``chemical_species`` in an order consistent with
    the model's ``type_names``.

    Example usage in config to write predictions and original dataset ``total_energy`` and ``forces`` to an ``xyz`` file:
    ::

        callbacks:
          - _target_: nequip.train.callbacks.TestTimeXYZFileWriter
            out_file: ${hydra:runtime.output_dir}/test
            output_fields_from_original_dataset: [total_energy, forces]
            chemical_symbols: ${chemical_symbols}

    Args:
        out_file (str): path to output file (must NOT contain ``.xyz`` or ``.extxyz`` extension)
        output_fields_from_original_dataset (List[str]): values from the original dataset to save in the ``out_file``
        chemical_species (List[str]): chemical species in the same order as model's ``type_names``
    """

    def __init__(
        self,
        out_file: str,
        output_fields_from_original_dataset: Optional[List[str]] = [],
        chemical_symbols: Optional[List[str]] = None,
    ):
        assert not (out_file.endswith(".xyz") or out_file.endswith(".extxyz"))
        self.out_file = out_file
        assert all(
            [
                field in (_NODE_FIELDS | _EDGE_FIELDS | _GRAPH_FIELDS)
                for field in output_fields_from_original_dataset
            ]
        )

        # special case total_energy (nequip's convention) vs energy (ase's convention)
        self.output_fields_from_original_dataset = []
        for field in output_fields_from_original_dataset:
            if field == "total_energy":
                self.output_fields_from_original_dataset.append("energy")
                register_fields(graph_fields=["original_dataset_energy"])
            else:
                self.output_fields_from_original_dataset.append(field)
        _register_field_prefix("original_dataset_")

        self.extra_fields = [
            "original_dataset_" + field
            for field in self.output_fields_from_original_dataset
        ]
        self.chemical_symbols = chemical_symbols

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        outputs: Dict[str, Union[torch.Tensor, AtomicDataDict.Type]],
        batch: AtomicDataDict.Type,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """"""
        with torch.no_grad():
            output_out = outputs[f"test_{dataloader_idx}_output"].copy()
            for field in self.output_fields_from_original_dataset:
                # special case total_energy (nequip's convention) vs energy (ase's convention)
                if field == "energy":
                    output_out["original_dataset_energy"] = batch["total_energy"]
                else:
                    output_out["original_dataset_" + field] = batch[field]

            # !! EXTREMELY IMPORTANT -- special handling of PBC key if present !!
            # ASE data inputs would possess it to be used at data precprocessing time (i.e. neighborlist construction)
            # but it won't be passed through the model, so we get it from `batch`
            if AtomicDataDict.PBC_KEY in batch:
                output_out[AtomicDataDict.PBC_KEY] = batch[AtomicDataDict.PBC_KEY]

            # append to the file
            ase.io.write(
                self.out_file + f"_dataset{dataloader_idx}.xyz",
                to_ase(
                    output_out,
                    chemical_symbols=self.chemical_symbols,
                    extra_fields=self.extra_fields,
                ),
                format="extxyz",
                append=True,
            )
            del output_out
