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


class XYZFileWriter(Callback):
    """Writes model outputs to an ``xyz`` file.

    Users must provide an ``out_file`` that does not contain an extension. The actual output file will take
    the form ``{out_file}_dataset{idx}[_epoch{epoch}].xyz`` where ``idx`` is the dataset index (would be ``0`` for a single
    validation set but varies depending on number of validation sets) and ``epoch`` is the epoch when the file is produced.

    To incorporate original dataset fields in the ``xyz`` file to simplify analysis, users may provide
    ``output_fields_from_original_dataset``. Such fields will have the prefix ``original_dataset_`` in the ``xyz`` file.

    To obtain correct chemical species information, users must provide ``chemical_species`` in an order consistent with
    the model's ``type_names``.

    To activate the option to save to a different file every epoch, users should set ``separate_file_per_epoch`` true.

    Args:
        out_file (str): path to output file (must NOT contain ``.xyz`` or ``.extxyz`` extension)
        output_fields_from_original_dataset (List[str]): values from the original dataset to save in the ``out_file``
        extra_fields (List[str]): extra fields to save in addition to ASE's default fields
        chemical_species (List[str]): chemical species in the same order as model's ``type_names``
    """

    def __init__(
        self,
        out_file: str,
        output_fields_from_original_dataset: Optional[List[str]] = [],
        extra_fields: List[str] = [],
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
        ] + extra_fields
        self.chemical_symbols = chemical_symbols

        # Could be overwritten by children:
        self.separate_file_per_epoch = False

        # To be overridden by children
        self.prefix = None

    def _batch_end(
        self,
        trainer: lightning.Trainer,
        outputs: Dict[str, Union[torch.Tensor, AtomicDataDict.Type]],
        batch: AtomicDataDict.Type,
        dataloader_idx=0,
    ):
        with torch.no_grad():
            output_out = outputs[f"{self.prefix}_{dataloader_idx}_output"].copy()
            for field in self.output_fields_from_original_dataset:
                # special case total_energy (nequip's convention) vs energy (ase's convention)
                if field == "energy":
                    output_out["original_dataset_energy"] = batch["total_energy"]
                else:
                    output_out["original_dataset_" + field] = batch[field]

            # !! EXTREMELY IMPORTANT -- special handling of PBC key if present !!
            # ASE data inputs would possess it to be used at data preprocessing time (i.e. neighborlist construction)
            # but it won't be passed through the model, so we get it from `batch`
            if AtomicDataDict.PBC_KEY in batch:
                output_out[AtomicDataDict.PBC_KEY] = batch[AtomicDataDict.PBC_KEY]

            # Determine the file
            if self.separate_file_per_epoch:
                out_path = (
                    self.out_file
                    + f"_dataset{dataloader_idx}_epoch{trainer.current_epoch}.xyz"
                )
            else:
                out_path = self.out_file + f"_dataset{dataloader_idx}.xyz"
            # append to the file
            ase.io.write(
                out_path,
                to_ase(
                    output_out,
                    chemical_symbols=self.chemical_symbols,
                    extra_fields=self.extra_fields,
                ),
                format="extxyz",
                append=True,
            )
            del output_out


class TestTimeXYZFileWriter(XYZFileWriter):
    """XYZFileWriter designed for saving Test Time Predictions

    Users must provide an ``out_file`` that does not contain an extension. The actual output file will take
    the form ``{out_file}_dataset{idx}[_epoch{epoch}].xyz`` where ``idx`` is the dataset index (would be ``0`` for a single
    validation set but varies depending on number of validation sets) and ``epoch`` is the epoch when the file is produced.

    To incorporate original dataset fields in the ``xyz`` file to simplify analysis, users may provide
    ``output_fields_from_original_dataset``. Such fields will have the prefix ``original_dataset_`` in the ``xyz`` file.

    To obtain correct chemical species information, users must provide ``chemical_species`` in an order consistent with
    the model's ``type_names``.

    To activate the option to save to a different file every epoch, users should set ``separate_file_per_epoch`` true.

    Args:
        out_file (str): path to output file (must NOT contain ``.xyz`` or ``.extxyz`` extension)
        output_fields_from_original_dataset (List[str]): values from the original dataset to save in the ``out_file``
        extra_fields (List[str]): extra fields to save in addition to ASE's default fields
        chemical_species (List[str]): chemical species in the same order as model's ``type_names``

    Example usage in config to write predictions and original dataset ``total_energy`` and ``forces`` to an ``xyz`` file:

    .. code-block:: yaml

        callbacks:
          - _target_: nequip.train.callbacks.TestTimeXYZFileWriter
            out_file: ${hydra:runtime.output_dir}/test
            output_fields_from_original_dataset: [total_energy, forces]
            chemical_symbols: ${chemical_symbols}
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prefix = "test"

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
        self._batch_end(
            trainer=trainer,
            outputs=outputs,
            batch=batch,
            dataloader_idx=dataloader_idx,
        )


class ValTimeXYZFileWriter(XYZFileWriter):
    """XYZFileWriter designed for saving Val Time Predictions

    Users must provide an ``out_file`` that does not contain an extension. The actual output file will take
    the form ``{out_file}_dataset{idx}[_epoch{epoch}].xyz`` where ``idx`` is the dataset index (would be ``0`` for a single
    validation set but varies depending on number of validation sets) and ``epoch`` is the epoch when the file is produced.

    To incorporate original dataset fields in the ``xyz`` file to simplify analysis, users may provide
    ``output_fields_from_original_dataset``. Such fields will have the prefix ``original_dataset_`` in the ``xyz`` file.

    To obtain correct chemical species information, users must provide ``chemical_species`` in an order consistent with
    the model's ``type_names``.

    To activate the option to save to a different file every epoch, users should set ``separate_file_per_epoch`` true.

    Args:
        out_file (str): path to output file (must NOT contain ``.xyz`` or ``.extxyz`` extension)
        output_fields_from_original_dataset (List[str]): values from the original dataset to save in the ``out_file``
        extra_fields (List[str]): extra fields to save in addition to ASE's default fields
        chemical_species (List[str]): chemical species in the same order as model's ``type_names``
        separate_file_per_epoch (bool): if True, write outputs to a separate file per epoch (Useful for ``Train`` run types with ValTimeXYZFileWriter)
        every_n_epochs (int): if nonzero, only call on epoch multiples of this variable

    Example usage in config to write predictions and original dataset ``total_energy`` and ``forces`` to an ``xyz`` file:

    .. code-block:: yaml

        callbacks:
          - _target_: nequip.train.callbacks.ValTimeXYZFileWriter
            out_file: ${hydra:runtime.output_dir}/val
            output_fields_from_original_dataset: [total_energy, forces]
            chemical_symbols: ${chemical_symbols}
            separate_file_per_epoch: true
            every_n_epochs: 5
    """

    def __init__(
        self,
        separate_file_per_epoch: bool = False,
        every_n_epochs: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prefix = "val"

        if every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be > 0")
        self.every_n_epochs = every_n_epochs
        self.separate_file_per_epoch = separate_file_per_epoch

    def on_validation_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        outputs: Dict[str, Union[torch.Tensor, AtomicDataDict.Type]],
        batch: AtomicDataDict.Type,
        batch_idx: int,
        dataloader_idx=0,
    ):
        """"""
        if not (trainer.current_epoch % self.every_n_epochs):
            self._batch_end(
                trainer=trainer,
                outputs=outputs,
                batch=batch,
                dataloader_idx=dataloader_idx,
            )
