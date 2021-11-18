import sys
import torch
from math import cos, pi
from nequip.nn import PerSpeciesScaleShift, GradientOutput
from nequip.data import AtomicData, AtomicDataDict
from nequip.utils.batch_ops import bincount
from nequip.utils.regressor import solver
from nequip.utils import find_first_of_type

if sys.version_info[1] >= 7:
    import contextlib
else:
    # has backport of nullcontext
    import contextlib2 as contextlib


def equal_loss(trainer):

    loss_f = trainer.mae_dict["Validation_loss_f"]
    loss_e = trainer.mae_dict["Validation_loss_e"]
    # coeff_e = loss_f / loss_e
    trainer.loss.coeffs["forces"] = torch.as_tensor(1, dtype=torch.get_default_dtype())
    trainer.loss.coeffs["total_energy"] = torch.as_tensor(
        loss_f / loss_e, dtype=torch.get_default_dtype()
    )
    trainer.logger.info(f"# update loss coeffs to 1 and {loss_f/loss_e}")


def cos_sin(trainer):

    f = trainer.kwargs.get("loss_f_mag", 1)
    e = trainer.kwargs.get("loss_e_mag", 1)
    phi_f = trainer.kwargs.get("loss_f_phi", 0)
    phi_e = trainer.kwargs.get("loss_e_phi", -10)
    cycle = trainer.kwargs.get("loss_coeff_cycle", 20)

    if phi_f == phi_e:
        return

    dtype = torch.get_default_dtype()

    f = torch.as_tensor(
        f * (cos((trainer.iepoch + phi_f) / cycle * pi) + 1), dtype=dtype
    )
    e = torch.as_tensor(
        e * (cos((trainer.iepoch + phi_e) / cycle * pi) + 1), dtype=dtype
    )

    trainer.loss.coeffs["forces"] = f
    trainer.loss.coeffs["total_energy"] = e

    trainer.logger.info(f"# update loss coeffs to {f} {e}")


def linear_regression(trainer):
    """do a linear regration after training epoch"""

    per_species_rescale = find_first_of_type(trainer.model, PerSpeciesScaleShift)
    if per_species_rescale is None:
        return

    _key = AtomicDataDict.TOTAL_ENERGY_KEY
    if trainer.use_ema:
        cm = trainer.ema.average_parameters()
    else:
        cm = contextlib.nullcontext()

    num_types = per_species_rescale.num_types
    force_module = find_first_of_type(trainer.model, GradientOutput)
    if force_module is not None:
        force_module.skip = True

    dataset = trainer.dl_train
    trainer.n_batches = len(dataset)
    trainer.model.train()

    X = []
    y = []
    with cm:

        for trainer.ibatch, data in enumerate(dataset):

            # trainer.optim.zero_grad(set_to_none=True)

            # Do any target rescaling
            data = data.to(trainer.torch_device)
            data = AtomicData.to_AtomicDataDict(data)
            if hasattr(trainer.model, "unscale"):
                data_unscaled = trainer.model.unscale(data)
            else:
                data_unscaled = data

            input_data = data_unscaled.copy()
            out = trainer.model(input_data)

            atom_types = input_data[AtomicDataDict.ATOM_TYPE_KEY]
            N = bincount(
                atom_types,
                input_data[AtomicDataDict.BATCH_KEY],
                minlength=num_types,
            )

            # N = N[(N > 0).any(dim=1)]  # deal with non-contiguous batch indexes
            N = N.type(torch.get_default_dtype())
            res = data_unscaled[_key] - out[_key]

            X += [N]
            y += [res]

    with torch.no_grad():
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        mean, _ = solver(X, y)

        trainer.logger.info(f"residue shifts {mean}")
        trainer.delta_shifts = mean

    if force_module is not None:
        force_module.skip = False

    return mean


def update_rescales(trainer):

    if not hasattr(itrainer, "delta_shifts"):
        return

    per_species_rescale = find_first_of_type(trainer.model, PerSpeciesScaleShift)
    trainer.logger.info(f"update shifts from {per_species_rescale.shifts} ...")

    per_species_rescale.shifts = per_species_rescale.shifts + trainer.delta_shifts

    trainer.logger.info(f"                to {per_species_rescale.shifts} .")

def recover_rescales(trainer):

    if not hasattr(itrainer, "delta_shifts"):
        return

    per_species_rescale = find_first_of_type(trainer.model, PerSpeciesScaleShift)

    trainer.logger.info(f"update shifts from {per_species_rescale.shifts} ...")
    per_species_rescale.shifts = per_species_rescale.shifts - trainer.delta_shifts
    trainer.logger.info(f"                to {per_species_rescale.shifts} .")
