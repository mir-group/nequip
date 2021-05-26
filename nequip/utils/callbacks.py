import torch
import logging
from math import cos, pi


def equal_loss(self):

    loss_f = self.mae_dict["Validation_loss_f"]
    loss_e = self.mae_dict["Validation_loss_e"]
    coeff_e = loss_f / loss_e
    self.loss.coeffs["forces"] = torch.as_tensor(1, dtype=torch.get_default_dtype())
    self.loss.coeffs["total_energy"] = torch.as_tensor(
        loss_f / loss_e, dtype=torch.get_default_dtype()
    )
    self.logger.info(f"# update loss coeffs to 1 and {loss_f/loss_e}")


def cos_sin(self):

    f = self.kwargs.get("loss_f_mag", 1)
    e = self.kwargs.get("loss_e_mag", 1)
    phi_f = self.kwargs.get("loss_f_phi", 0)
    phi_e = self.kwargs.get("loss_e_phi", -10)
    cycle = self.kwargs.get("loss_coeff_cycle", 20)

    if phi_f == phi_e:

        return

    dtype = torch.get_default_dtype()

    f = torch.as_tensor(f * (cos((self.iepoch + phi_f) / cycle * pi)+1), dtype=dtype)
    e = torch.as_tensor(e * (cos((self.iepoch + phi_e) / cycle * pi)+1), dtype=dtype)

    self.loss.coeffs["forces"] = f
    self.loss.coeffs["total_energy"] = e

    self.logger.info(f"# update loss coeffs to {f} {e}")
