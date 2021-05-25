import torch
import logging


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
    pi = self.kwargs.get("loss_coeff_pi", 20)

    dtype = torch.get_default_dtype()

    f = torch.as_tensor(f, dtype=dtype)
    e = torch.as_tensor(e, dtype=dtype)

    f = f * torch.sin(torch.as_tensor(self.iepoch / pi, dtype=dtype))
    e = e * torch.cos(torch.as_tensor(self.iepoch / pi, dtype=dtype))

    self.loss.coeffs["forces"] = f
    self.loss.coeffs["total_energy"] = e

    self.logger.info(f"# update loss coeffs to {f} {e}")
