# FAQ

## Loss functions

 - Despite changing the coefficients in `loss_coeffs`, the magnitude of my training loss isn't changing!

   Inidividual loss terms like `training_loss_f`, `training_loss_e`, etc. are reported **before** they are scaled by their coefficients for summing into the total loss.
   
