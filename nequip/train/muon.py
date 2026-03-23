# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
# This Muon implementation is based on the original Keller Jordan GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py

import torch


# This function is adapted from Muon (https://github.com/KellerJordan/Muon/).
# Original source: https://github.com/KellerJordan/Muon/blob/master/muon.py#L5.
# Changes: none.
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# This function is adapted from Muon (https://github.com/KellerJordan/Muon/).
# Original source: https://github.com/KellerJordan/Muon/blob/master/muon.py#L34.
# Changes: added `e3nn_reshaping` conditional to handle  e3nn ``Linear`` layer weights,
# which reshapes the weights to 2D based on slicing information from e3nn.
def muon_update(
    grad, momentum, beta=0.95, ns_steps=5, nesterov=True, e3nn_reshaping=None
):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)

    # NequIP addon: handle reshaping for e3nn ``Linear`` layers
    if e3nn_reshaping is not None:
        update_list = []
        for index_slice, shape_2D in e3nn_reshaping:  # square weight slices of updates
            weight_slice = update[index_slice].reshape(shape_2D)
            grad_slice = grad[index_slice].reshape(shape_2D)
            update_weight_slice = zeropower_via_newtonschulz5(
                weight_slice, steps=ns_steps
            )
            update_weight_slice *= (
                max(1, grad_slice.size(-2) / grad_slice.size(-1)) ** 0.5
            )
            # Flatten the weight slice
            update_list.append(update_weight_slice.flatten())
        # concatenate updates back into single vector
        update = torch.cat(update_list, dim=-1)
    else:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
        update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


# This function is adapted from Muon (https://github.com/KellerJordan/Muon/).
# Original source: https://github.com/KellerJordan/Muon/blob/master/muon.py#L130.
# Changes: none.
def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


# This function is adapted from Muon (https://github.com/KellerJordan/Muon/).
# Original source: https://github.com/KellerJordan/Muon/blob/master/muon.py#L138.
# Changes: added `e3nn_reshaping` element in parameter group keys for e3nn ``Linear``
# layer weights.
class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam (Originally SingleDeviceMuonWithAuxAdam).

    Config example with NequIP parameter groups:

    .. code-block:: yaml

        _target_: nequip.train.MuonWithAuxAdam
        param_groups:
          _target_: nequip.model.MuonParamGroups
          muon:
            lr: 0.01
            weight_decay: 1e-5
          adam:
            lr: 0.01
            weight_decay: 1e-5
    """

    def __init__(self, params):
        for group in params:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "momentum",
                        "weight_decay",
                        "use_muon",
                        "e3nn_reshaping",
                    ]
                )
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "betas",
                        "eps",
                        "weight_decay",
                        "use_muon",
                    ]
                )
        super().__init__(params, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for i, p in enumerate(group["params"]):
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        e3nn_reshaping=group["e3nn_reshaping"].get(i, None),
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
