import logging
import torch

from torch import matmul
from torch.linalg import solve, inv
from typing import Optional, Sequence
from opt_einsum import contract


def solver(X, y, alpha: Optional[float] = 0.001, stride: Optional[int] = 1, **kwargs):
    # results are in the same "units" as y, so same dtype too:
    dtype_out = y.dtype
    # always solve in float64 for numerical stability
    dtype = torch.float64
    X = X[::stride].to(dtype)
    y = y[::stride].to(dtype)

    X, y = down_sampling_by_composition(X, y)

    X_norm = torch.sum(X)

    X = X / X_norm
    y = y / X_norm

    y_mean = torch.sum(y) / torch.sum(X)

    feature_rms = torch.sqrt(torch.mean(X**2, axis=0))

    alpha_mat = torch.diag(feature_rms) * alpha * alpha

    A = matmul(X.T, X) + alpha_mat
    dy = y - (torch.sum(X, axis=1, keepdim=True) * y_mean).reshape(y.shape)
    Xy = matmul(X.T, dy)

    mean = solve(A, Xy)

    sigma2 = torch.var(matmul(X, mean) - dy)
    Ainv = inv(A)
    cov = torch.sqrt(sigma2 * contract("ij,kj,kl,li->i", Ainv, X, X, Ainv))

    mean = mean + y_mean.reshape([-1])

    logging.debug(f"Ridge Regression, residue {sigma2}")

    return mean.to(dtype_out), cov.to(dtype_out)


def down_sampling_by_composition(
    X: torch.Tensor, y: torch.Tensor, percentage: Sequence = [0.25, 0.5, 0.75]
):

    unique_comps, comp_ids = torch.unique(X, dim=0, return_inverse=True)

    n_types = torch.max(comp_ids) + 1

    sort_by = torch.argsort(comp_ids)

    # find out the block for each composition
    d_icomp = comp_ids[sort_by]
    d_icomp = d_icomp[:-1] - d_icomp[1:]
    node_icomp = torch.where(d_icomp != 0)[0]
    id_start = torch.cat((torch.as_tensor([0]), node_icomp + 1))
    id_end = torch.cat((node_icomp + 1, torch.as_tensor([len(sort_by)])))

    n_points = len(percentage)
    new_X = torch.zeros(
        (n_types * n_points, X.shape[1]), dtype=X.dtype, device=X.device
    )
    new_y = torch.zeros((n_types * n_points), dtype=y.dtype, device=y.device)
    for i in range(n_types):
        ids = sort_by[id_start[i] : id_end[i]]
        for j, p in enumerate(percentage):
            new_y[i * n_points + j] = torch.quantile(y[ids], p, interpolation="linear")
            new_X[i * n_points + j] = unique_comps[i]

    return new_X, new_y
