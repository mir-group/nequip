import logging
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct


def gp(X, y, alpha):

    kernel = DotProduct(sigma_0=0, sigma_0_bounds="fixed")
    print("alpha", alpha)
    alpha=1
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=alpha)
    gpr = gpr.fit(X, y)
    print(gpr.alpha_)

    vec = torch.diag(torch.ones(X.shape[1]))
    mean, std = gpr.predict(vec, return_std=True)

    mean = torch.as_tensor(mean, dtype=torch.get_default_dtype()).reshape([-1])
    std = torch.as_tensor(std, dtype=torch.get_default_dtype()).reshape([-1])

    res = torch.sqrt(torch.square(torch.matmul(X, mean.reshape([-1, 1]))-y).mean())
    logging.debug(f"GP fitting with alpha {alpha}: mean residue {res}")
    raise RuntimeError()

    return mean, std
