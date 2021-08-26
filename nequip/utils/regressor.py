import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct


def gp(X, y, alpha):

    print("alpha", alpha)

    kernel = DotProduct(sigma_0=0, sigma_0_bounds="fixed")
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=alpha).fit(X, y)

    vec = torch.diag(torch.ones(X.shape[1]))
    mean, std = gpr.predict(vec, return_std=True)

    mean = torch.as_tensor(mean, dtype=torch.get_default_dtype()).reshape([-1])
    # ignore all the off-diagonal terms
    std = torch.as_tensor(std, dtype=torch.get_default_dtype()).reshape([-1])

    return mean, std
