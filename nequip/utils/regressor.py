import logging
import torch
from typing import Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct


def solver(
    X,
    y,
    alpha=0.1,
    max_iteration: Optional[int] = 20,
    regressor: Optional[str] = "GaussianProcess",
):

    if regressor == "GaussianProcess":

        return gp(X, y, alpha, max_iteration)


def gp(X, y, alpha, max_iteration: int = 20):

    if len(y.shape) == 1:
        y = y.reshape([-1, 1])

    not_fit = True
    iteration = 0
    mean = None
    std = None
    while not_fit:
        logging.debug("GP fitting iteration", iteration, alpha)
        try:
            kernel = DotProduct(sigma_0=0, sigma_0_bounds="fixed")
            gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=alpha)
            gpr = gpr.fit(X, y)

            vec = torch.diag(torch.ones(X.shape[1]))
            mean, std = gpr.predict(vec, return_std=True)

            mean = torch.as_tensor(mean, dtype=torch.get_default_dtype()).reshape([-1])
            # ignore all the off-diagonal terms
            std = torch.as_tensor(std, dtype=torch.get_default_dtype()).reshape([-1])
            likelihood = gpr.log_marginal_likelihood()

            res = torch.sqrt(
                torch.square(torch.matmul(X, mean.reshape([-1, 1])) - y).mean()
            )

            logging.debug(
                f"GP fitting: alpha {alpha}:\n"
                f"            residue {res}\n"
                f"            mean {mean} std {std}\n"
                f"            log marginal likelihood {likelihood}"
            )
            not_fit = False

        except Exception as e:
            logging.info(f"GP fitting failed for alpha={alpha} and {e.args}")
            if alpha == 0 or alpha is None:
                logging.info(f"try a None zero alpha")
                not_fit = False
                raise ValueError(
                    f"Please set the {alpha} to none zero value. \n"
                    "The dataset energy is rank deficient to be solved with GP"
                )
            else:
                alpha = alpha * 2
                iteration += 1
                logging.debug(f"           increase alpha to {alpha}")

            if iteration >= max_iteration or not_fit == False:
                raise ValueError(
                    f"Please set the per species shift and scale to zeros and ones. \n"
                    "The dataset energy is to diverge to be solved with GP"
                )

    return mean, std
