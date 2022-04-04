import logging
import torch
import numpy as np
from typing import Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Kernel, Hyperparameter


def solver(X, y, regressor: Optional[str] = "NormalizedGaussianProcess", **kwargs):
    if regressor == "GaussianProcess":
        return gp(X, y, **kwargs)
    elif regressor == "NormalizedGaussianProcess":
        return normalized_gp(X, y, **kwargs)
    else:
        raise NotImplementedError(f"{regressor} is not implemented")


def normalized_gp(X, y, **kwargs):
    feature_rms = 1.0 / np.sqrt(np.average(X**2, axis=0))
    feature_rms = np.nan_to_num(feature_rms, 1)
    y_mean = torch.sum(y) / torch.sum(X)
    mean, std = base_gp(
        X,
        y - (torch.sum(X, axis=1) * y_mean).reshape(y.shape),
        NormalizedDotProduct,
        {"diagonal_elements": feature_rms},
        **kwargs,
    )
    return mean + y_mean, std


def gp(X, y, **kwargs):
    return base_gp(
        X, y, DotProduct, {"sigma_0": 0, "sigma_0_bounds": "fixed"}, **kwargs
    )


def base_gp(
    X,
    y,
    kernel,
    kernel_kwargs,
    alpha: Optional[float] = 0.1,
    max_iteration: int = 20,
    stride: Optional[int] = None,
):

    if len(y.shape) == 1:
        y = y.reshape([-1, 1])

    if stride is not None:
        X = X[::stride]
        y = y[::stride]

    not_fit = True
    iteration = 0
    mean = None
    std = None
    while not_fit:
        logging.debug(f"GP fitting iteration {iteration} {alpha}")
        try:
            _kernel = kernel(**kernel_kwargs)
            gpr = GaussianProcessRegressor(kernel=_kernel, random_state=0, alpha=alpha)
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
                logging.info("try a non-zero alpha")
                not_fit = False
                raise ValueError(
                    f"Please set the {alpha} to non-zero value. \n"
                    "The dataset energy is rank deficient to be solved with GP"
                )
            else:
                alpha = alpha * 2
                iteration += 1
                logging.debug(f"           increase alpha to {alpha}")

            if iteration >= max_iteration or not_fit is False:
                raise ValueError(
                    "Please set the per species shift and scale to zeros and ones. \n"
                    "The dataset energy is to diverge to be solved with GP"
                )

    return mean, std


class NormalizedDotProduct(Kernel):
    r"""Dot-Product kernel.
    .. math::
        k(x_i, x_j) = x_i \cdot A \cdot x_j
    """

    def __init__(self, diagonal_elements):
        # TO DO: check shape
        self.diagonal_elements = diagonal_elements
        self.A = np.diag(diagonal_elements)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            K = (X.dot(self.A)).dot(X.T)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            K = (X.dot(self.A)).dot(Y.T)

        if eval_gradient:
            return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y).
        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X).
        """
        return np.einsum("ij,ij,jj->i", X, X, self.A)

    def __repr__(self):
        return ""

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    @property
    def hyperparameter_diagonal_elements(self):
        return Hyperparameter("diagonal_elements", "numeric", "fixed")
