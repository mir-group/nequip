import logging
import torch
from sklearn.linear_model import Ridge


# def sklearn_ridge_regression(N, arr, algorithm_kwargs):
# 
#     n = N.shape[0]
#     num_types = N.shape[1]
#     fit_intercept = algorithm_kwargs.pop("fit_intercept", False)
#     ridge = Ridge(fit_intercept=False, **algorithm_kwargs)
#     ridge.fit(N, arr)
#     mean = torch.as_tensor(ridge.coef_, dtype=torch.get_default_dtype()).reshape([-1])
# 
#     res2 = (torch.square(torch.matmul(N, mean) - arr)).sum()
# 
#     print(res2)
#     mean, std = ridge_regression(N, arr, 4)
#     res2 = (torch.square(torch.matmul(N, mean) - arr)).mean()
#     print(res2)


def ridge_regression(N, arr, sigma):

    n = N.shape[0]
    num_types = N.shape[1]
    NT = torch.transpose(N, 1, 0)

    try:
        invNTN = torch.inverse(torch.matmul(NT, N))
    except Exception as e:
        if sigma is None:
            raise RuntimeError(
                f"Cannot get a solution {e} with sigma is None."
                "set PerSpeciesShiftScale_sigma to i.e. 0.2"
            )
        else:
            logging.warning(
                f"using ridge regression to compute per species weight."
                "Note that the result greatly depends on the sigma value"
            )
            NTN = torch.matmul(NT, N)
            if sigma <= 0:
                raise ValueError("sigma has to be > 0.")
            NTN += sigma * torch.diag(torch.ones(NTN.shape[0]))
            invNTN = torch.inverse(NTN)

    invN = torch.matmul(invNTN, NT)
    mean = torch.matmul(invN, arr)

    res2 = (torch.square(torch.matmul(N, mean) - arr)).sum()
    if n < num_types:
        cov = res2 * invNTN / n
    else:
        cov = res2 * invNTN / (n - num_types)
    std = torch.sqrt(torch.diagonal(cov))

    return mean.reshape([-1]), std.reshape([-1])
