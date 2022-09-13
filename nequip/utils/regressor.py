import logging
import torch
from typing import Optional
from sklearn.linear_model import Ridge


def solver(X, y, alpha=0.1, stride=1, **kwargs):

    dtype = torch.get_default_dtype()
    X = X[::stride].to(dtype)
    y = y[::stride].to(dtype)

    y_mean = torch.sum(y) / torch.sum(X)
    # feature_rms = 1.0 / torch.sqrt(torch.mean(X**2, axis=0))
    # feature_rms = torch.nan_to_num(feature_rms, 1)

    clf = Ridge(alpha, fit_intercept=False, **kwargs)
    clf.fit(X, y - (torch.sum(X, axis=1, keepdim=True) * y_mean).reshape(y.shape))
    vec = torch.diag(torch.ones(X.shape[1]))
    mean = torch.as_tensor(clf.predict(vec), dtype=dtype)

    mean = mean.reshape([-1]) + y_mean.reshape([-1])

    return mean, None
