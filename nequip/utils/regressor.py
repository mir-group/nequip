import logging
import torch
from typing import Optional
from sklearn.linear_model import Ridge


def solver(X, y, alpha=0.1):
    y_mean = torch.sum(y) / torch.sum(X)
    feature_rms = 1.0 / torch.sqrt(torch.mean(X**2, axis=0))
    feature_rms = torch.nan_to_num(feature_rms, 1)

    clf = Ridge(alpha)
    clf.fit(X, y - (torch.sum(X, axis=1) * y_mean).reshape(y.shape))
    vec = torch.diag(torch.ones(X.shape[1]))
    mean = torch.as_tensor(clf.predict(vec), dtype=torch.get_default_dtype())
    return mean.reshape([-1]) + y_mean.reshape([-1]), None
