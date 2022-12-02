from typing import Optional

from nequip.nn import SequentialGraphNetwork
from nequip.nn import (
    GaussianMixtureModelUncertainty as GaussianMixtureModelUncertaintyModule,
)
from nequip.data import AtomicDataDict, AtomicDataset
from nequip.utils import instantiate


def GaussianMixtureModelUncertainty(
    model: SequentialGraphNetwork,
    config,
    deploy: bool,
    dataset: Optional[AtomicDataset] = None,
    feature_field: str,
    out_field: str,
):
    r"""Use a GMM on some latent features to predict an uncertainty.

    Args:
        model

    Returns:
        SequentialGraphNetwork with the GMM added.
    """
    if not deploy:
        # it only makes sense to add or fit a GMM to a deployment model whose features are already trained
        return model
    gmm = instantiate(
        GaussianMixtureModelUncertaintyModule,
        prefix=out_field,
        args=dict(feature_field=feature_field, out_field=out_field),
        all_args=config,
    )
    # now fit the GMM
    # TODO have to figure out how to get the training dataset since deploy is true!
    # need to make training dataset available to `nequip-deploy`?
    # need to add a "tuning dataset"?  or that's basically what this is?
    # makes sense... when running `deploy` you might have
    # or is this the cue for `nequip-optimize`?? no, just an `--tune` flat to deploy
