from typing import Optional

from tqdm.auto import tqdm

import torch

from e3nn.util._argtools import _get_device

from nequip.nn import SequentialGraphNetwork
from nequip.nn import (
    GaussianMixtureModelUncertainty as GaussianMixtureModelUncertaintyModule,
)
from nequip.data import AtomicDataDict, AtomicData, AtomicDataset, Collater


def GaussianMixtureModelUncertainty(
    model: SequentialGraphNetwork,
    config,
    deploy: bool,
    dataset: Optional[AtomicDataset] = None,
    feature_field: str = AtomicDataDict.NODE_FEATURES_KEY,
    out_field: Optional[str] = None,
):
    r"""Use a GMM on some latent features to predict an uncertainty.

    Args:
        model

    Returns:
        SequentialGraphNetwork with the GMM added.
    """
    # = add GMM =
    if out_field is None:
        out_field = feature_field + "_nll"

    gmm: GaussianMixtureModelUncertaintyModule = model.append_from_parameters(
        builder=GaussianMixtureModelUncertaintyModule,
        name=feature_field + "_gmm",
        shared_params=config,
        params=dict(feature_field=feature_field, out_field=out_field),
    )

    if deploy:
        # it only makes sense to add or fit a GMM to a deployment model whose features are already trained

        if dataset is None:
            raise RuntimeError(
                "GaussianMixtureModelUncertainty requires a dataset to fit the GMM on; did you specify `nequip-deploy --using-dataset`?"
            )

        # = evaluate features =
        # set up model
        prev_training: bool = model.training
        prev_device: torch.device = _get_device(model)
        device = config.get("device", None)
        model.eval()
        model.to(device=device)
        # evaluate
        features = []
        collater = Collater.for_dataset(dataset=dataset)
        batch_size: int = config.get(
            "validation_batch_size", config.batch_size
        )  # TODO: better default?
        stride: int = config.get("dataset_statistics_stride", 1)
        # TODO: guard TQDM on interactive?
        for batch_start_i in tqdm(
            range(0, len(dataset), stride * batch_size),
            desc="GMM eval features on train set",
        ):
            batch = collater(
                [dataset[batch_start_i + i * stride] for i in range(batch_size)]
            )
            features.append(
                model(AtomicData.to_AtomicDataDict(batch.to(device=device)))[
                    feature_field
                ]
                .detach()
                .to("cpu")  # offload to not run out of GPU RAM
            )
        features = torch.cat(features, dim=0)
        assert features.ndim == 2
        # restore model
        model.train(mode=prev_training)
        model.to(device=prev_device)
        # fit GMM
        gmm.fit(features)
        del features

    return model
