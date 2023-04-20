from typing import Optional

from tqdm.auto import tqdm

import torch

from nequip.nn import GraphModel, SequentialGraphNetwork
from nequip.nn import (
    GaussianMixtureModelUncertainty as GaussianMixtureModelUncertaintyModule,
)
from nequip.data import AtomicDataDict, AtomicData, AtomicDataset, Collater
from nequip.utils import find_first_of_type


def GaussianMixtureModelUncertainty(
    graph_model: GraphModel,
    config,
    deploy: bool,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
    feature_field: str = AtomicDataDict.NODE_FEATURES_KEY,
    out_field: Optional[str] = None,
):
    r"""Use a GMM on some latent features to predict an uncertainty.

    Only for deployment time!  See `configs/minimal_gmm.yaml`.
    """
    # it only makes sense to add or fit a GMM to a deployment model whose features are already trained
    if (not deploy) or initialize:
        raise RuntimeError(
            "GaussianMixtureModelUncertainty can only be used at deployment time, see `configs/minimal_gmm.yaml`."
        )

    # = add GMM =
    if out_field is None:
        out_field = feature_field + "_nll"

    # TODO: this is VERY brittle!!!!
    seqnn: SequentialGraphNetwork = find_first_of_type(
        graph_model, SequentialGraphNetwork
    )

    gmm: GaussianMixtureModelUncertaintyModule = seqnn.append_from_parameters(
        builder=GaussianMixtureModelUncertaintyModule,
        name=feature_field + "_gmm",
        shared_params=config,
        params=dict(feature_field=feature_field, out_field=out_field),
    )

    if dataset is None:
        raise RuntimeError(
            "GaussianMixtureModelUncertainty requires a dataset to fit the GMM on; did you specify `nequip-deploy --using-dataset`?"
        )

    # = evaluate features =
    # set up model
    prev_training: bool = graph_model.training
    prev_device: torch.device = graph_model.get_device()
    device = config.get("device", None)
    graph_model.eval()
    graph_model.to(device=device)
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
        # TODO: !! assumption that final value of feature_field is what the
        #          GMM gets is very brittle, should really be extracting it
        #          from the GMM module somehow... not sure how that works.
        #          give it a training mode and exfiltrate it through a buffer?
        #          it is correct, however, for NequIP and Allegro energy models
        features.append(
            graph_model(AtomicData.to_AtomicDataDict(batch.to(device=device)))[
                feature_field
            ]
            .detach()
            .to("cpu")  # offload to not run out of GPU RAM
        )
    features = torch.cat(features, dim=0)
    assert features.ndim == 2
    # restore model
    graph_model.train(mode=prev_training)
    graph_model.to(device=prev_device)
    # fit GMM
    gmm.fit(features)
    del features

    return graph_model
