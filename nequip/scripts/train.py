# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""Train a network."""

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401
import torch

from ._workflow_utils import set_workflow_state
from nequip.utils import get_current_code_versions, RankedLogger
from nequip.utils.global_state import set_global_state, get_latest_global_state
from nequip.data.datamodule import NequIPDataModule
from nequip.train import NequIPLightningModule

from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate
import hydra
import os
from typing import Final, List

# pre-emptively set this env var to get the full stack trace for convenience
os.environ["HYDRA_FULL_ERROR"] = "1"
logger = RankedLogger(__name__, rank_zero_only=True)

_REQUIRED_CONFIG_SECTIONS: Final[List[str]] = [
    "run",
    "data",
    "trainer",
    "training_module",
    "global_options",
]


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def main(config: DictConfig) -> None:
    set_workflow_state("train")
    # === sanity checks ===

    # check that all base sections are present
    for section in _REQUIRED_CONFIG_SECTIONS:
        assert (
            section in config
        ), f"the `{section}` was not found in the config -- nequip config files must have the following section keys {_REQUIRED_CONFIG_SECTIONS}"

    assert "model" in config.training_module

    # determine run types
    assert (
        "run" in config
    ), "`run` must provided in the config -- it is a list that could include `train`, `val`, and/or `test`."
    if isinstance(config.run, ListConfig) or isinstance(config.run, list):
        runs = list(config.run)
    else:
        runs = [config.run]
    for run_type in runs:
        # don't have to be too safe for the `function` run type since it's advanced usage anyway
        assert (
            run_type in ["train", "val", "test"] or "function" in run_type.keys()
        ), f"`run` list can only contain `train`, `val`, or `test`, but found {run_type}"

    # ensure only single train at most, to protect restart and checkpointing logic later
    assert (
        sum([run_type == "train" for run_type in runs]) <= 1
    ), "only up to a single `train` instance can be present in `run`"

    # ensure that the relevant metrics are present
    if "train" in runs:
        assert (
            "loss" in config.training_module
        ), "`training_module.loss` must be provided in the config to perform a `train` run."
        assert (
            "val_metrics" in config.training_module
        ), "`training_module.val_metrics` must be provided in the config to perform a `train` run."
    if "val" in runs:
        assert (
            "val_metrics" in config.training_module
        ), "`training_module.val_metrics` must be provided in the config to perform a `train` run."
    if "test" in runs:
        assert (
            "test_metrics" in config.training_module
        ), "`training_module.test_metrics` must be provided in the config to perform a `test` run."

    versions = get_current_code_versions()

    logger.info(f"This `nequip-train` run will perform the following tasks: {runs}")
    logger.info(
        f"and use the output directory provided by Hydra: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    logger.debug("Setting global options ...")
    set_global_state(**OmegaConf.to_container(config.global_options, resolve=True))

    # === instantiate datamodule ===
    logger.info("Building datamodule ...")
    data = OmegaConf.to_container(config.data, resolve=True)
    datamodule = instantiate(data, _recursive_=False)
    assert isinstance(datamodule, NequIPDataModule)

    # === instantiate Lightning.Trainer ===
    trainer_cfg = OmegaConf.to_container(config.trainer, resolve=True)
    # enforce inference_mode=False to enable grad during inference
    # see https://lightning.ai/docs/pytorch/stable/common/trainer.html#inference-mode
    if "inference_mode" in trainer_cfg:
        raise ValueError(
            "`inference_mode` found in train.trainer in the config -- users shouldn't set this. NequIP will set `inference_mode=False`."
        )
    trainer = instantiate(trainer_cfg, inference_mode=False)

    # === instantiate NequIPLightningModule (including model) ===
    training_module = hydra.utils.get_class(config.training_module._target_)
    assert issubclass(training_module, NequIPLightningModule)

    # assemble various dicts to be saved by the NequIPLightningModule
    info_dict = {
        "versions": versions,
        "data": data,
        "trainer": trainer_cfg,
        "global_options": get_latest_global_state(),
        "runs": runs,
    }

    # `run_index` is used to restore the run stage from a checkpoint if restarting from one
    run_index = 0
    if "ckpt_path" in config:

        logger.info(
            f"`training_module` (and `model`) from the checkpoint file `{config.ckpt_path}` will be used, and the `training_module` details from the config used for this restart will be ignored. If you wish to alter training hyperparameters or anything in the config, start a new training run and use the `ModelFromCheckpoint` builder to load a pretrained model instead. There can be various obscure errors that arise during checkpoint state restoration when restarting with an altered config file."
        )

        # === instantiate from checkpoint file ===
        # dataset statistics need not be recalculated
        logger.info(
            f"Building `training_module` from checkpoint file {config.ckpt_path} ..."
        )
        # only the original model's config is used (with the dataset statistics already-computed)
        # everything else can be overriden
        # see https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html

        # === load checkpoint ===
        # for options see https://pytorch.org/docs/stable/generated/torch.load.html
        checkpoint = torch.load(
            config.ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        # check that runs from checkpoint match runs from config
        ckpt_runs = checkpoint["hyper_parameters"]["info_dict"]["runs"]
        # only check up to `len(ckpt_runs)` to allow for additional run stages added after
        assert all(
            [runs[idx] == ckpt_runs[idx] for idx in range(len(ckpt_runs))]
        ), f"`run` from checkpoint  must match the `run` from the config up to the length of the checkpoint `run` list, but mismatch found -- checkpoint: `{ckpt_runs}`, config: `{runs}`"

        # get run index
        # "run_stage" is registered as a buffer in NequIPLightningModule to preserve run state
        run_index = checkpoint["state_dict"]["run_stage"].item()

        # === load lightning module ===
        training_module = hydra.utils.get_class(
            checkpoint["hyper_parameters"]["info_dict"]["training_module"]["_target_"]
        )
        # propagate info dict
        info_dict.update(
            {
                "training_module": checkpoint["hyper_parameters"]["info_dict"][
                    "training_module"
                ]
            }
        )
        # TODO: consider allowing override of compile mode when continuing interrupted runs
        nequip_module = training_module.load_from_checkpoint(
            config.ckpt_path,
            num_datasets=datamodule.num_datasets,
            info_dict=info_dict,
        )
    else:
        # === compute dataset statistics use resolver to get dataset statistics to model config ===
        stats_dict = datamodule.get_statistics(dataset="train")

        def training_data_stats(stat_name: str):
            stat = stats_dict.get(stat_name, None)

            if stat is None:
                raise RuntimeError(
                    f"Data statistics field `{stat_name}` was requested for use in model initialization, but was not computed -- users must explicitly configure its computation with the `stats_manager` DataModule argument."
                )
            return stat

        OmegaConf.register_new_resolver(
            "training_data_stats",
            training_data_stats,
            use_cache=True,
        )
        # resolve dataset statistics among other params
        nequip_module_cfg = OmegaConf.to_container(config.training_module, resolve=True)
        info_dict.update({"training_module": nequip_module_cfg})

        # === instantiate training module ===
        logger.info("Building model and training_module from scratch ...")

        nequip_module = instantiate(
            nequip_module_cfg,
            # ensure lazy instantiation of lightning module attributes
            _recursive_=False,
            # make everything Python primitives (no DictConfig/ListConfig)
            _convert_="all",
            num_datasets=datamodule.num_datasets,
            info_dict=info_dict,
        )

    # pass world size from trainer to NequIPLightningModule
    nequip_module.world_size = trainer.world_size

    # === loop of run types ===
    # restart behavior is such that
    # - train from ckpt uses the correct ckpt file to restore training state (so it is given a specific `ckpt_path`)
    # - val/test/predict from ckpt would use the `nequip_module` from the ckpt (and so uses `ckpt_path=None`)
    # - if we train, then val/test/predict, we set `ckpt_path="best"` after training so val/test/predict tasks after that will use the "best" model
    ckpt_path = None
    while run_index < len(runs):
        run_type = runs[run_index]
        if run_type == "train":
            ckpt_path = config.get("ckpt_path", None)
            logger.info("TRAIN RUN START")
            trainer.fit(nequip_module, datamodule=datamodule, ckpt_path=ckpt_path)
            ckpt_path = "best"
            logger.info("TRAIN RUN END")
        elif run_type == "val":
            logger.info("VAL RUN START")
            trainer.validate(nequip_module, datamodule=datamodule, ckpt_path=ckpt_path)
            logger.info("VAL RUN END")
        elif run_type == "test":
            logger.info("TEST RUN START")
            trainer.test(nequip_module, datamodule=datamodule, ckpt_path=ckpt_path)
            logger.info("TEST RUN END")
        elif run_type == "predict":
            # TODO: `predict` run type is hidden from users, so we should never go through this path
            # potentially remove eventually if there's no need for a `predict` functionality, but it's here for now just in case
            logger.info("PREDICT RUN START")
            trainer.predict(nequip_module, datamodule=datamodule, ckpt_path=ckpt_path)
            logger.info("PREDICT RUN END")
        else:
            # TODO: make sure we load the correct `best` checkpoint model state dict before using `function`
            # TODO: make sure the model is set up to perform evaluations (e.g. EMA model is tricky)
            # for now this is unsupported
            if "function" in run_type:
                assert NotImplementedError(
                    "`function` run type is not ready for use yet"
                )
            instantiate(run_type["function"], training_module=nequip_module)
        # update `run_index` and update it in `nequip_module`'s state dict
        run_index += 1
        nequip_module.run_stage[0] = run_index

    set_workflow_state(None)
    return


if __name__ == "__main__":
    main()
