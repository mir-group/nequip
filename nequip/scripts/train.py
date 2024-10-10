""" Train a network."""

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

from nequip.utils.versions import check_code_version
from nequip.utils._global_options import _set_global_options, _get_latest_global_options
from nequip.utils.logger import RankedLogger
from nequip.data.datamodule import NequIPDataModule
from nequip.train import NequIPLightningModule

from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate
import hydra

import os

logger = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def main(config: DictConfig) -> None:
    # === sanity checks ===

    # determine run types
    assert (
        "run" in config
    ), "`run` must provided in the config -- it is a list that could include `train`, `val`, `test`, and/or `predict`."
    if isinstance(config.run, ListConfig) or isinstance(config.run, list):
        runs = config.run
    else:
        runs = [config.run]
    assert all([run_type in ["train", "val", "test", "predict"] for run_type in runs])

    # ensure only single train at most, to protect restart and checkpointing logic later
    assert (
        sum([run_type == "train" for run_type in runs]) == 1
    ), "only a single `train` instance can be present in `run`"

    if "train" in runs:
        assert (
            "loss" in config.train
        ), "`train.loss` must be provided in the config to perform a `train` run."
        assert (
            "val_metrics" in config.train
        ), "`train.val_metrics` must be provided in the config to perform a `train` run."
    if "val" in runs:
        assert (
            "val_metrics" in config.train
        ), "`train.val_metrics` must be provided in the config to perform a `train` run."
    if "test" in runs:
        assert (
            "test_metrics" in config.train
        ), "`train.test_metrics` must be provided in the config to perform a `test` run."

    logger.info(f"This `nequip-train` run will perform the following tasks: {runs}")
    logger.info(
        f"and use the output directory provided by Hydra: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    versions, commits = check_code_version(config)
    logger.debug("Setting global options ...")
    _set_global_options(**OmegaConf.to_container(config.global_options, resolve=True))

    # === instantiate datamodule ===
    logger.info("Building datamodule ...")

    # == silently include type_names in stats_manager if present ==
    assert "type_names" in config.model
    data = OmegaConf.to_container(config.data, resolve=True)
    if "stats_manager" in data:
        data["stats_manager"]["type_names"] = config.model.type_names
    datamodule = instantiate(data, _recursive_=False)
    assert isinstance(datamodule, NequIPDataModule)

    # get training module
    training_module = hydra.utils.get_class(config.train.training_module)
    assert issubclass(training_module, NequIPLightningModule)

    # get nequip_module config args and convert to pure Python dicts for wandb logging
    loss_cfg = OmegaConf.to_container(
        config.train.get("loss", DictConfig(None)), resolve=True
    )
    val_metrics_cfg = OmegaConf.to_container(
        config.train.get("val_metrics", DictConfig(None)), resolve=True
    )
    train_metrics_cfg = OmegaConf.to_container(
        config.train.get("train_metrics", DictConfig(None)), resolve=True
    )
    test_metrics_cfg = OmegaConf.to_container(
        config.train.get("test_metrics", DictConfig(None)), resolve=True
    )
    optimizer_cfg = OmegaConf.to_container(
        config.train.get("optimizer", DictConfig(None)), resolve=True
    )
    lr_scheduler_cfg = OmegaConf.to_container(
        config.train.get("lr_scheduler", DictConfig(None)), resolve=True
    )

    # the trainer config is not actually used by the nequip_module, but is just used for logging purposes
    trainer_cfg = OmegaConf.to_container(config.train.trainer, resolve=True)

    # assemble versions, global options and trainer dicts to be saved by the NequIPLightningModule
    info_dict = {
        "versions": versions,
        "trainer": trainer_cfg,
        "global_options": _get_latest_global_options(),
        "training_module": str(config.train.training_module),
    }

    if "ckpt_path" in config:
        # === instantiate from checkpoint file ===
        # dataset statistics need not be recalculated
        logger.info(
            f"Building model and training_module from checkpoint file {config.ckpt_path} ..."
        )
        # only the original model's config is used (with the dataset statistics already-computed)
        # everything else can be overriden
        # see https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
        nequip_module = training_module.load_from_checkpoint(
            config.ckpt_path,
            strict=False,
            loss_cfg=loss_cfg,
            val_metrics_cfg=val_metrics_cfg,
            train_metrics_cfg=train_metrics_cfg,
            test_metrics_cfg=test_metrics_cfg,
            optimizer_cfg=optimizer_cfg,
            lr_scheduler_cfg=lr_scheduler_cfg,
            num_datasets=datamodule.num_datasets,
            info_dict=info_dict,
        )
        # `strict=False` above and the next line required to override metrics, etc
        nequip_module.strict_loading = False
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
        # https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#omegaconf-to-container
        model_config = OmegaConf.to_container(config.model, resolve=True)

        # === instantiate training module ===
        logger.info("Building model and training_module from scratch ...")
        nequip_module = training_module(
            model_cfg=model_config,
            loss_cfg=loss_cfg,
            val_metrics_cfg=val_metrics_cfg,
            train_metrics_cfg=train_metrics_cfg,
            test_metrics_cfg=test_metrics_cfg,
            optimizer_cfg=optimizer_cfg,
            lr_scheduler_cfg=lr_scheduler_cfg,
            num_datasets=datamodule.num_datasets,
            info_dict=info_dict,
        )

    # === instantiate Lightning.Trainer ===
    # enforce inference_mode=False to enable grad during inference
    # see https://lightning.ai/docs/pytorch/stable/common/trainer.html#inference-mode
    if "inference_mode" in config.train.trainer:
        raise ValueError(
            "`inference_mode` found in train.trainer in the config -- users shouldn't set this. NequIP will set `inference_mode=False`."
        )
    trainer = instantiate(trainer_cfg, inference_mode=False)

    # === loop of run types ===
    # restart behavior is such that
    # - train from ckpt uses the correct ckpt file to restore training state (so it is given a specific `ckpt_path`)
    # - val/test/predict from ckpt would use the `nequip_module` from the ckpt (and so uses `ckpt_path=None`)
    # - if we train, then val/test/predict, we set `ckpt_path="best"` after training so val/test/predict tasks after that will use the "best" model
    ckpt_path = None
    for run_type in runs:
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
            logger.info("PREDICT RUN START")
            trainer.predict(nequip_module, datamodule=datamodule, ckpt_path=ckpt_path)
            logger.info("PREDICT RUN END")
    return


if __name__ == "__main__":
    main()
