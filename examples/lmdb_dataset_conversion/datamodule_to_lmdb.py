from nequip.data.dataset import NequIPLMDBDataset
from nequip.utils._global_options import _set_global_options
from nequip.utils import RankedLogger
import os
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig

logger = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="data")
def main(config: DictConfig):

    # === determine run types ===
    assert (
        "run" in config
    ), "`run` must provided in the config -- it is a list that could include `train`, `val`, `test`, and/or `predict`."
    if isinstance(config.run, ListConfig) or isinstance(config.run, list):
        runs = config.run
    else:
        runs = [config.run]
    assert all([run_type in ["train", "val", "test", "predict"] for run_type in runs])

    # == clean runs ==
    runs = set(runs)
    if "train" in runs and "val" not in runs:
        runs.add("val")

    # == helper dict ==
    run_map = {
        "train": "fit",
        "val": "validate",
        "test": "test",
        "predict": "predict",
    }

    # === global options (important for float64 data) ===
    _set_global_options(**OmegaConf.to_container(config.global_options, resolve=True))

    # === instantiate and prepare datamodule ===
    datamodule = hydra.utils.instantiate(config.data, _recursive_=False)
    datamodule.prepare_data()

    conversion_kwargs = OmegaConf.to_container(config.lmdb_kwargs, resolve=True)

    # === perform conversion ===
    for run in runs:
        try:
            datamodule.setup(run_map[run])
            for data_idx in range(datamodule.num_datasets[run]):
                logger.info(
                    f"Constructing LMDB data file for {config.file_path}_{run}_{data_idx} ..."
                )
                dloader = datamodule._get_dloader(
                    getattr(datamodule, run + "_dataset"),
                    datamodule.generator,
                    {"batch_size": 1},
                )
                NequIPLMDBDataset.save_from_iterator(
                    file_path=f"{config.file_path}_{run}_{data_idx}",
                    iterator=dloader[data_idx],
                    **conversion_kwargs,
                )
        finally:
            datamodule.teardown(run_map[run])


if __name__ == "__main__":
    main()
