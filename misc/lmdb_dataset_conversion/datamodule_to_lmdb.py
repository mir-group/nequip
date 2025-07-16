from nequip.data.dataset import NequIPLMDBDataset
from nequip.utils.global_state import set_global_state
from nequip.utils import RankedLogger
import os
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig

logger = RankedLogger(__name__, rank_zero_only=True)
os.environ["HYDRA_FULL_ERROR"] = "1"


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

    # === global state (important for float64 data) ===
    set_global_state(**OmegaConf.to_container(config.global_options, resolve=True))

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
                dloader_kwargs = getattr(datamodule, run + "_dataloader_config").copy()
                dloader_kwargs.update({"_target_": "torch.utils.data.DataLoader"})
                dloader_kwargs.update({"batch_size": 1})
                dloader = datamodule._get_dloader(
                    getattr(datamodule, run + "_dataset"),
                    datamodule.generator,
                    dloader_kwargs,
                )
                NequIPLMDBDataset.save_from_iterator(
                    file_path=f"{config.file_path}_{run}_{data_idx}",
                    iterator=tqdm(
                        dloader[data_idx],
                        total=len(getattr(datamodule, run + "_dataset")[data_idx]),
                    ),
                    **conversion_kwargs,
                )
        finally:
            datamodule.teardown(run_map[run])


if __name__ == "__main__":
    main()
