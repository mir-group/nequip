# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import contextlib
import pathlib
import requests
from tqdm.auto import tqdm

from nequip.model.utils import _EAGER_MODEL_KEY
from nequip.model.saved_models import ModelFromPackage, ModelFromCheckpoint
from nequip.model.modify_utils import only_apply_persistent_modifiers
from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip.utils import model_repository
from nequip.utils.logger import RankedLogger
from nequip.utils.model_cache import get_cached_model, cache_model

logger = RankedLogger(__name__, rank_zero_only=True)


@contextlib.contextmanager
def _get_model_file_path(input_path):
    """Context manager that provides a file path for both local and nequip.net models.

    For local files: yields the input path directly
    For nequip.net downloads: uses cache if available, otherwise downloads and caches
    (default cache location: ``~/.nequip/model_cache``, configurable via ``NEQUIP_CACHE_DIR``)

    Args:
        input_path: path to the model checkpoint or package file, or nequip.net model ID
                   (format: ``nequip.net:group-name/model-name:version``)

    Yields:
        pathlib.Path: Path to the model file (either original or cached)
    """
    is_nequip_net_download: bool = str(input_path).startswith("nequip.net:")

    if is_nequip_net_download:
        # get model ID
        model_id = str(input_path)[len("nequip.net:") :]
        logger.info(f"Fetching {model_id} from nequip.net...")

        # get download URL
        with model_repository.NequIPNetAPIClient() as client:
            model_info = client.get_model_download_info(model_id)

        if model_info.newer_version_id is not None:
            logger.info(
                f"Model {model_id} has a newer version available: {model_info.newer_version_id}"
            )

        logger.info(
            "Please cite this model in any resulting work -- see "
            "https://nequip.readthedocs.io/en/latest/guide/getting-started/foundation_potentials.html"
        )

        download_url = model_info.artifact.download_url

        # check cache first
        cached_path = get_cached_model(model_id, download_url)
        if cached_path is not None:
            yield cached_path
            return

        # cache miss: download and cache
        def download_fn(target_path: pathlib.Path):
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(target_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading from {model_info.artifact.host_name}",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        # download and cache (cache_model will skip caching if NEQUIP_NO_CACHE is set)
        cached_path = cache_model(model_id, download_url, download_fn)
        logger.info("Download complete, loading model...")
        yield cached_path
    else:
        logger.info(f"Loading model from {input_path} ...")
        yield pathlib.Path(input_path)


def load_saved_model(
    input_path,
    compile_mode: str = _EAGER_MODEL_KEY,
    model_key: str = _SOLE_MODEL_KEY,
    return_data_dict: bool = False,
):
    """Load a saved model from checkpoint, package, or nequip.net.

    This function can load models from:

    - **Checkpoint files** (``.ckpt``): saved during training runs
    - **Package files** (``.nequip.zip``): created with ``nequip-package``
    - **nequip.net models**: using model ID format ``nequip.net:group-name/model-name:version`` from `nequip.net <https://www.nequip.net/>`__

    Args:
        input_path: path to the model checkpoint or package file, or nequip.net model ID
                   (format: ``nequip.net:group-name/model-name:version``)
        compile_mode (str): compile mode for the model (default: ``"eager"``)
        model_key (str): key to select the model from ModuleDict (default: ``"sole_model"``)
        return_data_dict (bool): if ``True``, also return the data dict for compilation (default: ``False``)

    Returns:
        torch.nn.Module or tuple: the loaded model, or ``(model, data)`` tuple if ``return_data_dict=True``
    """

    with _get_model_file_path(input_path) as actual_input_path:
        # check if the resolved file exists
        if not actual_input_path.exists():
            raise ValueError(
                f"Model file does not exist: {input_path} (resolved to: {actual_input_path})"
            )

        # use package load path if extension matches, otherwise assume checkpoint file
        use_ckpt = not str(actual_input_path).endswith(".nequip.zip")

        # load model
        if use_ckpt:
            # we only apply persistent modifiers when building from checkpoint
            # i.e. acceleration modifiers won't be applied, and have to be specified during compile time
            with only_apply_persistent_modifiers(persistent_only=True):
                model = ModelFromCheckpoint(
                    actual_input_path, compile_mode=compile_mode
                )
        else:
            # packaged models will never have non-persistent modifiers built in
            model = ModelFromPackage(actual_input_path, compile_mode=compile_mode)

        if model_key is not None:
            model = model[model_key]
            # ^ `ModuleDict` of `GraphModel` is loaded, we then select the desired `GraphModel` (`model_key` defaults to work for single model case)
            # otherwise, return the `ModuleDict`

        # load data dict if requested
        if return_data_dict:
            from nequip.model.saved_models.checkpoint import data_dict_from_checkpoint
            from nequip.model.saved_models.package import data_dict_from_package

            if use_ckpt:
                data = data_dict_from_checkpoint(str(actual_input_path))
            else:
                data = data_dict_from_package(str(actual_input_path))

            return model, data
        else:
            return model
