# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import tempfile
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

logger = RankedLogger(__name__, rank_zero_only=True)


@contextlib.contextmanager
def _get_model_file_path(input_path):
    """Context manager that provides a file path for both local and nequip.net models.

    For local files: yields the input path directly
    For nequip.net downloads: downloads to temp file and yields that path

    Args:
        input_path: path to the model checkpoint or package file, or nequip.net model ID
                   (format: nequip.net:group-name/model-name:version)

    Yields:
        pathlib.Path: Path to the model file (either original or temporary)
    """
    is_nequip_net_download: bool = str(input_path).startswith("nequip.net:")

    with (
        tempfile.NamedTemporaryFile(suffix=".nequip.zip")
        if is_nequip_net_download
        else contextlib.nullcontext()
    ) as tmpfile:
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

            # download the model package
            response = requests.get(model_info.artifact.download_url, stream=True)
            response.raise_for_status()

            # Get the total file size from headers
            total_size = int(response.headers.get("content-length", 0))

            # Create progress bar
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading from {model_info.artifact.host_name}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        tmpfile.write(chunk)
                        pbar.update(len(chunk))
            tmpfile.flush()
            logger.info("Download complete, loading model...")
            yield pathlib.Path(tmpfile.name)
            del model_info, model_id, response
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

    Args:
        input_path: path to the model checkpoint or package file, or nequip.net model ID
                   (format: nequip.net:group-name/model-name:version)
        compile_mode (str): compile mode for the model (default: _EAGER_MODEL_KEY)
        model_key (str): key to select the model from ModuleDict (default: _SOLE_MODEL_KEY)
        return_data_dict (bool): if True, also return the data dict for compilation (default: False)

    Returns:
        model: The loaded model
        data (optional): Data dict if return_data_dict=True, returned as tuple (model, data)
    """

    with _get_model_file_path(input_path) as actual_input_path:
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

        model = model[model_key]
        # ^ `ModuleDict` of `GraphModel` is loaded, we then select the desired `GraphModel` (`model_key` defaults to work for single model case)

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
