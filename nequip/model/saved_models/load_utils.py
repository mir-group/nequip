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


def _download_to_file(
    url: str, tmpfile: tempfile.NamedTemporaryFile, desc: str = "Downloading"
):
    """Download a file from a URL with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=desc,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                tmpfile.write(chunk)
                pbar.update(len(chunk))
    tmpfile.flush()

    del response


@contextlib.contextmanager
def _get_model_file_path(input_path):
    """Context manager that provides a file path for local files, URLs, and nequip.net models.

    For local files: yields the input path directly
    For URLs: downloads to temp file and yields that path
    For nequip.net downloads: downloads to temp file and yields that path

    Args:
        input_path: path to the model checkpoint or package file, URL, or nequip.net model ID
                   - Local file: any path that doesn't start with http://, https://, or nequip.net:
                   - URL: http://... or https://...
                   - nequip.net: nequip.net:group-name/model-name:version

    Yields:
        pathlib.Path: Path to the model file (either original or temporary)
    """
    input_str = str(input_path)
    is_nequip_net_download: bool = input_str.startswith("nequip.net:")
    is_url_download: bool = input_str.startswith(("http://", "https://"))
    needs_download: bool = is_nequip_net_download or is_url_download

    with (
        tempfile.NamedTemporaryFile(suffix=".nequip.zip")
        if needs_download
        else contextlib.nullcontext()
    ) as tmpfile:
        if is_nequip_net_download:
            model_id = input_str[len("nequip.net:") :]
            logger.info(f"Fetching {model_id} from nequip.net...")
            with model_repository.NequIPNetAPIClient() as client:
                model_info = client.get_model_download_info(model_id)

            if model_info.newer_version_id is not None:
                logger.info(
                    f"Model {model_id} has a newer version available: {model_info.newer_version_id}"
                )

            _download_to_file(
                model_info.artifact.download_url,
                tmpfile,
                desc=f"Downloading from {model_info.artifact.host_name}",
            )
            logger.info("Download complete, loading model...")
            yield pathlib.Path(tmpfile.name)
            del model_info, model_id
        elif is_url_download:
            logger.info(f"Downloading model from {input_str}...")
            _download_to_file(
                input_str, tmpfile, desc=f"Downloading model from {input_str}"
            )
            logger.info("Download complete, loading model...")
            yield pathlib.Path(tmpfile.name)
        else:
            logger.info(f"Loading model from {input_path} ...")
            yield pathlib.Path(input_path)


def load_saved_model(
    input_path,
    compile_mode: str = _EAGER_MODEL_KEY,
    model_key: str = _SOLE_MODEL_KEY,
    return_data_dict: bool = False,
):
    """Load a saved model from checkpoint, package, URL, or nequip.net.

    Args:
        input_path: path to the model checkpoint or package file, URL, or nequip.net model ID
                   - Local file: any path that doesn't start with http://, https://, or nequip.net:
                   - URL: http://... or https://...
                   - nequip.net: nequip.net:group-name/model-name:version
        compile_mode (str): compile mode for the model (default: _EAGER_MODEL_KEY)
        model_key (str): key to select the model from ModuleDict (default: _SOLE_MODEL_KEY)
        return_data_dict (bool): if True, also return the data dict for compilation (default: False)

    Returns:
        model: The loaded model
        data (optional): Data dict if return_data_dict=True, returned as tuple (model, data)
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
