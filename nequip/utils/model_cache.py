# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import os
import json
import hashlib
import pathlib
import shutil
from datetime import datetime
from typing import Optional, Callable, Final
from nequip.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

_NEQUIP_NO_CACHE: Final[bool] = os.environ.get("NEQUIP_NO_CACHE", "").lower() in (
    "1",
    "true",
    "yes",
    "y",
)


def get_cache_dir() -> pathlib.Path:
    """Get the model cache directory from environment or default location."""
    cache_dir = os.environ.get("NEQUIP_CACHE_DIR")
    if cache_dir:
        path = pathlib.Path(cache_dir).expanduser().resolve()
    else:
        path = pathlib.Path.home() / ".nequip" / "model_cache"

    path.mkdir(parents=True, exist_ok=True)
    return path


def _compute_cache_key(model_id: Optional[str], download_url: str) -> str:
    """Compute cache key from model_id and download URL.

    For nequip.net models: hash(model_id + url) to account for version changes
    For arbitrary URLs: hash(url) only
    """
    if model_id:
        cache_input = f"{model_id}|{download_url}"
    else:
        cache_input = download_url

    return hashlib.sha256(cache_input.encode()).hexdigest()


def _compute_file_hash(file_path: pathlib.Path) -> str:
    """Compute SHA256 hash of file content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_metadata_path(cache_dir: pathlib.Path, cache_key: str) -> pathlib.Path:
    """Get path to metadata JSON file."""
    return cache_dir / f"{cache_key}.metadata.json"


def _get_model_path(cache_dir: pathlib.Path, cache_key: str) -> pathlib.Path:
    """Get path to cached model file."""
    return cache_dir / f"{cache_key}.nequip.zip"


def get_cached_model(
    model_id: Optional[str], download_url: str
) -> Optional[pathlib.Path]:
    """Check if model is cached and validate it.

    Returns cached file path if valid, None otherwise.
    """
    if _NEQUIP_NO_CACHE:
        return None

    cache_dir = get_cache_dir()
    cache_key = _compute_cache_key(model_id, download_url)

    model_path = _get_model_path(cache_dir, cache_key)
    metadata_path = _get_metadata_path(cache_dir, cache_key)

    if not model_path.exists() or not metadata_path.exists():
        return None

    # load metadata
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read cache metadata: {e}, redownloading")
        return None

    # validate file hash
    try:
        actual_hash = _compute_file_hash(model_path)
        expected_hash = metadata.get("file_sha256")

        if actual_hash != expected_hash:
            logger.warning(
                f"Cache validation failed: hash mismatch for {model_id or download_url}, redownloading"
            )
            return None
    except IOError as e:
        logger.warning(f"Failed to validate cached file: {e}, redownloading")
        return None

    logger.info(f"Using cached model from {model_path}")
    return model_path


def cache_model(
    model_id: Optional[str],
    download_url: str,
    download_fn: Callable[[pathlib.Path], None],
) -> pathlib.Path:
    """Download model and save to cache.

    Args:
        model_id: nequip.net model ID (or None for arbitrary URLs)
        download_url: URL to download from
        download_fn: function that downloads to the provided path

    Returns:
        Path to cached model file (or temporary file if caching is disabled)
    """
    if _NEQUIP_NO_CACHE:
        # download to temporary file without caching
        import tempfile

        tmpfile = tempfile.NamedTemporaryFile(suffix=".nequip.zip", delete=False)
        try:
            download_fn(pathlib.Path(tmpfile.name))
            tmpfile.close()
            return pathlib.Path(tmpfile.name)
        except Exception:
            tmpfile.close()
            pathlib.Path(tmpfile.name).unlink(missing_ok=True)
            raise

    cache_dir = get_cache_dir()
    cache_key = _compute_cache_key(model_id, download_url)

    model_path = _get_model_path(cache_dir, cache_key)
    metadata_path = _get_metadata_path(cache_dir, cache_key)

    # download to .partial file first, then rename on success
    # (avoids leaving corrupted files if download fails mid-way)
    partial_path = cache_dir / f"{cache_key}.nequip.zip.partial"

    try:
        # download
        download_fn(partial_path)

        # compute file hash
        file_hash = _compute_file_hash(partial_path)

        # save metadata
        metadata = {
            "model_id": model_id,
            "download_url": download_url,
            "file_sha256": file_hash,
            "cached_at": datetime.utcnow().isoformat(),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # rename to final location
        shutil.move(str(partial_path), str(model_path))

        logger.info(f"Model cached to {model_path}")
        return model_path

    except Exception:
        # clean up partial file on failure
        if partial_path.exists():
            partial_path.unlink()
        raise
