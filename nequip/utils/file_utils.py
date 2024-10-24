# === Utility functions adapted from torch geometric (https://pytorch-geometric.readthedocs.io/en/latest/) ===

import ssl
import os
import os.path as osp
import urllib
import zipfile
from typing import Optional
from .logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def download_url(url: str, folder: str, filename: Optional[str] = None) -> None:
    """Downloads the content of an URL to a specific folder.

    Args:
        url (str)     : the url
        folder (str)  : the folder
        filename (str): the filename
    """

    filename = url.rpartition("/")[2].split("?")[0] if filename is None else filename
    path = osp.join(folder, filename)
    if osp.exists(path):  # pragma: no cover
        logger.info(f"Using existing file {path}")
        return path
    logger.info(f"Downloading from {url}")
    os.makedirs(folder, exist_ok=True)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)
    with open(path, "wb") as f:
        f.write(data.read())
    return path


def extract_zip(path: str, folder: str) -> None:
    """Extracts a zip archive to a specific folder.

    Args:
        path (str)  : the path to the tar archive
        folder (str): the folder
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)