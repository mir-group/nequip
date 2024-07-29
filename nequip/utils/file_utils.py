# === Utility functions adapted from torch geometric (https://pytorch-geometric.readthedocs.io/en/latest/) ===

import ssl
import os
import os.path as osp
import urllib
import zipfile
import logging

logger = logging.getLogger(__name__)


def download_url(url, folder, filename=None):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The url.
        folder (str): The folder.
        filename (str): The filename
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


def extract_zip(path, folder):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (str): The path to the tar archive.
        folder (str): The folder.
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)
