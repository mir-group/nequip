"""
Script to download tutorial data
"""

from nequip.utils import download_url
import os

url = "https://archive.materialscloud.org/records/ycbvx-knj69/files/fcu.xyz?download=1"
path = download_url(url, os.getcwd(), filename="fcu.xyz")
print(f"Downloaded data to {path}")
