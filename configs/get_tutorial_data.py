"""
Script to download tutorial data
"""

from nequip.utils import download_url
import os

url = "https://archive.materialscloud.org/record/file?record_id=1302&filename=fcu.xyz"
path = download_url(url, os.getcwd(), filename="fcu.xyz")
print(f"Downloaded data to {path}")
