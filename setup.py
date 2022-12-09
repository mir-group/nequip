from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "nequip/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="nequip",
    version=version,
    description="NequIP is an open-source code for building E(3)-equivariant interatomic potentials.",
    download_url="https://github.com/mir-group/nequip",
    author="Simon Batzner, Albert Musealian, Lixin Sun, Anders Johansson, Mario Geiger, Tess Smidt",
    python_requires=">=3.7",
    packages=find_packages(include=["nequip", "nequip.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "nequip-train = nequip.scripts.train:main",
            "nequip-evaluate = nequip.scripts.evaluate:main",
            "nequip-benchmark = nequip.scripts.benchmark:main",
            "nequip-deploy = nequip.scripts.deploy:main",
        ]
    },
    install_requires=[
        "numpy",
        "ase",
        "tqdm",
        "torch>=1.8,<=1.12,!=1.9.0",  # torch.fx added in 1.8
        "e3nn>=0.3.5,<0.6.0",
        "pyyaml",
        "contextlib2;python_version<'3.7'",  # backport of nullcontext
        'contextvars;python_version<"3.7"',  # backport of contextvars for savenload
        "typing_extensions;python_version<'3.8'",  # backport of Final
        "torch-runstats>=0.2.0",
        "torch-ema>=0.3.0",
    ],
    zip_safe=True,
)
