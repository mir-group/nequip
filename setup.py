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
    description="NequIP is a software for building SE(3)-equivariant neural network interatomic potentials ",
    download_url="https://github.com/mir-group/nequip",
    author="Simon Batzner, Anders Johansson, Albert Musealian, Lixin Sun, Mario Geiger, Tess Smidt",
    python_requires=">=3.8",
    packages=find_packages(include=["nequip", "nequip.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "nequip-train = nequip._scripts.train:main",
            "nequip-restart = nequip._scripts.restart:main",
            "nequip-requeue = nequip._scripts.requeue:main",
            "nequip-deploy = nequip._scripts.deploy:main",
        ]
    },
    install_requires=[
        "numpy",
        "scipy",
        "ase",
        "torch",
        "torch_geometric",
        "e3nn>=0.2.5",
        "pyyaml",
    ],
    zip_safe=True,
)
