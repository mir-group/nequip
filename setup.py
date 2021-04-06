from setuptools import setup
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
    packages=["nequip"],
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "nequip_train = nequip._scripts.train:main",
            "nequip_restart = nequip._scripts.restart:main",
            "nequip_requeue = nequip._scripts.requeue:main",
            "nequip_deploy = nequip._scripts.deploy:main",
        ]
    },
    install_requires=[
        "numpy",
        "scipy",
        "ase",
        "torch",
        "torch_geometric",
        "e3nn",
        "pyyaml",
    ],
    zip_safe=True,
)
