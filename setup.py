from setuptools import setup

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open("...nequip/_version.py") as fp:
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
