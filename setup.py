from setuptools import setup, find_packages

setup(
    name="nequip",
    version="0.0.1",
    description="NequIP is a software for building SE(3)-equivariant neural network interatomic potentials ",
    download_url="https://github.com/mir-group/nequip",
    author="Simon Batzner, Anders Johansson, Albert Musealian, Lixin Sun, Mario Geiger, Tess Smidt",
    python_requires=">=3.6",
    packages=find_packages(include=["nequip", "nequip.*"]),
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
