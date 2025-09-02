# ML-IAP

```{note}
This integration is currently in beta testing. Please report any issues or inconsistencies you encounter.
```

LAMMPS [ML-IAP](https://docs.lammps.org/Packages_details.html#pkg-ml-iap) is a LAMMPS integration that provides an interface for using machine learning potentials via the [pair_mliap](https://docs.lammps.org/pair_mliap.html) pair style. The NequIP framework provides a wrapper and workflow tools for ML-IAP integration.

## Building ML-IAP

Start with a Python environment that has `nequip` and other extensions such as `allegro` installed.

Install the required dependencies:
```bash
pip install cython==3.0.11 cupy-cuda12x
```

Clone LAMMPS and configure the build:
```bash
git clone --depth=1 https://github.com/lammps/lammps
cd lammps
```

Configure the build with the necessary options for ML-IAP support:
```bash
cmake \
  -B build-mliap \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=$(pwd)/lib/kokkos/bin/nvcc_wrapper \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D BUILD_MPI=ON \
  -D PKG_ML-IAP=ON \
  -D PKG_ML-SNAP=ON \
  -D MLIAP_ENABLE_PYTHON=ON \
  -D PKG_PYTHON=ON \
  -D BUILD_SHARED_LIBS=ON \
  cmake
```

Build LAMMPS:
```bash
cmake --build build-mliap -j 8
```

Install the Python interface:
```bash
cd build-mliap
make install-python
```

## Usage with NequIP Framework

The NequIP framework provides the `nequip-prepare-lmp-mliap` command-line tool to prepare models for use with LAMMPS ML-IAP.

### Preparing Models with nequip-prepare-lmp-mliap

The `nequip-prepare-lmp-mliap` command prepares NequIP models for use with LAMMPS ML-IAP. This tool works with either [checkpoint files](../../guide/getting-started/files.md#checkpoint-files) or [package files](../../guide/getting-started/files.md#package-files) from your trained models. The output file must have a `.nequip.lmp.pt` extension.

Basic usage:
```bash
nequip-prepare-lmp-mliap \
  ckpt_file_or_package_file \
  output.nequip.lmp.pt \
  --modifiers modifier_to_apply
```

NequIP models support [OpenEquivariance acceleration](../../guide/accelerations/openequivariance.md) via `--modifiers enable_OpenEquivariance`.

```{tip}
To see all available command line options, use `nequip-prepare-lmp-mliap -h`.
```

## LAMMPS Script Example

Below is part of a LAMMPS script example for using NequIP models with ML-IAP:

```lammps
# general settings
units         metal
boundary      p p p
atom_style    atomic
atom_modify   map yes
newton        on

# data
read_data     system.data
replicate     2 2 2

# set up potential
pair_style    mliap unified output.nequip.lmp.pt 0
pair_coeff    * * H O
```

Replace `output.nequip.lmp.pt` with the path to your ML-IAP interface file, `system.data` with your data file, and specify the element types in your system in the `pair_coeff` line.

Example command to run the LAMMPS simulation with Kokkos GPU support:
```bash
srun -n 1 /path/to/lammps/build/lmp -in in.lammps -k on g 1 -sf kk -pk kokkos newton on neigh half
```