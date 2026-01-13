# ML-IAP

```{note}
This integration is currently in beta testing. Please report any issues or inconsistencies you encounter.
```

LAMMPS [ML-IAP](https://docs.lammps.org/Packages_details.html#pkg-ml-iap) is a LAMMPS integration that provides an interface for using machine learning potentials via the [pair_mliap](https://docs.lammps.org/pair_mliap.html) pair style. The NequIP framework provides a wrapper and workflow tools for ML-IAP integration with the **KOKKOS package**.

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

The `nequip-prepare-lmp-mliap` command prepares NequIP models for use with LAMMPS ML-IAP. This tool works with either [checkpoint files](../../guide/getting-started/files.md#checkpoint-files), [package files](../../guide/getting-started/files.md#package-files) from your trained models, or models from [nequip.net](https://nequip.net). The output file must have a `.nequip.lmp.pt` extension.

Basic usage:
```bash
# Using local checkpoint or package file
nequip-prepare-lmp-mliap \
  ckpt_file_or_package_file \
  output.nequip.lmp.pt \
  --modifiers modifier_to_apply

# Using model from nequip.net
nequip-prepare-lmp-mliap \
  nequip.net:group-name/model-name:version \
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

## Performance Optimization

### Persistent Compilation Cache

When using the same model repeatedly with ML-IAP, `torch.compile` can introduce significant overhead as it recompiles the model for each run.
By default, compiled artifacts are stored in a temporary directory and discarded between runs.

To enable persistent caching of compiled artifacts across runs, set the following environment variables before launching LAMMPS:

```bash
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/path/to/cache/
```

```{important}
Choose a cache directory with fast read/write access for optimal performance.
```

After the first run, compiled artifacts will be reused from the cache directory, potentially saving several minutes of compilation time on subsequent runs.
