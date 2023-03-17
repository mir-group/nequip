import tempfile
import functools
import itertools
from os.path import dirname, basename, abspath
from typing import Dict, Any, List, Union, Optional, Sequence

import ase
import ase.io

import torch
import torch.multiprocessing as mp


from nequip.utils.multiprocessing import num_tasks
from .. import AtomicData
from ..transforms import TypeMapper
from ._base_datasets import AtomicInMemoryDataset


def _ase_dataset_reader(
    rank: int,
    world_size: int,
    tmpdir: str,
    ase_kwargs: dict,
    atomicdata_kwargs: dict,
    include_frames,
    global_options: dict,
) -> Union[str, List[AtomicData]]:
    """Parallel reader for all frames in file."""
    if world_size > 1:
        from nequip.utils._global_options import _set_global_options

        # ^ avoid import loop
        # we only `multiprocessing` if world_size > 1
        _set_global_options(global_options)
    # interleave--- in theory it is better for performance for the ranks
    # to read consecutive blocks, but the way ASE is written the whole
    # file gets streamed through all ranks anyway, so just trust the OS
    # to cache things sanely, which it will.
    # ASE handles correctly the case where there are no frames in index
    # and just gives an empty list, so that will succeed:
    index = slice(rank, None, world_size)
    if include_frames is None:
        # count includes 0, 1, ..., inf
        include_frames = itertools.count()

    datas = []
    # stream them from ase too using iread
    for i, atoms in enumerate(ase.io.iread(**ase_kwargs, index=index, parallel=False)):
        global_index = rank + (world_size * i)
        datas.append(
            (
                global_index,
                AtomicData.from_ase(atoms=atoms, **atomicdata_kwargs)
                if global_index in include_frames
                # in-memory dataset will ignore this later, but needed for indexing to work out
                else None,
            )
        )
    # Save to a tempfile---
    # there can be a _lot_ of tensors here, and rather than dealing with
    # the complications of running out of file descriptors and setting
    # sharing methods, since this is a one time thing, just make it simple
    # and avoid shared memory entirely.
    if world_size > 1:
        path = f"{tmpdir}/rank{rank}.pth"
        torch.save(datas, path)
        return path
    else:
        return datas


class ASEDataset(AtomicInMemoryDataset):
    """

    Args:
        ase_args (dict): arguments for ase.io.read
        include_keys (list): in addition to forces and energy, the keys that needs to
             be parsed into dataset
             The data stored in ase.atoms.Atoms.array has the lowest priority,
             and it will be overrided by data in ase.atoms.Atoms.info
             and ase.atoms.Atoms.calc.results. Optional
        key_mapping (dict): rename some of the keys to the value str. Optional

    Example: Given an atomic data stored in "H2.extxyz" that looks like below:

    ```H2.extxyz
    2
    Properties=species:S:1:pos:R:3 energy=-10 user_label=2.0 pbc="F F F"
     H       0.00000000       0.00000000       0.00000000
     H       0.00000000       0.00000000       1.02000000
    ```

    The yaml input should be

    ```
    dataset: ase
    dataset_file_name: H2.extxyz
    ase_args:
      format: extxyz
    include_keys:
      - user_label
    key_mapping:
      user_label: label0
    chemical_symbols:
      - H
    ```

    for VASP parser, the yaml input should be
    ```
    dataset: ase
    dataset_file_name: OUTCAR
    ase_args:
      format: vasp-out
    key_mapping:
      free_energy: total_energy
    chemical_symbols:
      - H
    ```

    """

    def __init__(
        self,
        root: str,
        ase_args: dict = {},
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        AtomicData_options: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
        type_mapper: TypeMapper = None,
        key_mapping: Optional[dict] = None,
        include_keys: Optional[List[str]] = None,
    ):
        self.ase_args = {}
        self.ase_args.update(getattr(type(self), "ASE_ARGS", dict()))
        self.ase_args.update(ase_args)
        assert "index" not in self.ase_args
        assert "filename" not in self.ase_args

        self.include_keys = include_keys
        self.key_mapping = key_mapping

        super().__init__(
            file_name=file_name,
            url=url,
            root=root,
            AtomicData_options=AtomicData_options,
            include_frames=include_frames,
            type_mapper=type_mapper,
        )

    @classmethod
    def from_atoms_list(cls, atoms: Sequence[ase.Atoms], **kwargs):
        """Make an ``ASEDataset`` from a list of ``ase.Atoms`` objects.

        If `root` is not provided, a temporary directory will be used.

        Please note that this is a convinience method that does NOT avoid a round-trip to disk; the provided ``atoms`` will be written out to a file.

        Ignores ``kwargs["file_name"]`` if it is provided.

        Args:
            atoms
            **kwargs: passed through to the constructor
        Returns:
            The constructed ``ASEDataset``.
        """
        if "root" not in kwargs:
            tmpdir = tempfile.TemporaryDirectory()
            kwargs["root"] = tmpdir.name
        else:
            tmpdir = None
        kwargs["file_name"] = tmpdir.name + "/atoms.xyz"
        atoms = list(atoms)
        # Write them out
        ase.io.write(kwargs["file_name"], atoms, format="extxyz")
        # Read them in
        obj = cls(**kwargs)
        if tmpdir is not None:
            # Make it keep a reference to the tmpdir to keep it alive
            # When the dataset is garbage collected, the tmpdir will
            # be too, and will (hopefully) get deleted eventually.
            # Or at least by end of program...
            obj._tmpdir_ref = tmpdir
        return obj

    @property
    def raw_file_names(self):
        return [basename(self.file_name)]

    @property
    def raw_dir(self):
        return dirname(abspath(self.file_name))

    def get_data(self):
        ase_args = {"filename": self.raw_dir + "/" + self.raw_file_names[0]}
        ase_args.update(self.ase_args)

        # skip the None arguments
        kwargs = dict(
            include_keys=self.include_keys,
            key_mapping=self.key_mapping,
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.update(self.AtomicData_options)
        n_proc = num_tasks()
        with tempfile.TemporaryDirectory() as tmpdir:
            from nequip.utils._global_options import _get_latest_global_options

            # ^ avoid import loop
            reader = functools.partial(
                _ase_dataset_reader,
                world_size=n_proc,
                tmpdir=tmpdir,
                ase_kwargs=ase_args,
                atomicdata_kwargs=kwargs,
                include_frames=self.include_frames,
                # get the global options of the parent to initialize the worker correctly
                global_options=_get_latest_global_options(),
            )
            if n_proc > 1:
                # things hang for some obscure OpenMP reason on some systems when using `fork` method
                ctx = mp.get_context("forkserver")
                with ctx.Pool(processes=n_proc) as p:
                    # map it over the `rank` argument
                    datas = p.map(reader, list(range(n_proc)))
                # clean up the pool before loading the data
                datas = [torch.load(d) for d in datas]
                datas = sum(datas, [])
                # un-interleave the datas
                datas = sorted(datas, key=lambda e: e[0])
            else:
                datas = reader(rank=0)
                # datas here is already in order, stride 1 start 0
                # no need to un-interleave
        # return list of AtomicData:
        return [e[1] for e in datas]
