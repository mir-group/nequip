Two examples are provided.

1. `example_lmdb_conversion.py` is a an example of how one can use `NequIPLMDBDataset.save_from_iterator()` to convert an iterator of AtomicDataDicts to LMDB-formatted data, based on the specific case of using `ase`-readable data. One can adapt the script with custom functions that can convert custom data formats to AtomicDataDicts.

2.  `datamodule_to_lmdb.py` is a script that can generally be used to convert a `datamodule` defined in a similar format as a usual config to LMDB-formatted data, to be used with a config file. `data.yaml` is provided as an example.

Note that due to the LMDB memory mapping, the LMDB dataset file size may appear as having a larger file size (with e.g. `ls -l`) than reality. `du -hd 0` will show how much disk space is actually being used in the given directory.
