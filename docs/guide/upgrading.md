# Upgrading from pre-`0.7.0` `nequip`

This page summarizes selected features or options of older `nequip` versions and what replaced them in `nequip>=0.7.0`.

```{warning}
Importing, restarting, or migrating models or training runs from pre-0.7.0 versions of `nequip` is not supported.  Please use Python environment management to maintain separate installations of older `nequip` versions to keep working with that data, if necessary.
```

 - `nequip-evaluate` is replaced by using the `test` run type with `nequip-train`
 - `nequip-deploy` (which previously produces a TorchScript `.pth` file) is replaced by `nequip-compile` that can produce either a TorchScript `.nequip.pth` file or an AOT Inductor `.nequip.pt2` file (note that the extensions are imposed to mark the files as compatible with the NequIP framework)