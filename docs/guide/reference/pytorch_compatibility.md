# PyTorch Version Compatibility

This page documents known issues with specific PyTorch versions when using NequIP framework models.

```{warning}
We recommend testing your workflow with your target PyTorch version before deploying to production, especially when using compilation features.
```

## Known Issues by PyTorch Version

### PyTorch 2.10.0

**Issue:** CPU + AOTInductor compilation failure

**Affected Feature:** `nequip-compile --mode aotinductor --device cpu`

**Status:** Known bug, may be fixed in future versions

**Workaround:** Use PyTorch 2.9.1, or use `--mode torchscript`, or compile for CUDA

---

### PyTorch 2.9.1

**Issue:** AOTInductor compilation accuracy failure on A100 GPUs

**Affected Feature:** `nequip-compile --mode aotinductor --device cuda` (with OpenEquivariance)

**Reproducer:** ([#574](https://github.com/mir-group/nequip/issues/574))
```bash
nequip-compile \
  nequip.net:mir-group/NequIP-OAM-L:0.1 \
  mir-group__NequIP-OAM-L__0.1.nequip.pt2 \
  --mode aotinductor \
  --device cuda \
  --target ase \
  --modifiers enable_OpenEquivariance
```

**Status:** Known bug in PyTorch 2.9.1, compilation check fails with MaxAbsError exceeding tolerance

**Workaround:** Use PyTorch 2.9.0

---

### PyTorch 2.6.0+

**Issue:** CPU train-time compilation issues

**Affected Feature:** `compile_mode: compile` on CPU

**Status:** Known limitation

**Workaround:** Avoid CPU train-time compilation; use GPU or eager mode

## General CPU Compilation Advisory

```{warning}
**CPU + Compilation has known problems across multiple PyTorch versions.**

Be aware that there are known issues when using PyTorch compilation features on CPU devices:
- `nequip-compile` with `--mode aotinductor`
- Train-time compilation with `compile_mode: compile`

If you plan to use compilation on CPU, carefully test your specific workflow and PyTorch version before deployment.
```

## Reporting Issues

If you encounter PyTorch version-specific issues not listed here, please open a [GitHub issue](https://github.com/mir-group/nequip/issues) with your PyTorch version, workflow details, and error messages.
