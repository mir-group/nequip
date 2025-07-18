# Model Packaging

The NequIP packaging system creates portable, version-independent model archives using PyTorch's `torch.package` infrastructure.

## Overview

Packaging converts checkpoint files (`.ckpt`) into package files (`.nequip.zip`) that solve the portability problem inherent in checkpoint files. While checkpoint files are tied to specific software versions and may not load in different environments, package files bundle both the model and the code needed to run it, making them largely version-independent.

A package file contains:

- Model weights and architecture
- Snapshot of the implementation code  
- Metadata and configuration
- Example data for compilation

For user-facing information on packaging workflows and CLI usage, see the [packaging workflow](../../guide/getting-started/workflow.md#packaging) and [package files overview](../../guide/getting-started/files.md#package-files) in the user guide.

## Developer Notes

### Package Format Versioning

The packaging system uses `_CURRENT_NEQUIP_PACKAGE_VERSION` to track when the packaging mechanism has changed. This counter is incremented whenever breaking changes are made to the package format, as these changes represent the main barrier to maintaining backwards compatibility of packaged models. The {func}`~nequip.model.ModelFromPackage` loader includes logic to handle different package format versions based on this counter.

### Dependency Management

The packaging system handles dependencies by categorizing Python modules into three types:

- **Internal**: Core NequIP code (`nequip`, `e3nn`) - gets packaged with the model
- **External**: Large libraries (`numpy`, `triton`) - expected to be available in the target environment
- **Mock**: Optional dependencies (`matplotlib`) - imports are allowed but runtime usage raises errors

Developers of NequIP extension packages can register custom dependencies using the module registration system. This function allows extension packages to properly categorize their dependencies, ensuring they are handled correctly during packaging. Libraries with custom C++/CUDA ops or large stable third-party libraries should typically be registered as external, while optional dependencies can be mocked.

```{eval-rst}
.. autofunction:: nequip.scripts._package_utils.register_libraries_as_external_for_packaging
```

### Repackaging Support

The system includes complex logic to handle creating packages from other packages while maintaining proper importer chains. During repackaging, shared importers ensure all models come from the same source, which is required by PyTorch's packaging infrastructure.

### Sharp Edges with Model Modifiers

[`nequip-package`](../../guide/getting-started/workflow.md#packaging) will pick up files as long as there are no errors loading the files. However, certain coding patterns can cause loading errors that prevent packaging. Common pitfalls include:

**External dependencies not installed at package-time**: Top-level imports of optional dependencies will cause packaging to fail if those dependencies aren't available during packaging. Use lazy imports instead:

```python
# bad: top-level import
from openequivariance import TensorProductConv  # fails if not installed

# good: lazy import in __init__
def __init__(self, ...):
    super().__init__()
    from openequivariance import TensorProductConv
    self.tp_conv = TensorProductConv(...)
```

**Triton/GPU dependencies**: Triton decorators like `@triton.autotune` cause errors if the code is loaded on machines without GPUs. Wrap GPU-dependent code in conditional blocks:

```python
# hide GPU-dependent code to allow packaging on CPU-only systems
if torch.cuda.is_available():
    @triton.autotune(...)
    def gpu_kernel(...):
        # GPU implementation
```
