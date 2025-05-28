# Troubleshooting

This page covers how one can troubleshoot a variety of issues when using the NequIP framework, and will be continuously improved based on user feedback.
You are also encouraged to look through past [GitHub issues](https://github.com/mir-group/nequip/issues), [GitHub discussions](https://github.com/mir-group/nequip/discussions), and our [FAQ page](faq.md).


## Codebase changes break my checkpoints

While we strive for stability, there are times when breaking changes are necessary to enable new features and capabilities.
Checkpoint files are unlikely to survive such breaking changes, i.e. one may no longer be able to load a checkpoint file to continue training or `nequip-compile` the model when upgrading versions.

The NequIP framework's proposed solution is to [package](workflow.md/#packaging) the model with `nequip-package`, which produces a packaged model file that contains a snapshot of the code (from the code version at package-time) associated with a model.
Packaged models can be compiled with `nequip-compile` for inference and/or used as pretrained models for fine-tuning (with the possibility of repackaging the model with `nequip-package` again).
Packaged models are more reliable in allowing for old models to be used across version changes of the NequIP codebase.
While packaged models guard against breaking changes in model code, they may no be compatible if there are fundamental changes to the code that performs packaging and loads packaged models.
Changes like this should be rare, and we will refrain from making such changes unless absolutely necessary.

More information can be found in the docs on [packaging](workflow.md/#packaging) and on the different [file types](files.md).

## Poor Training Behavior

If you're model doesn't seem like it's learning, the reasons could range from problematic model hyperparameters, to problematic training hyperparameters, to data-side problems.

As a start, ensure that the data format conforms to [NequIP conventions](conventions.md). For example, a common failure mode is when a dataset uses the opposite sign convention for stress.

In terms, of training hyperparameters, NequIP models tend to learn with learning rates of around 0.01, while Allegro models are better with learning rates of around 0.001.
One can always try to lower the learning rate to test if the problem could lie with a large learning rate causing large jumps in the loss landscape.


## Commons Errors

### `nequip-train config.yaml` fails

  **Problem**: Trying to run `nequip-train` as follows fails.
```bash
nequip-train config.yaml
```
  **Solution**: Read the [workflow docs](workflow.md) and follow hydra's command line options, e.g.
```bash
nequip-train -cn config.yaml
```

### Compilation with `nequip-compile` fails in AOT Inductor Mode

  **Problem**: Trying to run `nequip-compile` as follows, fails:
  ```bash
  nequip-compile \
  path/to/ckpt_file/or/package_file \
  path/to/compiled_model.nequip.pt2 \
  --device (cpu/cuda) \
  --mode aotinductor \
  --target target_integration
  ```

  with an error like this:
  ```bash
  torch._inductor.exc.CppCompileError: C++ compile error
  ```
  or like this:
  ```bash
  allegro_torch27/lib/python3.10/site-packages/torch/include/torch/csrc/inductor/aoti_include/common.h:4:10: fatal error: filesystem: No such file or directory
  #include <filesystem>
          ^~~~~~~~~~~~
  compilation terminated.
  ```
  
  **Solution**: Use newer GCC Version
  It's likely your GCC version does not support C++17. Try a GCC version >= 11 that supports C++17 by default (see [https://gcc.gnu.org/projects/cxx-status.html#cxx17](https://gcc.gnu.org/projects/cxx-status.html#cxx17)) 

  On HPC clusters, you can usually `module load` to a newer version of GCC.
