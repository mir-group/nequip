# Troubleshooting

This page covers how one can troubleshoot a variety of issues when using the NequIP framework, and will be continuously improved based on user feedback.
You are also encouraged to look through past [GitHub issues](https://github.com/mir-group/nequip/issues), [GitHub discussions](https://github.com/mir-group/nequip/discussions), and our [FAQ page](faq.md).

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
  --input-path path/to/ckpt_file/or/package_file \
  --output-path path/to/compiled_model.nequip.pt2 \
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
