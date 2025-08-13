# Precision Settings

NequIP supports various precision settings that can affect both training speed and numerical accuracy.

```{warning}
Be cautious when using reduced precision during training and inference. While performance gains can be substantial, reduced precision can be detrimental for certain atomistic modeling tasks such as structure relaxations or static point calculations.
```

## Lightning Precision Settings

PyTorch Lightning provides built-in support for various precision modes through the `precision` trainer argument, e.g.:

```yaml
trainer:
  precision: bf16-mixed
```

For available options and details, see the [Lightning precision documentation](https://lightning.ai/docs/pytorch/stable/common/precision_basic.html).

## TensorFloat-32 (TF32)

If tensor cores are available (NVIDIA GPUs since Ampere architecture), TensorFloat-32 (TF32) can improve the speed of matrix multiplication operations in exchange for reduced numerical precision. This operates at the PyTorch backend level, independent of Lightning's precision settings.

Refer to the [PyTorch TF32 documentation](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices) for technical details.

### Training with TF32

During training, TF32 can be configured using the {class}`~nequip.train.callbacks.TF32Scheduler` callback. You can either enable it for all training or use dynamic scheduling:

```yaml
# Static TF32 setting
callbacks:
  - _target_: nequip.train.callbacks.TF32Scheduler
    schedule:
      0: true  # Enable TF32 for all training

# Dynamic TF32 scheduling for fast early training + precise convergence
callbacks:
  - _target_: nequip.train.callbacks.TF32Scheduler
    schedule:
      0: true      # Enable TF32 for faster early training  
      100: false   # Disable TF32 at epoch 100 for precise convergence
```

```{note}
TF32 settings only affect `float32` computations (i.e., when `model_dtype: float32`). For `float64` models, TF32 settings are ignored.
```

### TF32 at Inference

Whether TF32 is used during inference is determined by [compilation time](../getting-started/workflow.md#compilation) flags. When calling `nequip-compile`, you can specify `--tf32` or `--no-tf32`:

```bash
# Enable TF32 for inference
nequip-compile model.ckpt compiled_model.pt --tf32 ...

# Disable TF32 for inference (default)
nequip-compile model.ckpt compiled_model.pt --no-tf32 ...
```

The default behavior is to compile without TF32, regardless of training settings.