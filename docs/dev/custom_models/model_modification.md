# Model Modification

## What is a model modifier

A model modifier is a class method decorated with {func}`~nequip.nn.model_modifier_utils.model_modifier` that dynamically alters models on-the-fly. 

```{eval-rst}
.. autofunction:: nequip.nn.model_modifier_utils.model_modifier
```

Example usage:

```python
@model_modifier(persistent=True)
@classmethod
def modify_PerTypeScaleShift(cls, model, scales=None, shifts=None, ...):
    # Implementation here
    pass
```

There are two types:

- **Non-persistent** modifiers: Change implementation without altering the model itself, often for accelerations (e.g. `enable_OpenEquivariance`).
- **Persistent** modifiers: Alter the model's parameters or structure (e.g. `modify_PerTypeScaleShift` for changing per-type energy scales).

Model modifiers **MUST** preserve model state by transferring weights if there are trainable parameters.

Model modifiers can be written with the help of utility functions like {func}`~nequip.nn.model_modifier_utils.replace_submodules`:

```{eval-rst}
.. autofunction:: nequip.nn.model_modifier_utils.replace_submodules
```

## How are model modifiers used

There are two ways to use model modifiers:

### 1. Train-time usage

Use {func}`~nequip.model.modify` in your [training config file](../../guide/getting-started/workflow.md#training) to wrap your model.
This can be used for train-time acceleration (e.g. [OpenEquivariance](../../guide/accelerations/openequivariance.md)) or fine-tuning utilities (e.g. `modify_PerTypeScaleShift`):

```yaml
model:
  _target_: nequip.model.modify
  modifiers:
    - modifier: modify_PerTypeScaleShift
      shifts:
        C: 1.23
        H: 0.12
  model:
    _target_: nequip.model.ModelFromPackage
    # ... 
```

### 2. Compile-time usage

Use `nequip-compile --modifiers` for **accelerations** during [compilation](../../guide/getting-started/workflow.md#compilation):

```bash
nequip-compile model.ckpt compiled.pth --modifiers enable_OpenEquivariance ...
```