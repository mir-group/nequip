# Miscellaneous

## Custom Module Implementation

Try to use {meth}`torch.nn.Module.extra_repr` when defining custom {class}`torch.nn.Module` subclasses to convey crucial information about the model. (The model structure is printed before training if `_NEQUIP_LOG_LEVEL=DEBUG` environment variable is set.)