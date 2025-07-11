# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.


from nequip.model.utils import _EAGER_MODEL_KEY
from nequip.model.saved_models import ModelFromPackage, ModelFromCheckpoint
from nequip.model.modify_utils import only_apply_persistent_modifiers
from nequip.train.lightning import _SOLE_MODEL_KEY


def load_saved_model(
    input_path,
    compile_mode: str = _EAGER_MODEL_KEY,
    model_key: str = _SOLE_MODEL_KEY,
):
    """Load a saved model from checkpoint or package.

    Args:
        input_path: path to the model checkpoint or package file
        compile_mode (str): compile mode for the model (default: _EAGER_MODEL_KEY)
        model_key (str): key to select the model from ModuleDict (default: _SOLE_MODEL_KEY)
    """

    # use package load path if extension matches, otherwise assume checkpoint file
    use_ckpt = not str(input_path).endswith(".nequip.zip")
    if use_ckpt:
        # we only apply persistent modifiers when building from checkpoint
        # i.e. acceleration modifiers won't be applied, and have to be specified during compile time
        with only_apply_persistent_modifiers(persistent_only=True):
            model = ModelFromCheckpoint(input_path, compile_mode=compile_mode)
    else:
        # packaged models will never have non-persistent modifiers built in
        model = ModelFromPackage(input_path, compile_mode=compile_mode)

    model = model[model_key]
    # ^ `ModuleDict` of `GraphModel` is loaded, we then select the desired `GraphModel` (`model_key` defaults to work for single model case)

    return model
