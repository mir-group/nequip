from nequip.utils import load_callable
import dataclasses


class CallbackManager:
    """Parent callback class

    Centralized object to manage various callbacks that can be added-on.
    """

    def __init__(
        self,
        callbacks={},
    ):
        CALLBACK_TYPES = [
            "init",
            "start_of_epoch",
            "end_of_epoch",
            "end_of_batch",
            "end_of_train",
            "final",
        ]
        # load all callbacks
        self.callbacks = {callback_type: [] for callback_type in CALLBACK_TYPES}

        for callback_type in callbacks:
            if callback_type not in CALLBACK_TYPES:
                raise ValueError(
                    f"{callback_type} is not a supported callback type.\nSupported callback types include "
                    + str(CALLBACK_TYPES)
                )
            # make sure callbacks are either dataclasses or functions
            for callback in callbacks[callback_type]:
                if not (dataclasses.is_dataclass(callback) or callable(callback)):
                    raise ValueError(
                        f"Callbacks must be of type dataclass or callable. Error found on the callback {callback} of type {callback_type}"
                    )
                self.callbacks[callback_type].append(load_callable(callback))

    def apply(self, trainer, callback_type: str):

        for callback in self.callbacks.get(callback_type):
            callback(trainer)

    def state_dict(self):
        return {"callback_manager_obj_callbacks": self.callbacks}

    def load_state_dict(self, state_dict):
        self.callbacks = state_dict.get("callback_manager_obj_callbacks")
