from nequip.utils import load_callable
import dataclasses


class CallbackManager:
    """Parent callback class

    Centralized object to manage various callbacks that can be added-on.
    """

    def __init__(
        self,
        init_callbacks=[],
        start_of_epoch_callbacks=[],
        end_of_epoch_callbacks=[],
        end_of_batch_callbacks=[],
        end_of_train_callbacks=[],
        final_callbacks=[],
    ):

        # load all callbacks
        self.callbacks = {
            "init": [load_callable(callback) for callback in init_callbacks],
            "start_of_epoch": [
                load_callable(callback) for callback in start_of_epoch_callbacks
            ],
            "end_of_epoch": [
                load_callable(callback) for callback in end_of_epoch_callbacks
            ],
            "end_of_batch": [
                load_callable(callback) for callback in end_of_batch_callbacks
            ],
            "end_of_train": [
                load_callable(callback) for callback in end_of_train_callbacks
            ],
            "final": [load_callable(callback) for callback in final_callbacks],
        }

        for callback_type in self.callbacks:
            for callback in self.callbacks.get(callback_type):
                assert dataclasses.is_dataclass(callback) or callable(callback)

    def apply(self, trainer, callback_type: str):

        for callback in self.callbacks.get(callback_type):
            callback(trainer)

    def state_dict(self):
        return {"callback_manager_obj_callbacks": self.callbacks}

    def load_state_dict(self, state_dict):
        self.callbacks = state_dict.get("callback_manager_obj_callbacks")
