from conftest import TrainingInvarianceBaseTest 

class TestOEQTrainingInvariance(TrainingInvarianceBaseTest):
    def modify_model_config(self, original_config):
        new_config = original_config.copy()
        training_module = new_config["training_module"]
        original_model = training_module["model"]
        training_module["model"] = {
            "_target_": "nequip.model.modify",
            "modifiers": [{"modifier": "enable_OpenEquivariance"}],
            "model": original_model,
        }
        return new_config
    
    def map_location(self):
        return "cuda"