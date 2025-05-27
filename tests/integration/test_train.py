from conftest import TrainingInvarianceBaseTest 

class TestTrainingInvariance(TrainingInvarianceBaseTest):
    # Test restarts and validation loss invariance with 
    # exactly the same model config 
    def modify_model_config(self, original_config):
        return original_config.copy() 