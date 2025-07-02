import numpy as np
np.random.seed(42)
class InputLayer:
    def __init__(self, inputs: np.ndarray, hid_size: int)-> None:
        self.inputs = inputs
        self.w_i = np.random.uniform(low=0, high=1, size=(hid_size, len(inputs[0])))
        self.del_w_i = np.zeros_like(self.w_i)

    def get_inputs(self, timestep: int) -> np.ndarray:
        return self.inputs[timestep]
        
    def weighted_sum(self, timestep: int)-> np.ndarray:
        return self.w_i @ self.get_inputs(timestep)
    

    def cal_dels_per_step(self, timestep: int, del_weighted_sum: np. ndarray)-> None:
        self.del_w_i += del_weighted_sum @ self.get_inputs(timestep).T

    def update_weights_biases(self, learning_rate: float)-> None:
        self.del_w_i -= learning_rate*self.del_w_i

    