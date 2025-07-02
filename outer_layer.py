import numpy as np
from scipy.special import softmax

np.random.seed(42)
class OuterLayer:
    def __init__(self, size: int, hid_size: int)-> None:
        self.V = np.random.uniform(low=0, high=1,size=(size, hid_size))
        self.bias = np.random.uniform(low=0, high=1, size=(size,1))
        self.states = np.zeros(shape=(size,size,1))
        self.del_bias = np.zeros_like(self.bias)
        self.del_V = np.zeros_like(self.V)

    def get_state(self, timestep: int)-> np.ndarray:
        return self.states[timestep]
    
    def set_state(self, timestep: int, pred: np.ndarray)-> np.ndarray:
        self.states[timestep] = pred

    def pred(self, hid_size: np.ndarray, timestep: int)-> np.ndarray:
        out = self.V @ hid_size + self.bias
        pred = softmax(out)
        self.set_state(timestep, pred)
        return pred

    def cal_del_per_step(self, expected: np.ndarray, hid_state: np.ndarray, timestep: int)-> np.ndarray:
        del_out = self.get_state(timestep) - expected
        self.del_V += del_out @ hid_state.T

        self.del_bias += del_out
        return self.V.T @ del_out
    
    def update_weights_bias(self, lr: float)-> None:
        self.V -= lr* self.del_V
        self.bias -= lr* self.del_bias