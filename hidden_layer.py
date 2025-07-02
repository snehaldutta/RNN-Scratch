import numpy as np
np.random.seed(42)

class HiddenLayer:
    def __init__(self, vocab_size: int, size: int) -> None:
        self.W = np.random.uniform(low=0, high=1, size=(size,size))
        self.bias = np.random.uniform(low=0, high=1, size=(size,1))
        self.states = np.zeros(shape=(vocab_size,size,1))
        self.next_del_activate = np.zeros(shape=(size,1))
        self.del_bias = np.zeros_like(self.bias)
        self.del_W = np.zeros_like(self.W)

    def get_hid_state(self, timestep:int) -> np.ndarray:
        if timestep < 0:
            return np.zeros_like(self.states[0])
        
        return self.states[timestep]
    

    def set_hid_state(self, timestep: int, hid_state: np.ndarray) -> None:
        self.states[timestep] = hid_state

    def activation(self, w_in: np.ndarray, timestep: int)-> np.ndarray:
        prev_hid_state = self.get_hid_state(timestep-1)
        w_hid_state = self.W @ prev_hid_state
        weighted_sum = w_in + w_hid_state + self.bias
        activate = np.tanh(weighted_sum)
        self.set_hid_state(timestep,activate)
        return activate
    
    def cal_del_per_step(self, timestep: int, del_out: np.ndarray)-> np.ndarray:
        del_activate = del_out+ self.next_del_activate
        del_w_sum = del_activate * (1-self.get_hid_state(timestep)**2)

        self.next_del_activate = self.W.T @ del_w_sum
        self.del_W += del_w_sum @ self.get_hid_state(timestep-1).T

        self.del_bias += del_w_sum

        return del_w_sum
    
    def update_weights_bias(self, lr: float)-> None:
        self.W -= lr* self.del_W
        self.bias -= lr* self.del_bias