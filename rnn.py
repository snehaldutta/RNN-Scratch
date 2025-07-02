import numpy as np
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from outer_layer import OuterLayer
from typing import List

class VanillaRNN:
    def __init__(self, vocab_size: int, hid_size: int, alpha: float)-> None:
        self.hiddenLayer = HiddenLayer(vocab_size, hid_size)
        self.outer_layer = OuterLayer(vocab_size, hid_size)
        self.hid_size = hid_size
        self.alpha = alpha

    def ffrwd(self, inputs: np.ndarray)-> OuterLayer:
        self.input_layer = InputLayer(inputs, self.hid_size)
        for step in range(len(inputs)):
            w_in = self.input_layer.weighted_sum(step)
            activate = self.hiddenLayer.activation(w_in, step)
            self.outer_layer.pred(activate, step)

        return self.outer_layer
    
    def backprop(self, expected: np.ndarray)-> None:
        for step in reversed(range(len(expected))):
            del_out = self.outer_layer.cal_del_per_step(
                expected[step],
                self.hiddenLayer.get_hid_state(step), step)
            
            del_w_sum = self.hiddenLayer.cal_del_per_step(
                step, del_out
            )
            self.input_layer.cal_dels_per_step(step,del_w_sum)

        self.outer_layer.update_weights_bias(self.alpha)
        self.hiddenLayer.update_weights_bias(self.alpha)
        self.outer_layer.update_weights_bias(self.alpha)


    def loss(self, y_hat: List[np.ndarray], y: List[np.ndarray]) -> float:
        return sum(-np.sum(y[i]* np.log(y_hat[i])for i in range(len(y))))
    

    def train(self, inputs: np.ndarray, expected: np.ndarray, epochs: int)-> None:
        for epoch in range(epochs):
            print(f'Epoch : {epoch}')
            for idx, input in enumerate(inputs):
                y_hats = self.ffrwd(input)
                self.backprop(expected[idx])
                print(
                    f"Loss round: {self.loss([y for y in y_hats.states], expected[idx])}"
                )