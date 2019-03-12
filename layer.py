import math
import numpy as np
import copy


def softmax(x) -> np.ndarray:
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator


def d_tanh(x) -> np.ndarray:
    return 1 - (np.tanh(x) ** 2)


def initialize_wages(in_size, layer_size) -> np.ndarray:
    return np.random.rand(in_size, layer_size) / layer_size


def initialize_bias(layer_size) -> np.ndarray:
    return np.array(np.random.rand(layer_size, 1) / layer_size)


class Layer:
    def __init__(self, hidden_layers, input_size, layer_size, output_size, learning_rate):
        self.learning_rate = learning_rate
        self.w = initialize_wages(input_size, layer_size)
        self.b = initialize_bias(layer_size)
        self.gw = np.zeros_like(self.w)
        self.gb = np.zeros_like(self.b)
        hidden_layers -= 1
        if hidden_layers > 0:
            self.nextLayer = Layer(hidden_layers - 1, layer_size, layer_size, output_size, learning_rate)
        else:
            self.nextLayer = OutLayer(layer_size, output_size, learning_rate)

    def teach(self, a_in, t) -> np.ndarray:
        z = self.w.T @ a_in + self.b
        a_out = np.tanh(z)
        next_diff = self.nextLayer.teach(a_out, t)
        diff = (self.nextLayer.w @ next_diff) * d_tanh(z)
        gw = a_in @ diff.T
        gb = diff
        self.gw += gw
        self.gb += gb
        return diff

    def classify(self, a_in) -> np.ndarray:
        z = self.w.T @ a_in + self.b
        a_out = np.tanh(z)
        return self.nextLayer.classify(a_out)

    def apply_gradients(self):
        self.w -= self.gw * self.learning_rate
        self.b -= self.gb * self.learning_rate
        self.nextLayer.apply_gradients()
        self.gw = np.zeros_like(self.w)
        self.gb = np.zeros_like(self.b)

    def cost_fun(self, a_in, t) -> float:
        classification = self.classify(a_in)
        for i in range(0, len(t)):
            if t[i] == 1:
                return -np.log(classification[i])
        # return - (np.log(classification).T@t).item()
        # macierzowo

    def check_next_grad(self, a_in, t):
        z = self.w.T @ a_in + self.b
        a_out = np.tanh(z)
        return self.nextLayer.check_gradient(a_out, t)

    def check_gradient(self, a_in, t):
        prev_gw = copy.copy(self.gw)  # Copying it is an ugly hack, but it's only debug module
        self.teach(a_in, t)
        curr_gw = self.gw - prev_gw
        eps = 0.05
        eps_arr = np.full_like(self.gw, 0)
        eps_arr[0, 0] = eps
        self.w += eps_arr
        up_cost = self.cost_fun(a_in, t)
        self.w -= 2 * eps_arr
        down_cost = self.cost_fun(a_in, t)
        self.w += eps_arr
        numerical_grad = (up_cost - down_cost) / (2 * eps)
        print(f"numerical grad = {numerical_grad}, backpropagated = {curr_gw[0, 0]}")
        return curr_gw[0, 0]


class OutLayer:
    def __init__(self, in_size, layer_size, learning_rate):
        self.learning_rate = learning_rate
        self.w = initialize_wages(in_size, layer_size)
        self.b = initialize_bias(layer_size)
        self.gw = np.zeros_like(self.w)
        self.gb = np.zeros_like(self.b)

    def teach(self, a_in, t) -> np.ndarray:
        z = self.w.T @ a_in + self.b
        a_out = softmax(z)
        diff = a_out - t
        gw = a_in @ diff.T
        gb = diff
        self.gw += gw
        self.gb += gb
        return diff

    def classify(self, a_in) -> np.ndarray:
        z = self.w.T @ a_in + self.b
        a_out = softmax(z)
        return a_out

    def apply_gradients(self):
        self.w -= self.gw * self.learning_rate
        self.b -= self.gb * self.learning_rate
        self.gw = np.zeros_like(self.w)
        self.gb = np.zeros_like(self.b)
