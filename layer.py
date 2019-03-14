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


def init_wages(in_size, layer_size, init_type="Default") -> (np.ndarray, np.ndarray):
    wages_init = np.random.rand(in_size, layer_size)
    bias_init = np.random.rand(layer_size, 1)
    if init_type == "Default":
        wages_init *= 1/layer_size
    if init_type == "Xavier":
        wages_init *= np.sqrt(1/in_size)
        bias_init *= np.sqrt(1/in_size)
    return wages_init, bias_init


class Layer:
    def __init__(self, hidden_layers, input_size, layer_size, output_size, learning_rate, init_type="Default"):
        self.learning_rate = learning_rate
        self.w, self.b = init_wages(input_size, layer_size, init_type=init_type)
        self.gw = np.zeros_like(self.w)
        self.gb = np.zeros_like(self.b)
        hidden_layers -= 1
        if hidden_layers > 0:
            self.nextLayer = Layer(hidden_layers - 1, layer_size, layer_size, output_size, learning_rate, init_type)
        else:
            self.nextLayer = OutLayer(layer_size, output_size, learning_rate, init_type)

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

    def check_nth_grad(self, a_in, t, n):
        n -= 1
        if n <= 0:
            self.check_gradient(a_in, t)
        else:
            z = self.w.T @ a_in + self.b
            a_out = np.tanh(z)
            self.nextLayer.check_nth_grad(a_out, t, n)

    def check_gradient(self, a_in, t):
        prev_gw = copy.copy(self.gw)  # Copying it is an ugly hack, but it's only debug module
        prev_gb = copy.copy(self.gb)
        numerical_grad = np.empty_like(self.w)
        # Calculate gradient with backpropagation
        self.teach(a_in, t)
        curr_gw = self.gw - prev_gw
        # Calculate numerical gradient
        eps = 0.05
        eps_arr = np.full_like(self.gw, 0)
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                eps_arr[i, j] = eps
                self.w += eps_arr
                up_cost = self.cost_fun(a_in, t)
                self.w -= 2 * eps_arr
                down_cost = self.cost_fun(a_in, t)
                self.w += eps_arr
                numerical_grad[i, j] = (up_cost - down_cost) / (2 * eps)
                eps_arr[i, j] = 0
        # print(f"numerical grad = {numerical_grad} \n backpropagated = {curr_gw}")
        print(f"mean error = {np.sum(numerical_grad - curr_gw) / numerical_grad.size}")
        # Undo changes done in teach
        self.gw = prev_gw
        self.gb = prev_gb


class OutLayer:
    def __init__(self, input_size, layer_size, learning_rate, init_type="Default"):
        self.learning_rate = learning_rate
        self.w, self.b = init_wages(input_size, layer_size, init_type=init_type)
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
