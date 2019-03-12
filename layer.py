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
        # Adam members
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.eps = 0.00000001
        self.mw = np.zeros_like(self.w)
        self.mb = np.zeros_like(self.b)
        self.vw = np.zeros_like(self.w)
        self.vb = np.zeros_like(self.b)

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
        # Uses Adam algorithm
        self.mw = self.beta1 * self.mw + (1 - self.beta1) * self.gw
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * self.gb
        self.vw = self.beta2 * self.vw + (1 - self.beta2) * np.power(self.gw, 2)
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * np.power(self.gb, 2)
        mw_dash = self.mw / (1 - self.beta1)
        mb_dash = self.mb / (1 - self.beta1)
        vw_dash = self.vw / (1 - self.beta2)
        vb_dash = self.vb / (1 - self.beta2)
        self.w -= self.learning_rate * mw_dash / (vw_dash + self.eps)
        self.b -= self.learning_rate * mb_dash / (vb_dash + self.eps)
        # self.w -= self.learning_rate * self.gw
        # self.b -= self.learning_rate * self.gb
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
    def __init__(self, in_size, layer_size, learning_rate):
        self.learning_rate = learning_rate
        self.w = initialize_wages(in_size, layer_size)
        self.b = initialize_bias(layer_size)
        self.gw = np.zeros_like(self.w)
        self.gb = np.zeros_like(self.b)
        # Adam members
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.eps = 0.00000001
        self.mw = np.zeros_like(self.w)
        self.mb = np.zeros_like(self.b)
        self.vw = np.zeros_like(self.w)
        self.vb = np.zeros_like(self.b)

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
        # Uses Adam algorithm
        self.mw = self.beta1 * self.mw + (1 - self.beta1) * self.gw
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * self.gb
        self.vw = self.beta2 * self.vw + (1 - self.beta2) * np.power(self.gw, 2)
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * np.power(self.gb, 2)
        mw_dash = self.mw / (1 - self.beta1)
        mb_dash = self.mb / (1 - self.beta1)
        vw_dash = self.vw / (1 - self.beta2)
        vb_dash = self.vb / (1 - self.beta2)
        self.w -= self.learning_rate * mw_dash / (vw_dash + self.eps)
        self.b -= self.learning_rate * mb_dash / (vb_dash + self.eps)
        # self.w -= self.gw * self.learning_rate
        # self.b -= self.gb * self.learning_rate
        self.gw = np.zeros_like(self.w)
        self.gb = np.zeros_like(self.b)
