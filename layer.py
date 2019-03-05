import math
import numpy as np


def softmax(x) -> np.ndarray:
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator/denominator


def d_tanh(x) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def cost_fun(x, y) -> float:
    return sum(x * math.log(y))


def initialize_wages(in_size, layer_size) -> np.ndarray:
    return np.random.rand(in_size, layer_size) / layer_size


def initialize_bias(layer_size) -> np.ndarray:
    return np.array(np.random.rand(layer_size, 1) / layer_size)


class Layer:
    def __init__(self, hidden_layers, input_size, layer_size, output_size, learning_rate):
        self.learning_rate = learning_rate
        self.w = initialize_wages(input_size, layer_size)
        self.b = initialize_bias(layer_size)
        self.gw = 0.0
        self.gb = 0.0
        hidden_layers -= 1
        if hidden_layers > 0:
            self.nextLayer = Layer(hidden_layers - 1, layer_size, layer_size, output_size, learning_rate)
        else:
            self.nextLayer = OutLayer(layer_size, output_size, learning_rate)

    def teach(self, a_in, t):
        z = self.w.T @ a_in + self.b
        a_out = np.tanh(z)
        next_diff = self.nextLayer.teach(a_out, t)
        diff = (self.nextLayer.w @ next_diff) * d_tanh(z)
        gw = a_in @ diff.T
        gb = diff
        self.gw += gw
        self.gb += gb
        # print(f"a{a_in.shape}, "
        #       f"w{self.w.shape}, "
        #       f"gw{gw.shape}, "
        #       f"b{self.b.shape}, "
        #       f"gb{gb.shape}, "
        #       f"diff{diff.shape}, "
        #       f"next_diff{next_diff.shape}")
        return diff

    def classify(self, a_in):
        z = self.w.T @ a_in + self.b
        a_out = np.tanh(z)
        # print(f"z{z.shape}"
        #       f"w{self.w.shape}"
        #       f"b{self.b.shape}"
        #       f"a_in{a_in.shape}")
        return self.nextLayer.classify(a_out)

    def apply_gradients(self):
        self.w -= self.gw * self.learning_rate
        self.b -= self.gb * self.learning_rate
        self.nextLayer.apply_gradients()

    def cost_fun(self, a_in, t):
        classification = self.classify(a_in)
        return np.log(np.dot(classification.T, t))


class OutLayer:
    def __init__(self, in_size, layer_size, learning_rate):
        self.learning_rate = learning_rate
        self.w = initialize_wages(in_size, layer_size)
        self.b = initialize_bias(layer_size)
        self.gw = 0.0
        self.gb = 0.0

    def teach(self, a_in, t):
        z = self.w.T @ a_in + self.b
        a_out = softmax(z)
        diff = a_out - t
        # print(f"diff{sum(diff)}")
        gw = a_in @ diff.T
        gb = diff
        self.gw += gw
        self.gb += gb
        # print(f" w{self.w.shape}, "
        #       f"a{a_in.shape}, "
        #       f"b{self.b.shape}, "
        #       f"t{t.shape}, "
        #       f"z{z.shape}, "
        #       f"a_out{a_out.shape}, "
        #       f"diff{diff.shape}, "
        #       f"gw{gw.shape}, "
        #       f"gb{gb.shape}")
        return diff

    def classify(self, a_in):
        z = self.w.T @ a_in + self.b
        # print(f"z{z.shape}"
        #       f"w{self.w.shape}"
        #       f"b{self.b.shape}"
        #       f"a_in{a_in.shape}")
        # attrs = vars(self)
        # print(', \n'.join("%s: %s" % item for item in attrs.items()))
        return softmax(z)

    def apply_gradients(self):
        self.w -= self.gw * self.learning_rate
        self.b -= self.gb * self.learning_rate
