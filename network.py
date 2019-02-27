# %% Imports
import math
import numpy as np
import pandas as pd
import layer
from sklearn.datasets import fetch_openml

# %% Download Data
X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)


# %% Initialize params
input_size = X.shape[1]
layer_size = 5
output_size = 10

w_0 = np.random.rand(layer_size, input_size) / layer_size
w_1 = np.random.rand(layer_size, layer_size) / layer_size
w_2 = np.random.rand(output_size, layer_size) / layer_size
b_0 = (np.random.rand(layer_size, 1) / layer_size)[:, 0]
b_1 = (np.random.rand(layer_size, 1) / layer_size)[:, 0]
b_2 = (np.random.rand(output_size, 1) / layer_size)[:, 0]

# %% Normalize
x = X[3]
y = int(Y[3])
x = x / 255.0

# %% Feed-forward
i_1 = np.dot(w_0, x) + b_0
a_1 = np.tanh(i_1)
i_2 = np.dot(w_1, a_1) + b_1
a_2 = np.tanh(i_2)
output = softmax(np.dot(w_2, a_2) + b_2)
print(f"Predicted digit: {np.argmax(output)}")

# %% Backpropagation
# f is expected output
f = np.zeros(output_size)
f[y] = 1
# output layer
d_2 = (output - f)
gw_2 = np.matmul(d_2.reshape(output_size, 1), a_2.reshape(1, layer_size))
gb_2 = d_2
# hidden layer
d_1 = np.dot(w_2.T, d_2) * (1/(np.exp(i_2) + np.exp(-i_2)))
gw_1 = np.matmul(d_1.reshape(layer_size, 1), a_1.reshape(1, layer_size))
gb_1 = d_1
# input layer
d_0 = np.dot(w_1.T, d_1) * (1/(np.exp(i_1) + np.exp(-i_1)))
gw_0 = np.matmul(d_0.reshape(layer_size, 1), x.reshape(1, input_size))
gb_0 = d_0
# update weights
learing_rate = 0.1
w_0 = w_0 - learing_rate * gw_0
b_0 = b_0 - learing_rate * gb_0
w_1 = w_1 - learing_rate * gw_1
b_1 = b_1 - learing_rate * gb_1
w_2 = w_2 - learing_rate * gw_2
b_2 = b_2 - learing_rate * gb_2

# %% Defs
def softmax(x) -> np.ndarray:
    return np.exp(x) / sum(np.exp(x))


def cost_fun(x, y) -> float:
    return sum(x * math.log(y))

# %% recursive layer interface test
    network = layer.Layer(2, input_size, layer_size, output_size)
    f = np.zeros(output_size)
    f[y] = 1
    network.teach(x, f, 0.1)
