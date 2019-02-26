# %% Imports
import math
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# %% Download Data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# %% Initialize params
input_size = X.shape[1]
layer_size = 1000
output_size = 10

w_0 = np.random.rand(layer_size, input_size) / 1000
w_1 = np.random.rand(layer_size, layer_size) / 1000
w_2 = np.random.rand(output_size, layer_size) / 1000
b_0 = (np.random.rand(layer_size, 1) / 1000)[:, 0]
b_1 = (np.random.rand(layer_size, 1) / 1000)[:, 0]
b_2 = (np.random.rand(output_size, 1) / 1000)[:, 0]

# %% Normalize
x = X[3]
x = x / 255.0

# %% Feed-forward
a_1 = np.tanh(np.dot(w_0, x) + b_0)
a_2 = np.tanh(np.dot(w_1, a_1) + b_1)
output = np.tanh(np.dot(w_2, a_2) + b_2)
print(f"Predicted digit: {np.argmax(output)}")


# %% Defs
def softmax(x) -> np.ndarray:
    return np.exp(x) / sum(np.exp(x))


def cost_fun(x, y) -> float:
    return sum(x * math.log(y))


print(softmax(output))
