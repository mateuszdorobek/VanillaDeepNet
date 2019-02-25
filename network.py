#%%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#%%
print(type(X))
print(type(y))
print(X.shape)
print(y.shape)

print(np.__version__)
print(pd.__version__)

#%% Initialize params
input_size = X.shape[1]
layer_size = 1000
output_size = 10
# w_0 = np.zeros((layer_size, input_size + 1))
# w_1 = np.zeros((layer_size, layer_size + 1))
# w_2 = np.zeros((10, layer_size + 1))
w_0 = np.random.rand(layer_size, input_size + 1)/1000
w_1 = np.random.rand(layer_size, layer_size + 1)/1000
w_2 = np.random.rand(10, layer_size + 1)/1000

#%% Normalize
x = X[0]
x = x/255.0
# Feed-forward
l_1 = np.tanh(np.dot(w_0,np.append(x, [1])))
l_2 = np.tanh(np.dot(w_1,np.append(l_1, [1])))
output = np.tanh(np.dot(w_2,np.append(l_2, [1])))
print(f"Predicted digit: {np.argmax(output)}")