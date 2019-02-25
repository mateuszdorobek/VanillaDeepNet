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