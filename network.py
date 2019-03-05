# %% Imports
import math
import numpy as np
import pandas as pd
import layer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist():
    print("Loading data...")
    # x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    mnist = fetch_openml('mnist_784', cache=False)
    x = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    x /= 255.0
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=17)
    assert (X_train.shape[0] + X_test.shape[0] == mnist.data.shape[0])
    print("done")
    return X_train, X_test, y_train, y_test


def load_sample_data():
    x = np.random.rand(110, 3) * 255.0
    y = np.asarray([np.random.randint(0, 9, 110)]).T
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=17)
    return X_train, X_test, y_train, y_test


def get_example(X, Y):
    x = np.array([X[0]]).T
    y = int(Y[0])
    x = x / 255.0
    return x, y


def one_hot(y, output_size):
    t = np.array([np.zeros(output_size)]).T
    t[y] = 1
    return t

# %% main
if __name__ == "__main__":

    digits_nr = 10
    X_train, X_test, y_train, y_test = load_sample_data()
# %% Training
    x_example, y_example = get_example(X_train, y_train)
    t = one_hot(y_example, output_size=digits_nr)
    network = layer.Layer(hidden_layers=4,
                          input_size=X_train.shape[1],
                          layer_size=10,
                          output_size=digits_nr,
                          learning_rate=0.01)
    print("Training Loop:")
    for epoch in range(10):
        print(f"epoch: {epoch}")
        for i in range(X_train.shape[0]):
            t = one_hot(y_train[i], output_size=digits_nr)
            x = np.array([X_train[i]]).T
            network.teach(x, t)
            print(f"cost = {network.cost_fun(x, t)}")
            network.check_next_grad(x, t)
        network.apply_gradients()

    x = np.array([X_train[0]]).T
    print(network.classify(x))
    print(y_train[0])
