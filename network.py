import numpy as np
import matplotlib.pyplot as plt
import layer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist():
    print("Loading data...")
    mnist = fetch_openml('mnist_784', cache=False)
    x = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    x /= 255.0
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=17)
    assert (X_train.shape[0] + X_test.shape[0] == mnist.data.shape[0])
    print("Done")
    return X_train[0:1000], X_test, y_train[0:1000], y_test


def one_hot(y, output_size):
    t = np.array([np.zeros(output_size)]).T
    t[y] = 1
    return t


if __name__ == "__main__":

    digits_nr = 10
    X_train, X_test, y_train, y_test = load_mnist()
    network = layer.Layer(hidden_layers=5,
                          input_size=X_train.shape[1],
                          layer_size=100,
                          output_size=digits_nr,
                          learning_rate=0.003)
    x_check_cost = X_train
    y_check_cost = y_train
    print("Training Loop:")
    epochs = 50
    loss_values = []
    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        for i in range(X_train.shape[0]):
            t = one_hot(y_train[i], output_size=digits_nr)
            x = np.array([X_train[i]]).T
            network.teach(x, t)
            # print(f"cost = {network.cost_fun(x, t)}")
            network.check_next_grad(x, t)
            network.cost_fun(x, t)
            network.apply_gradients()
        mean_loss = 0.0
        for i in range(x_check_cost.shape[0]):
            t = one_hot(y_check_cost[i], output_size=digits_nr)
            x = np.array([x_check_cost[i]]).T
            mean_loss += network.cost_fun(x, t)
        loss_values.append(mean_loss / x_check_cost.shape[0])
    plt.plot(loss_values)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()
