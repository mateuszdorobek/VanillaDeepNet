import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import layer
import copy
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

digits_nr = 10
epochs = 100
batch_size = 64
learning_rate = 0.005
layer_size = 64
hidden_layers = 2
augment_times = 8
stop_threshold = 3
init_type = "Default"


def load_mnist():
    print("Loading data...")
    mnist = fetch_openml('mnist_784', cache=False)
    print(mnist.__class__)
    x = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    x /= 255.0
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=17)
    assert (X_train.shape[0] + X_test.shape[0] == mnist.data.shape[0])
    print("Done")
    return X_train, X_test, y_train, y_test


def one_hot(y, output_size):
    t = np.array([np.zeros(output_size)]).T
    t[y] = 1
    return t


def check_loss(network, x_check_cost, y_check_cost):
    mean_loss = 0.0
    for i in range(x_check_cost.shape[0]):
        t = one_hot(y_check_cost[i], output_size=digits_nr)
        x = np.array([x_check_cost[i]]).T
        mean_loss += network.cost_fun(x, t)
    return mean_loss / x_check_cost.shape[0]


def pickle_mnist():
    print("Pickling MNIST...")
    mnist = fetch_openml('mnist_784', cache=False)
    pickling_on = open("mnist.pickle", "wb")
    pickle.dump(mnist, pickling_on)
    pickling_on.close()
    print("Pickling done")


seq = iaa.Sequential([
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-8, 8),
        shear=(-3, 3)
    )
])


def augment(X, y, img_shape, augment_times) -> (np.ndarray, np.ndarray):
    assert (X.shape[1] == img_shape[0] * img_shape[1])
    X_augmented = np.empty((X.shape[0] * augment_times, X.shape[1]))
    X_augmented[0:X.shape[0], :] = X
    y_augmented = np.empty((y.shape[0] * augment_times), dtype=int)
    y_augmented[0:y.shape[0]] = y
    X_arr = X.reshape(X.shape[0], img_shape[0], img_shape[1])
    for i in range(1, augment_times):
        X_aug_arr = seq.augment_images(X_arr)
        X_aug_vec = X_aug_arr.reshape(X_aug_arr.shape[0], img_shape[0] * img_shape[1])
        # X_augmented.append(X_aug_vec)
        # y_augmented.append(y)
        X_augmented[i * X.shape[0]: (i+1) * X.shape[0], :] = X_aug_vec
        y_augmented[i * y.shape[0]: (i+1) * y.shape[0]] = y
    return X_augmented, y_augmented


def unpickle_mnist():
    pickle_off = open("mnist.pickle", "rb")
    mnist = pickle.load(pickle_off)
    pickle_off.close()
    x = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    x /= 255.0
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=17)
    assert (X_train.shape[0] + X_test.shape[0] == mnist.data.shape[0])
    return X_train[0:1000], X_test, y_train[0:1000], y_test


def confusion_matrix(network, X, y):
    y_pred = pd.Series(copy.copy(y), dtype=int, name="Predicted") # copying y works as prealocation
    for i in range(X.shape[0]):
        x = np.array([X[i]]).T
        predicted = np.argmax(network.classify(x))
        y_pred.set_value(i, predicted)
    y_actu = pd.Series(y, name="Actual")
    return pd.crosstab(y_actu, y_pred)


def check_accuracy(network, X_test, y_test):
    # Code duplication, yet it does not copy memory, hence it it's faster
    true_positive_counter = 0.0
    for i in range(X_test.shape[0]):
        x = np.array([X_test[i]]).T
        if np.argmax(network.classify(x)) == y_test[i]:
            true_positive_counter += 1
    return true_positive_counter / X_test.shape[0]


if __name__ == "__main__":
    # if you want to perform loading data faster pickle database first
    # and then use pickled binary file (unpickle_mnist)
    # pickle_mnist()
    X_train, X_test, y_train, y_test = unpickle_mnist()
    # plt.imshow(X_train[0].reshape(28, 28))
    # X_train, X_test, y_train, y_test = load_mnist()
    X_train, y_train = augment(X_train, y_train, (28, 28), augment_times)
    network = layer.Layer(hidden_layers=hidden_layers,
                          input_size=X_train.shape[1],
                          layer_size=layer_size,
                          output_size=digits_nr,
                          learning_rate=learning_rate,
                          init_type=init_type)
    print("Training Loop:")
    train_loss_values = []
    test_loss_values = []
    accuracy = []
    loss_values = []
    prev_loss = float("inf")
    loss_rise_count = 0
    best_network = copy.deepcopy(network)

    for epoch in range(epochs):
        print(f"epoch: {epoch + 1}")
        for i in range(X_train.shape[0]):
            t = one_hot(y_train[i], output_size=digits_nr)
            x = np.array([X_train[i]]).T
            network.teach(x, t)
            if (i + 1) % batch_size == 0:
                network.apply_gradients()
        train_loss_values.append(check_loss(network, X_train, y_train))
        test_loss = check_loss(network, X_test, y_test)
        test_loss_values.append(test_loss)
        accuracy.append(check_accuracy(network, X_test, y_test))
        if prev_loss < test_loss:
            loss_rise_count += 1
            if loss_rise_count >= stop_threshold:
                break
        else:
            best_network = copy.deepcopy(network)
            loss_rise_count = 0
        prev_loss = test_loss
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print(f"hidden_layers: {hidden_layers}\nlayer_size: {layer_size}\n"
          f"learning_rate: {learning_rate}\ninit_type: {init_type}\nbatch_size: {batch_size}")
    print(f"train loss: {train_loss_values[-1].item()}")
    print(f"test loss: {test_loss_values[-1].item()}")
    print(f"accuracy: {accuracy[-1]}")
    print(f"confusion matrix:\n{confusion_matrix(network, X_test, y_test)}")

    plt.subplot(211)
    plt.plot(train_loss_values, linestyle='-.', label='training')
    plt.plot(test_loss_values, linestyle='-', label='test')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.subplot(212)
    plt.plot(accuracy, linestyle='-', label='training')
    plt.show()
