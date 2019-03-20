import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import layer
import copy
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from imgaug import augmenters as iaa
import math

digits_nr = 10
epochs = 20
batch_size = 64
learning_rate = 0.005
layer_size = 128
hidden_layers = 3
augment_times = 4
stop_threshold = 3
init_type = "Default"
optimizer = "Default"


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


def pickle_mnist(file_name):
    print("Pickling MNIST...")
    mnist = fetch_openml('mnist_784', cache=False)
    pickling_on = open(file_name, "wb")
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
        X_augmented[i * X.shape[0]: (i + 1) * X.shape[0], :] = X_aug_vec
        y_augmented[i * y.shape[0]: (i + 1) * y.shape[0]] = y
    return X_augmented, y_augmented


def unpickle_mnist(file_name, size):
    pickle_off = open(file_name, "rb")
    mnist = pickle.load(pickle_off)
    pickle_off.close()
    x = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    x /= 255.0
    X_train, X_test, y_train, y_test = train_test_split(x[0:size], y[0:size], test_size=0.20, random_state=17)
    assert (X_train.shape[0] + X_test.shape[0] == size)
    return X_train, X_test, y_train, y_test


def confusion(network, X, y) -> (np.array, list):
    wrong_list = [[] for i in range(10)]
    y_pred = pd.Series(copy.copy(y), dtype=int, name="Predicted")  # copying y works as prealocation
    for i in range(X.shape[0]):
        x = np.array([X[i]]).T
        predicted = np.argmax(network.classify(x))
        y_pred.set_value(i, predicted)
    y_actu = pd.Series(y, name="Actual")
    wrong_pred_mask = y_actu != y_pred
    X_wrong = X[wrong_pred_mask]
    y_wrong = y_pred[wrong_pred_mask]
    for wrong in zip(y_wrong, X_wrong):
        wrong_list[wrong[0]].append(wrong[1].reshape(28, 28))
    return pd.crosstab(y_actu, y_pred), wrong_list


def check_accuracy(network, X_test, y_test):
    # Code duplication, yet it does not copy memory, hence it it's faster
    true_positive_counter = 0.0
    for i in range(X_test.shape[0]):
        x = np.array([X_test[i]]).T
        if np.argmax(network.classify(x)) == y_test[i]:
            true_positive_counter += 1
    return true_positive_counter / X_test.shape[0]


def pickle_network(network, file_name):
    print("Pickling Network...")
    pickling_on = open(file_name, "wb")
    pickle.dump(network, pickling_on)
    pickling_on.close()
    print("Pickling done")


def unpickle_network(file_name):
    pickle_off = open(file_name, "rb")
    net = pickle.load(pickle_off)
    pickle_off.close()
    return net


def plot_wrong(wrong_list):
    columns = 8
    fig_width = 8
    for digit in range(len(wrong_list)):
        rows = math.ceil(len(wrong_list[digit]) / columns)
        fig_height = fig_width * (rows / columns)
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle(f"Cyfry uznane za {digit}", fontsize=10)
        i = 1
        for image in wrong_list[digit]:
            fig.add_subplot(rows, columns, i)
            i += 1
            plt.imshow(image)
            plt.axis('off')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(top=0.85)
        plt.show()


def roc_curves(network, X, y):
    preds = np.empty((X.shape[0], digits_nr))
    print(preds.shape)
    for i in range(X.shape[0]):
        x = np.array([X[i]]).T
        pred = network.classify(x)
        preds[i, :] = pred.flatten()
    for i in range(digits_nr):
        digit_pred = preds[:, i]
        y_expected = y == i
        fpr, tpr, thresholds = roc_curve(y_expected, digit_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, alpha=0.9, color='r', label='ROC curve')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='g', label='Random classifier', alpha=0.4)
        plt.title(f"ROC curve for digit {i}, AOC = {roc_auc}")
        plt.legend(loc='lower right')
        plt.show()



if __name__ == "__main__":
    # if you want to perform loading data faster pickle database first
    # and then use pickled binary file (unpickle_mnist)
    # pickle_mnist("mnist.pickle")
    X_train, X_test, y_train, y_test = unpickle_mnist("mnist.pickle", 2_000)
    X_train, y_train = augment(X_train, y_train, (28, 28), augment_times)
    # plt.imshow(X_train[201].reshape(28, 28))
    # plt.show()
    # X_train, X_test, y_train, y_test = load_mnist()
    network = layer.Layer(hidden_layers=hidden_layers,
                          input_size=X_train.shape[1],
                          layer_size=layer_size,
                          output_size=digits_nr,
                          learning_rate=learning_rate,
                          init_type=init_type)
    # network = unpickle_network("network.pickle")
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
                network.apply_gradients(optimizer=optimizer)
        train_loss_values.append(check_loss(network, X_train, y_train))
        print(f"train loss: {check_loss(network, X_train, y_train)}")
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

    pickle_network(network, "network.pickle")

    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    print(f"hidden_layers: {hidden_layers}\nlayer_size: {layer_size}\n"
          f"learning_rate: {learning_rate}\ninit_type: {init_type}\nbatch_size: {batch_size}")
    print(f"train loss: {train_loss_values[-1].item()}")
    print(f"test loss: {test_loss_values[-1].item()}")
    print(f"accuracy: {accuracy[-1]}")
    confusion_matrix, wrong_list = confusion(network, X_test, y_test)
    roc_curves(network, X_test, y_test)
    print(f"confusion matrix:\n{confusion_matrix}")
    # plot_wrong(wrong_list)

    plt.subplot(211)
    plt.plot(train_loss_values, linestyle='-.', label='training')
    plt.plot(test_loss_values, linestyle='-', label='test')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.subplot(212)
    plt.plot(accuracy, linestyle='-', label='training')
    plt.show()
