"""
Run from parent directory, e.g. `pipenv run python ./src/main.py`
"""

import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

from neuralnetwork import NeuralNetwork, BatchMethod
from utils import split_binary_data, normalize


def main():
    data_frame = pd.read_csv('res/zip.train', header=None, sep=' ')
    X_train = data_frame.iloc[:, 1:-1].values
    y_train = data_frame.iloc[:, 0].values
    data_frame = pd.read_csv('res/zip.test', header=None, sep=' ')
    X_test = data_frame.iloc[:, 1:].values
    y_test = data_frame.iloc[:, 0].values

    # X_train, y_train = X_train[:2, 10:13], [6-5, 5-5]
    # X_test, y_test = X_test[:2,  10:13], [2, 1]
    # neural_network = NeuralNetwork(X_train, y_train, 3, 3, [2, 2])

    accuracies = []

    NUM_EPOCHS = 20
    neural_network = NeuralNetwork(X_train, y_train, 16*16, 10, [40, 50])
    neural_network.train(
        X_train, y_train,
        batch_size=32,
        learning_constant=1e-3,
        num_epochs=NUM_EPOCHS,
        callback=lambda weights: accuracies.append(
            neural_network.accuracy(X_test, y_test, weights=weights)
        )
    )
    # neural_network.train(X_train, y_train, batch_method=BatchMethod.ONLINE_BATCH)
    # neural_network.print_confusion_matrix(X_test[:2], y_test[:2])
    # neural_network.print_confusion_matrix(X_test, y_test)
    print(accuracies)
    plt.plot(accuracies)
    plt.axis([1, NUM_EPOCHS, 0, 1])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
