"""
Run from parent directory, e.g. `pipenv run python ./src/main.py`
"""

import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

from neuralnetwork import NeuralNetwork, BatchMethod


def main():
    data_frame = pd.read_csv('res/zip.train', header=None, sep=' ')
    X_train = data_frame.iloc[:, 1:-1].values
    y_train = data_frame.iloc[:, 0].values
    data_frame = pd.read_csv('res/zip.test', header=None, sep=' ')
    X_test = data_frame.iloc[:, 1:].values
    y_test = data_frame.iloc[:, 0].values

    accuracies = []
    NUM_EPOCHS = 20
    neural_network = NeuralNetwork(X_train, y_train, 16*16, 10, [180, 100, 50])
    neural_network.train(
        X_train, y_train,
        batch_size=32,
        learning_constant=1e-5,
        num_epochs=NUM_EPOCHS,
        callback=lambda weights: accuracies.append(
            neural_network.accuracy(X_test, y_test, weights=weights)
        )
    )

    plt.plot(accuracies)
    plt.axis([0, NUM_EPOCHS, 0, 1])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
