from abc import ABC, abstractmethod

import numpy as np


class Classifier(ABC):
    """
    Abstract superclass for all classifiers
    """

    def __init__(self, X, y, num_classes=None):
        self.X = X
        self.y = y
        self.num_classes = num_classes or len(set(y))

    @classmethod
    def trained(cls, X, y):
        instance = cls(X, y)
        instance.train(X, y)
        return instance

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict_label(self, x_test):
        pass

    def get_confusion_matrix(self, X_test, y_test, shape=None, **kwargs):
        if shape is None:
            shape = (self.num_classes, self.num_classes)
        matrix = np.zeros(shape=shape)
        for i, x in enumerate(X_test):
            true_label = y_test[i]
            predicted_label = self.predict_label(x, **kwargs)
            matrix[true_label][predicted_label] += 1
        return matrix

    def print_confusion_matrix(self, X_test, y_test):
        matrix = self.get_confusion_matrix(X_test, y_test)
        print(matrix)
        print('accuracy: {}'.format(self.accuracy(X_test, y_test, matrix)))
        return matrix

    def accuracy(self, X_test, y_test, matrix=None, **kwargs):
        if matrix is None:
            matrix = self.get_confusion_matrix(X_test, y_test, **kwargs)
        return np.sum(np.diag(matrix)) / len(X_test)
