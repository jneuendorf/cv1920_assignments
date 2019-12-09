from collections import Counter
import math

import numpy as np
import numpy.linalg
import numpy.matlib


def hot_one_encode_ints(num_classes, ints):
    """
    'ints' May also be a single int.
    """
    # See https://stackoverflow.com/a/42874726/6928824
    targets = np.array(ints).reshape(-1)
    one_hot_targets = np.eye(num_classes)[targets]
    return one_hot_targets.reshape(-1)


def hot_one_decode_int(encoded):
    return np.argmax(encoded, axis=0)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.clip(x, 0, np.inf)


def relu_d(x):
    return (x >= 0).astype(float)


def augmented(array, append=True):
    """Add ones to 0-axis."""
    shape = array.shape
    ones = np.ones((1, *shape[1:]))
    if append:
        items = (array, ones)
    else:
        items = (ones, array)
    return np.concatenate(items, axis=0)


def unaugmented(array, appended=True):
    """Inverse operation to 'augmented'."""
    if appended:
        s = np.s_[:-1]
    else:
        s = np.s_[1:]
    return array[s]
