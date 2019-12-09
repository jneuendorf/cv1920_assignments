import math
from typing import Any, Callable, List, Tuple

import numpy as np

from classifier import Classifier
import utils
from utils import hot_one_encode_ints as hoe_ints, hot_one_decode_int as hod_int


class BatchMethod:
    BATCH = 0
    MINI_BATCH = 1
    ONLINE_BATCH = 2


class NeuralNetwork(Classifier):
    learning_constant = 1e-3

    def __init__(self, X, y,
                 size_in: int, size_out: int,
                 hidden_layers: List[int],
                 hot_one_encode_y: Callable[[int, Any], np.ndarray] = hoe_ints,
                 hot_one_decode: Callable[[np.ndarray], int] = hod_int):
        """
        'size_in' Number of features.
        'size_out' Number of classes.
        'hidden_layers' Defines how many nodes each layer has.
        'hot_one_encode_y' Hot-one encodes a label.
        """
        super().__init__(X, y, num_classes=size_out)
        assert len(hidden_layers) > 0, 'Need at least 1 hidden layer.'

        self.size_in = size_in
        self.size_out = size_out
        self.hidden_layers = hidden_layers
        self.hot_one_encode_y = hot_one_encode_y
        self.hot_one_decode = hot_one_decode

    def train(self, X, y, *,
              num_epochs=10,
              batch_method=BatchMethod.MINI_BATCH,
              batch_size=None,
              learning_constant=1e-3,
              callback=None):
        N = len(X)
        if batch_method == BatchMethod.MINI_BATCH:
            if batch_size is None:
                batch_size = N // 20
        elif batch_method == BatchMethod.BATCH:
            batch_size = N
            if batch_size is not None:
                print('WARNING: batch_size given but ignored.')
        elif batch_method == BatchMethod.ONLINE_BATCH:
            batch_size = 1
            if batch_size is not None:
                print('WARNING: batch_size given but ignored.')
        else:
            raise ValueError('Invalid batch method.')

        X_shuffled = X[:]
        np.random.shuffle(X_shuffled)

        weights, augmented_weights = self._initialize_weight_matrices()
        num_batches = math.ceil(N / batch_size)
        for epoch in range(num_epochs):
            for batch_index in range(num_batches):
                batch = X_shuffled[batch_index:(batch_index + batch_size)]
                corrections = [
                    np.zeros(matrix.shape)
                    for matrix in augmented_weights
                ]
                for i, x in enumerate(batch):
                    new_corrections = self.backpropagation(
                        weights,
                        *self.feed_forward(augmented_weights, x, y[i]),
                        learning_constant=learning_constant,
                    )
                    corrections = self._sum_matrix_lists(
                        corrections,
                        new_corrections
                    )

                weights, augmented_weights = self._apply_weight_corrections(
                    augmented_weights,
                    corrections
                )
            if callable(callback):
                callback(augmented_weights)

        self.weights = augmented_weights

    def feed_forward(self, weights, x, y_i):
        # print('y_i', y_i)
        outputs = [self._O_hat(x)]
        diagonals = []
        s = utils.sigmoid
        sd = utils.sigmoid_d
        # s = utils.relu
        # sd = utils.relu_d

        # Start at 1 to match math notation.
        for i, augmented_matrix in enumerate(weights, start=1):
            O_hat_prev = outputs[i - 1]
            W = augmented_matrix
            O = s(O_hat_prev.T @ W)
            D = np.diag(sd(O))

            outputs.append(self._O_hat(O))
            diagonals.append(D)
        try:
            t = self.hot_one_encode_y(self.num_classes, int(y_i))
        except IndexError as e:
            raise ValueError((
                'Cannot hot one encode "{}" because too few outputs '
                '(change "size_out" argument for "__init__")'
            ).format(int(y_i))) from e

        # O is the final (unaugmented) output.
        error = O - t
        return outputs, diagonals, error

    def backpropagation(self, weights, outputs, diagonals, error,
                        *,
                        learning_constant):
        """
        'weights' Weight matrices W_i.
        'outputs' Augmented output vectors.
        'diagonals' Diagonal matrices D_i containing derivates
        'error' Error derivate vector e
        """
        N = len(diagonals)
        deltas = []
        i_max = N - 1
        for i in range(i_max, -1, -1):
            D = diagonals[i]
            if i == i_max:
                delta = D @ error
            else:
                W = weights[i + 1]
                delta = D @ W @ prev_delta
            # Prepend delta to keep order equal to the other variables.
            deltas.insert(0, delta)
            prev_delta = delta

        # The corrections' indices must be ascending
        # to match the order of weight matrices.
        return [
            -learning_constant * np.outer(delta, outputs[i]).T
            for i, delta in enumerate(deltas)
        ]

    def predict_label(self, x_test, weights=None):
        """
        'weights' Override self.weights, used for accuracy measurement.
        """
        if weights is None:
            weights = self.weights

        # outputs = [self._O_hat(x_test)]
        O_hat_prev = self._O_hat(x_test)
        s = utils.sigmoid

        # Start at 1 to match math notation.
        for i, augmented_matrix in enumerate(weights, start=1):
            # O_hat_prev = outputs[i - 1]
            W = augmented_matrix
            # print('>', i, O_hat_prev.T)
            # !!!!!!
            # print('>', i, O_hat_prev.T @ W)
            # print(i, W)
            O = s(O_hat_prev.T @ W)
            # outputs.append(self._O_hat(O))
            O_hat_prev = self._O_hat(O)

        # print(O, O_hat_prev)
        # print(O)
        return self.hot_one_decode(O)

    def _initialize_weight_matrices(self) -> Tuple[List[np.ndarray]]:
        matrices = []
        prev_dim_size = self.size_in
        for layer_size in self.hidden_layers:
            shape = (prev_dim_size, layer_size)
            matrix = np.random.uniform(0, 1, shape)
            matrices.append(matrix)
            prev_dim_size = layer_size

        shape = (prev_dim_size, self.size_out)
        matrix = np.random.uniform(0, 1, shape)
        matrices.append(matrix)
        return matrices, [utils.augmented(matrix) for matrix in matrices]

    def _apply_weight_corrections(self, augmented_weights, corrections):
        corrected_weights = self._sum_matrix_lists(
            augmented_weights,
            corrections
        )
        return (
            [utils.unaugmented(matrix) for matrix in corrected_weights],
            corrected_weights,
        )

    def _sum_matrix_lists(self, a, b):
        # TODO: Use zip
        if len(a) != len(b):
            raise ValueError('Unequally long lists.')
        return [a_i + b[i] for i, a_i in enumerate(a)]

    def _O_hat(self, O):
        return utils.augmented(O)
