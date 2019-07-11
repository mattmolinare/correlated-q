# -*- coding: utf-8 -*-

import csv
import numpy as np
from scipy.optimize import linprog

__all__ = [
    'compute_alpha',
    'minimax',
    'Writer'
]


def minimax(Qs):

    # coefficients of objective function
    c = [-1, 0, 0, 0, 0, 0]

    # inequality constraints
    A_ub = np.zeros((10, 6))
    np.fill_diagonal(A_ub[:, 1:], -1)
    A_ub[5:, 0] = 1
    np.negative(Qs, out=A_ub[5:, 1:])

    b_ub = np.zeros(10)

    # equality constraints
    A_eq = [[0, 1, 1, 1, 1, 1]]
    b_eq = [1]

    # minimize objective function
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

    return res


def compute_alpha(iteration, num_iterations, max_alpha, min_alpha):
    """Compute learning rate with exponential decay
    """
    return max_alpha * (min_alpha / max_alpha) ** (iteration / num_iterations)


class Writer:

    def __init__(self, file):
        self._file = file

    def __enter__(self):
        self._file = open(self._file, 'w')
        self._writer = csv.writer(self._file)
        self._writer.writerow(['iteration', 'old', 'new'])
        return self

    def __exit__(self, type, value, traceback):
        self._file.close()

    def write(self, iteration, Q_old, Q_new):
        self._writer.writerow([iteration, Q_old, Q_new])
