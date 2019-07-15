# -*- coding: utf-8 -*-

import csv
import pandas as pd

__all__ = [
    'learning_rate',
    'read_csv',
    'Writer'
]


def learning_rate(iteration, num_iterations, max_alpha, min_alpha):
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


def read_csv(file):
    df = pd.read_csv(file)
    df['difference'] = (df['new'] - df['old']).abs()
    return df
