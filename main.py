#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# local imports
import ceq


def plot(df, **fig_kwargs):

    iteration = df['iteration']
    Q_diff = (df['new'] - df['old']).abs()

    fig = plt.figure(**fig_kwargs)
    fig.clf()
    ax = fig.gca()
    ax.plot(iteration, Q_diff, c='k', lw=0.8)

    ax.set_xlim(0, 1000000)
    ax.set_ylim(0, 0.5)

    ax.set_xticks(np.linspace(0, 1000000, 11))
    x_formatter = plt.FuncFormatter(lambda x, pos: '%i' % (x / 100000))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.annotate('x $10^5$', (1000000, -0.05), annotation_clip=False,
                ha='right')

    ax.set_yticks(np.linspace(0, 0.5, 11))
    y_formatter = plt.FuncFormatter(lambda y, pos: ('%f' % y).rstrip('0'))
    ax.yaxis.set_major_formatter(y_formatter)

    ax.tick_params(direction='in', top=True, right=True)

    ax.set_xlabel('Simulation Iteration')
    ax.set_ylabel('Q-value Difference')

    return fig


if __name__ == '__main__':

    max_alpha = 0.2
    min_alpha = 0.001
    gamma = 0.9
    seed = 0

    with ceq.Writer('test.csv') as writer:
        try:
            ceq.correlated_q_learning(writer, 1000000, max_alpha, min_alpha, gamma,
                           seed=seed)
        except KeyboardInterrupt:
            pass

    df = pd.read_csv('test.csv')
    fig = plot(df)
    fig.savefig('test.png', bbox_inches='tight')
