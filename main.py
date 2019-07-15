#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

# local imports
import ceq


def plot(df, **fig_kwargs):

    fig = plt.figure(**fig_kwargs)
    fig.clf()
    ax = fig.gca()
    ax.plot(df['iteration'], df['difference'], c='k', lw=0.8)

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file',
        type=str
    )
    parser.add_argument(
        'output_prefix',
        type=str,
    )
    parser.add_argument(
        '--overwrite',
        type=bool,
        default=False
    )
    args = parser.parse_args()

    output_prefix = os.path.abspath(args.output_prefix)
    output_dir = os.path.dirname(output_prefix)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    csv_file = output_prefix + '.csv'
    if os.path.isfile(csv_file) and not args.overwrite:
        raise IOError('CSV file already exists: ' + csv_file)

    with open(args.config_file, 'r') as fp:
        params = yaml.safe_load(fp)

    learner = getattr(ceq.learners, params['learner_type'])
    with ceq.Writer(csv_file) as writer:
        try:
            learner(writer, 1000000, params['max_alpha'], params['min_alpha'],
                    params['gamma'], seed=params['seed'])
        except KeyboardInterrupt:
            pass

    df = ceq.read_csv(csv_file)
    fig = plot(df)
    fig.show()
    fig.savefig(output_prefix + '.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
