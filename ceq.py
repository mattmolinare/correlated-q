#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import random

STATE = 34


def compute_alpha(iteration, num_iterations, max_alpha, min_alpha):
    return max_alpha * (min_alpha / max_alpha) ** (iteration / num_iterations)


class CSVWriter:

    def __init__(self, file):
        self._file = file

    def __enter__(self):
        self._file = open(self._file, 'w')
        self._writer = csv.writer(self._file)
        self._writer.writerow(['iteration', 'old', 'new', 'difference'])
        return self

    def __exit__(self, type, value, traceback):
        self._file.close()

    def write(self, iteration, Q_old, Q_new, Q_diff):
        self._writer.writerow([iteration, Q_old, Q_new, Q_diff])


class State:

    nx = 4
    ny = 2

    shape = (nx, ny, nx, ny, 2)

    def __init__(self, index):
        self.index = index

    def __repr__(self):

        xa, ya, xb, yb, possession = self.tuple

        s = np.full((State.ny, State.nx), '-')
        s[[ya, yb], [xa, xb]] = list('Ab' if possession else 'aB')

        return '\n'.join(s.view('U%i' % State.nx).flat)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
        self.tuple = np.unravel_index(self.index, State.shape)

    @classmethod
    def from_tuple(cls, state_tuple):
        index = np.ravel_multi_index(state_tuple, State.shape)
        return cls(index)


class Env:

    init_state = 100

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self._state = State(Env.init_state)

    def __repr__(self):
        return repr(self._state)

    def reset(self):
        super().__init__()
        return self._state.index

    def sample_action(self):
        return self.rng.randint(0, 4)

    @staticmethod
    def take_action(coords, action):

        x, y = coords

        if action == 0:
            y = min(State.ny - 1, y + 1)
        elif action == 1:
            y = max(0, y - 1)
        elif action == 2:
            x = min(State.nx - 1, x + 1)
        elif action == 3:
            x = max(0, x - 1)
        elif action != 4:
            raise ValueError('invalid action: %i' % action)

        return (x, y)

    def step(self, action_a, action_b):

        xa, ya, xb, yb, possession = self._state.tuple

        coords_a = (xa, ya)
        coords_b = (xb, yb)

        new_coords_a = Env.take_action(coords_a, action_a)
        new_coords_b = Env.take_action(coords_b, action_b)

        if self.rng.random() > 0.5:

            # player A moves first

            if new_coords_a == coords_b:
                # A collides with B, B secures possession, no one moves
                xa, ya = coords_a
                xb, yb = coords_b
                possession = False

            elif new_coords_a == new_coords_b:
                # B collides with A, A secures possession, only A moves
                xa, ya = new_coords_a
                xb, yb = coords_b
                possession = True

        else:

            # player B moves first

            if new_coords_b == coords_a:
                # B collides with A, A secures possession, no one moves
                xa, ya = coords_a
                xb, yb = coords_b
                possession = True

            elif new_coords_a == new_coords_b:
                # A collides with B, B secures possession, only B moves
                xa, ya = coords_a
                xb, yb = new_coords_b
                possession = False

        if possession:
            if xa == 0:
                reward = 100
                game_over = True
            elif xa == State.nx - 1:
                reward = -100
                game_over = True
            else:
                reward = 0
                game_over = False
        else:
            if xb == 0:
                reward = 100
                game_over = True
            elif xb == State.nx - 1:
                reward = -100
                game_over = True
            else:
                reward = 0
                game_over = False

        self._state = State.from_tuple((xa, ya, xb, yb, possession))

        return self._state.index, reward, game_over


def q_learning(csv_writer, num_iterations, max_alpha, min_alpha, gamma,
               seed=None):

    rng = random.Random(seed)
    env = Env(rng)
    state = env.reset()

    alpha = max_alpha
    decay_rate = (min_alpha / max_alpha) ** (1 / num_iterations)

    Q = np.ones((128, 5))

    for iteration in range(1, num_iterations + 1):

        action = env.sample_action()
        opponent_action = env.sample_action()

        next_state, reward, game_over = env.step(action, opponent_action)

        alpha *= decay_rate

        Q_old = Q[state, action]
        Q_new = (1 - alpha) * Q_old + \
            alpha * (reward + gamma * Q[next_state].max())

        if (state, action) == (68, 1):  # TODO: generalize state-action
            Q_diff = abs(Q_new - Q_old)
            csv_writer.write(iteration, Q_old, Q_new, Q_diff)
            print('Update at iteration %i, Q_diff: %f' % (iteration, Q_diff))

        Q[state, action] = Q_new

        state = env.reset() if game_over else next_state

    return Q


if __name__ == '__main__':

    num_iterations = 1000000
    max_alpha = 0.2
    min_alpha = 0.001
    gamma = 0.9
    seed = 0

    csv_file = 'test.csv'
    with CSVWriter(csv_file) as writer:
        q_learning(writer, num_iterations, max_alpha, min_alpha, gamma,
                   seed=seed)
