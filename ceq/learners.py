# -*- coding: utf-8 -*-

import numpy as np
import random

# local imports
from . import linprog

__all__ = [
    'q_learning',
    'friend_q_learning',
    'foe_q_learning',
    'correlated_q_learning'
]


class State:

    nx = 4
    ny = 2

    shape = (nx, ny, nx, ny, 2)

    def __init__(self, index):
        self.index = index

    def __repr__(self):

        xa, ya, xb, yb, possession = self.tuple

        sa, sb = 'Ab' if possession else 'aB'

        s = np.full((State.ny, State.nx), '-')
        s[ya, xa] = sa
        s[yb, xb] = sb

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
        self._state = State(Env.init_state)
        return self._state.index

    def sample_action(self):
        return self.rng.randint(0, 4)

    @staticmethod
    def move(coords, action):

        x, y = coords

        if action == 0:
            y = max(0, y - 1)
        elif action == 1:
            y = min(State.ny - 1, y + 1)
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

        # apply actions
        new_coords_a = Env.move(coords_a, action_a)
        new_coords_b = Env.move(coords_b, action_b)

        if self.rng.random() > 0.5:

            # player A moves first

            if new_coords_a == coords_b:
                # A collides with B, no one moves, B gets ball
                xa, ya = coords_a
                xb, yb = coords_b
                possession = False

            elif new_coords_a == new_coords_b:
                # B collides with A, only A moves, A gets ball
                xa, ya = new_coords_a
                xb, yb = coords_b
                possession = True

            else:
                # no collision
                xa, ya = new_coords_a
                xb, yb = new_coords_b
        else:

            # player B moves first

            if new_coords_b == coords_a:
                # B collides with A, no one moves, A gets ball
                xa, ya = coords_a
                xb, yb = coords_b
                possession = True

            elif new_coords_b == new_coords_a:
                # A collides with B, only B moves, B gets ball
                xa, ya = coords_a
                xb, yb = new_coords_b
                possession = False

            else:
                # no collision
                xa, ya = new_coords_a
                xb, yb = new_coords_b

        # update state
        self._state = State.from_tuple((xa, ya, xb, yb, possession))

        # get reward
        x = xa if possession else xb

        if x == 0:
            # ball in goal A
            reward = 100
            game_over = True

        elif x == State.nx - 1:
            # ball in goal B
            reward = -100
            game_over = True

        else:
            # ball in play
            reward = 0
            game_over = False

        return self._state.index, reward, game_over


def q_learning(writer, num_iterations, max_alpha, min_alpha, gamma, seed=None,
               print_interval=10000):

    # initialize environment
    env = Env(seed)
    state = env.reset()

    # learning rate
    alpha = max_alpha
    decay_rate = (min_alpha / max_alpha) ** (1 / num_iterations)

    # initialize Q-table
    Q = np.ones((128, 5))

    for iteration in range(1, num_iterations + 1):

        # get actions
        action_a = env.sample_action()
        action_b = env.sample_action()

        # play game
        next_state, reward, game_over = env.step(action_a, action_b)

        # update Q-table
        state_action = (state, action_a)

        Q_old = Q[state_action]
        Q_new = (1 - alpha) * Q_old + \
            alpha * (reward + gamma * Q[next_state].max())
        Q[state_action] = Q_new

        if (state_action) == (68, 1):
            writer.write(iteration, Q_old, Q_new)

        if iteration % print_interval == 0:
            print('Iteration: %i, Learning rate: %f' % (iteration, alpha))

        # update learning rate
        alpha *= decay_rate

        # update state
        state = env.reset() if game_over else next_state

    return Q


def friend_q_learning(writer, num_iterations, max_alpha, min_alpha, gamma,
                      seed=None, print_interval=10000):

    # initialize environment
    env = Env(seed)
    state = env.reset()

    # learning rate
    alpha = max_alpha
    decay_rate = (min_alpha / max_alpha) ** (1 / num_iterations)

    # initialize Q-table
    Q = np.ones((128, 5, 5))

    for iteration in range(1, num_iterations + 1):

        # get actions
        action_a = env.sample_action()
        action_b = env.sample_action()

        # play game
        next_state, reward, game_over = env.step(action_a, action_b)

        # update Q-table
        state_action = (state, action_a, action_b)

        Q_old = Q[state_action]
        Q_new = (1 - alpha) * Q_old + \
            alpha * (reward + gamma * Q[next_state].max())
        Q[state_action] = Q_new

        if state_action == (68, 1, 4):
            writer.write(iteration, Q_old, Q_new)

        if iteration % print_interval == 0:
            print('Iteration: %i, Learning rate: %f' % (iteration, alpha))

        # update learning rate
        alpha *= decay_rate

        # update state
        state = env.reset() if game_over else next_state

    return Q


def foe_q_learning(writer, num_iterations, max_alpha, min_alpha, gamma,
                   seed=None, print_interval=1000):

    # initialize environment
    env = Env(seed)
    state = env.reset()

    # learning rate
    alpha = max_alpha
    decay_rate = (min_alpha / max_alpha) ** (1 / num_iterations)

    # initialize Q-table
    Q = np.ones((128, 5, 5))

    for iteration in range(1, num_iterations + 1):

        # get actions
        action_a = env.sample_action()
        action_b = env.sample_action()

        # play game
        next_state, reward, game_over = env.step(action_a, action_b)

        # compute value
        V, success = linprog.minimax_cvxopt_lp(Q[next_state])

        # update Q-table
        state_action = (state, action_a, action_b)

        Q_old = Q[state_action]
        Q_new = (1 - alpha) * Q_old + alpha * (reward + gamma * V)
        Q[state_action] = Q_new

        if state_action == (68, 1, 4):
            writer.write(iteration, Q_old, Q_new)

        if iteration % print_interval == 0:
            print('Iteration: %i, Learning rate: %f' % (iteration, alpha))

        # update learning rate
        alpha *= decay_rate

        # update state
        state = env.reset() if game_over else next_state


def correlated_q_learning(writer, num_iterations, max_alpha, min_alpha, gamma,
                          seed=None, print_interval=1000):

    # initialize environment
    env = Env(seed)
    state = env.reset()

    # learning rate
    alpha = max_alpha
    decay_rate = (min_alpha / max_alpha) ** (1 / num_iterations)

    # initialize Q-tables
    Qa = np.ones((128, 5, 5))
    Qb = np.ones_like(Qa)

    # initialize values
    Va = np.ones(128)
    Vb = np.ones_like(Va)

    for iteration in range(1, num_iterations + 1):

        # get actions
        action_a = env.sample_action()
        action_b = env.sample_action()

        # play game
        next_state, reward, game_over = env.step(action_a, action_b)

        # compute values
        Vsa, Vsb, success = linprog.ceq(Qa[next_state], Qb[next_state])

        if success:
            # update values
            Va[next_state] = Vsa
            Vb[next_state] = Vsb

        # update Q-tables
        state_action = (state, action_a, action_b)

        Qa_old = Qa[state_action]
        Qa_new = (1 - alpha) * Qa_old + \
            alpha * (reward + gamma * Va[next_state])
        Qa[state_action] = Qa_new

        if state_action == (68, 1, 4):
            writer.write(iteration, Qa_old, Qa_new)

        Qb[state_action] = (1 - alpha) * Qb[state_action] + \
            alpha * (-reward + gamma * Vb[next_state])

        if iteration % print_interval == 0:
            print('Iteration: %i, Learning rate: %f' % (iteration, alpha))

        # update learning rate
        alpha *= decay_rate

        # update state
        state = env.reset() if game_over else next_state
