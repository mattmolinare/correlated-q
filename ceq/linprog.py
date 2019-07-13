#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

__all__ = ['minimax', 'ceq']


def ceq(Qs1, Qs2):

    from cvxopt import matrix, solvers

    solvers.options['abstol'] = 1e-4
    solvers.options['reltol'] = 1e-4
    solvers.options['feastol'] = 1e-4
    solvers.options['show_progress'] = False
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF', 'LPX_K_MSGLEV': 0}

    c = np.zeros((26, 1))
    c[25, 0] = -1.0
    c = matrix(c)

    G = np.zeros((65, 26))

    # p >= 0
    np.fill_diagonal(G[:25, :25], -1.0)

    # player 1 rationality constraints
    m1 = Qs1 - Qs1[:, np.newaxis]

    for i in range(5):

        r1 = 25 + 4 * i
        r2 = r1 + 4

        c1 = 5 * i
        c2 = c1 + 5

        G[r1:r2, c1:c2] = np.delete(m1[i], i, axis=0)

    # player 2 rationality constraints
    m2 = Qs2.T - Qs2.T[:, np.newaxis]

    for i in range(5):

        r1 = 45 + 4 * i
        r2 = r1 + 4

        G[r1:r2, i:25:5] = np.delete(m2[i], i, axis=0)

    G = matrix(G)

    h = matrix(np.zeros((65, 1)))

    A = np.zeros((2, 26))
    A[0, :25] = 1.0
    A[1, :25] = (Qs1 + Qs2).flat
    A[1, 25] = -1.0
    A = matrix(A)

    b = np.zeros((2, 1))
    b[0] = 1
    b = matrix(b)

    sol = solvers.lp(c, G, h, A=A, b=b, solver='glpk')
    if sol['x'] is None:
        sol = solvers.lp(c, G, h, A=A, b=b)
    success = sol['status'] == 'optimal'

    p = np.array(sol['x'])[:25].reshape((5, 5))

    Vs1 = (p * Qs1).sum()
    Vs2 = (p * Qs2).sum()

    return Vs1, Vs2, success


def minimax_lp(Qs):

    from cvxopt import matrix, solvers

    solvers.options['abstol'] = 1e-4
    solvers.options['reltol'] = 1e-4
    solvers.options['feastol'] = 1e-4
    solvers.options['show_progress'] = False
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF', 'LPX_K_MSGLEV': 0}

    c = np.zeros((6, 1))
    c[5, 0] = -1.0
    c = matrix(c)

    G = np.zeros((10, 6))
    np.fill_diagonal(G[:5, :5], -1.0)
    np.negative(Qs.T, out=G[5:, :5])
    G[5:, 5] = 1.0
    G = matrix(G)

    h = matrix(np.zeros((10, 1)))

    A = np.zeros((1, 6))
    A[0, :5] = 1.0
    A = matrix(A)

    b = matrix(np.ones((1, 1)))

    sol = solvers.lp(c, G, h, A=A, b=b, solver='glpk')
    if sol['x'] is None:
        sol = solvers.lp(c, G, h, A=A, b=b)
    success = sol['status'] == 'optimal'

    V = sol['x'][5]

    return V, success


def minimax_op(Qs):

    from cvxopt import matrix, solvers
    from cvxopt.modeling import dot, op, variable

    solvers.options['show_progress'] = False

    v = variable()
    p = variable(5)

    c1 = p >= 0
    c2 = sum(p) == 1
    c3 = dot(matrix(Qs), p) >= v

    lp = op(-v, [c1, c2, c3])
    lp.solve()
    success = lp.status == 'optimal'

    V = v.value[0]

    return V, success


minimax = minimax_lp
