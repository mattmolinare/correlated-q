#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

__all__ = ['ceq', 'minimax']


def minimax_cvxopt_lp(Qs):

    from cvxopt import matrix, solvers

    solvers.options['show_progress'] = False

    c = np.zeros((6, 1))
    c[5, 0] = -1
    c = matrix(c)

    G = np.zeros((10, 6))
    np.fill_diagonal(G[:5, :5], -1)
    np.negative(Qs.T, out=G[5:, :5])
    G[5:, 5] = 1
    G = matrix(G)

    h = matrix(np.zeros(10))

    A = np.zeros((1, 6))
    A[0, :5] = 1
    A = matrix(A)

    b = matrix(np.ones((1, 1)))

    sol = solvers.lp(c, G, h, A=A, b=b)

    return sol['x'][5]


def minimax_cvxopt_op(Qs):

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


#def ceq(Qs1, Qs2):
#
#    from cvxopt import matrix, solvers
#    from cvxopt.modeling import dot, op, variable
#
#    solvers.options['show_progress'] = False
#
#    v = variable()
#    p = variable(25)
#
#    m = np.empty((25, 2))
#    m[:, 0] = Qs1.flat
#    m[:, 1] = Qs2.flat
#    m = matrix(m)
#
#    c1 = p >= 0
#    c2 = sum(p) == 1
#    c3 =
#    c4 =
#    c5 = sum(dot(m, p)) == v
#
#    lp = op(-v, [c1, c2, c3, c4])
#    lp.solve()
#    success = lp.status == 'optimal'
#
#    p = p.value.reshape(5, 5)
#
#    Vs1 = (p * Qs1).sum()
#    Vs2 = (p * Qs2).sum()
#
#    return Vs1, Vs2, success


def minimax_scipy(Qs):

    from scipy.optimize import linprog

    c = [0, 0, 0, 0, 0, -1]

    A_ub = np.zeros((10, 6))
    np.fill_diagonal(A_ub[:5, :5], -1)
    np.negative(Qs.T, out=A_ub[5:, :5])
    A_ub[5:, 5] = 1

    b_ub = np.zeros(10)

    A_eq = [[1, 1, 1, 1, 1, 0]]

    b_eq = [1]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

    return res.x[5]


minimax = minimax_cvxopt_op
